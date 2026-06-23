use std::{
    env,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::Duration,
};

use tauri::{Manager, State};

struct RuntimeState {
    api: Mutex<Option<Child>>,
    ollama: Mutex<Option<Child>>,
}

impl RuntimeState {
    fn new() -> Self {
        Self {
            api: Mutex::new(None),
            ollama: Mutex::new(None),
        }
    }
}

impl Drop for RuntimeState {
    fn drop(&mut self) {
        stop_child(&self.api);
        stop_child(&self.ollama);
    }
}

fn stop_child(child: &Mutex<Option<Child>>) {
    if let Ok(mut guard) = child.lock() {
        if let Some(mut process) = guard.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

#[tauri::command]
fn sidecar_health() -> Result<bool, String> {
    Ok(service_healthy("http://127.0.0.1:8765/api/health"))
}

#[tauri::command]
fn app_data_dir(app: tauri::AppHandle) -> Result<String, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.to_string_lossy().to_string())
        .map_err(|error| error.to_string())
}

fn service_healthy(url: &str) -> bool {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(800))
        .build()
        .and_then(|client| client.get(url).send())
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn start_ollama(app: &tauri::AppHandle, state: State<RuntimeState>) -> Result<(), String> {
    if service_healthy("http://127.0.0.1:11434/api/version") {
        return Ok(());
    }
    let mut guard = state.ollama.lock().map_err(|error| error.to_string())?;
    if guard.is_some() {
        return Ok(());
    }

    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;
    let models_dir = app_data.join("models").join("ollama");
    fs::create_dir_all(&models_dir).map_err(|error| error.to_string())?;
    let executable = find_ollama_executable(app);
    let working_dir = executable
        .parent()
        .filter(|path| path.exists())
        .map(Path::to_path_buf);
    let (stdout, stderr) = log_streams(&app_data, "ollama")?;

    let mut command = Command::new(&executable);
    command
        .arg("serve")
        .env("OLLAMA_HOST", "127.0.0.1:11434")
        .env("OLLAMA_MODELS", models_dir)
        .env("OLLAMA_NO_CLOUD", "1")
        .stdout(stdout)
        .stderr(stderr);
    if let Some(directory) = working_dir {
        command.current_dir(directory);
    }
    let child = command
        .spawn()
        .map_err(|error| format!("failed to start the bundled Ollama runtime: {error}"))?;
    *guard = Some(child);
    Ok(())
}

fn start_python_sidecar(app: &tauri::AppHandle, state: State<RuntimeState>) -> Result<(), String> {
    if service_healthy("http://127.0.0.1:8765/api/health") {
        return Ok(());
    }
    let mut guard = state.api.lock().map_err(|error| error.to_string())?;
    if guard.is_some() {
        return Ok(());
    }

    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;
    fs::create_dir_all(&app_data).map_err(|error| error.to_string())?;
    let (stdout, stderr) = log_streams(&app_data, "local-api")?;

    let bundled = bundled_executable(app, "api", "teacherlm-local-api.exe");
    let mut command = if let Some(executable) = bundled {
        let mut packaged = Command::new(&executable);
        if let Some(directory) = executable.parent() {
            packaged.current_dir(directory);
        }
        packaged
    } else {
        development_api_command()?
    };

    let child = command
        .env("TEACHERLM_APP_DATA_DIR", &app_data)
        .env("OLLAMA_HOST", "http://127.0.0.1:11434")
        .env("PYTHONUNBUFFERED", "1")
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .map_err(|error| format!("failed to start local API sidecar: {error}"))?;

    *guard = Some(child);
    Ok(())
}

fn bundled_executable(app: &tauri::AppHandle, directory: &str, filename: &str) -> Option<PathBuf> {
    let resources = app.path().resource_dir().ok()?;
    [
        resources.join(directory).join(filename),
        resources.join("resources").join(directory).join(filename),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
}

fn find_ollama_executable(app: &tauri::AppHandle) -> PathBuf {
    if let Some(executable) = bundled_executable(app, "ollama", "ollama.exe") {
        return executable;
    }
    if let Some(configured) = env::var_os("TEACHERLM_OLLAMA_EXE") {
        let executable = PathBuf::from(configured);
        if executable.is_file() {
            return executable;
        }
    }
    if let Some(local_app_data) = env::var_os("LOCALAPPDATA") {
        let executable = PathBuf::from(local_app_data)
            .join("Programs")
            .join("Ollama")
            .join("ollama.exe");
        if executable.is_file() {
            return executable;
        }
    }
    PathBuf::from("ollama")
}

fn development_api_command() -> Result<Command, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .ok_or("could not resolve project root")?
        .to_path_buf();
    let api_dir = project_root.join("python").join("local_api");
    let core_dir = project_root.join("python").join("teacherlm_core");
    let python_path =
        env::join_paths([api_dir.clone(), core_dir]).map_err(|error| error.to_string())?;
    let venv_python = project_root
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    let python = if venv_python.is_file() {
        venv_python
    } else {
        PathBuf::from("python")
    };
    let mut command = Command::new(python);
    command
        .arg("-m")
        .arg("uvicorn")
        .arg("local_api.main:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("8765")
        .current_dir(api_dir)
        .env("PYTHONPATH", python_path);
    Ok(command)
}

fn log_streams(app_data: &Path, name: &str) -> Result<(Stdio, Stdio), String> {
    let logs = app_data.join("logs");
    fs::create_dir_all(&logs).map_err(|error| error.to_string())?;
    let file = open_log(&logs.join(format!("{name}.log")))?;
    let error_file = file.try_clone().map_err(|error| error.to_string())?;
    Ok((Stdio::from(file), Stdio::from(error_file)))
}

fn open_log(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())
}

fn main() {
    tauri::Builder::default()
        .manage(RuntimeState::new())
        .setup(|app| {
            let handle = app.handle().clone();
            if let Err(error) = start_ollama(&handle, handle.state::<RuntimeState>()) {
                eprintln!("{error}");
            }
            if let Err(error) = start_python_sidecar(&handle, handle.state::<RuntimeState>()) {
                eprintln!("{error}");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![sidecar_health, app_data_dir])
        .run(tauri::generate_context!())
        .expect("error while running TeacherLM");
}
