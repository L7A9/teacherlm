use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::Duration,
};

use sha2::{Digest, Sha256};
use tauri::{Manager, State};

const OLLAMA_VERSION: &str = "v0.30.10";
const OLLAMA_DOWNLOAD_URL: &str =
    "https://github.com/ollama/ollama/releases/download/v0.30.10/ollama-windows-amd64.zip";
const OLLAMA_SHA256: &str = "9606cee7501703a0969682667def313130f99ed73f44a88a7a8efe82d4b565f0";

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

    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;
    fs::create_dir_all(&app_data).map_err(|error| error.to_string())?;
    let _ = fs::remove_file(app_data.join("runtime").join("ollama-install-error.txt"));
    let models_dir = app_data.join("models").join("ollama");
    fs::create_dir_all(&models_dir).map_err(|error| error.to_string())?;
    let executable = match find_ollama_executable(app, &app_data) {
        Some(executable) => executable,
        None => install_ollama_runtime(&app_data)?,
    };

    let mut guard = state.ollama.lock().map_err(|error| error.to_string())?;
    if guard.is_some() {
        return Ok(());
    }
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
        .map_err(|error| format!("failed to start the local Ollama runtime: {error}"))?;
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

fn find_ollama_executable(app: &tauri::AppHandle, app_data: &Path) -> Option<PathBuf> {
    if let Some(executable) = bundled_executable(app, "ollama", "ollama.exe") {
        return Some(executable);
    }
    let downloaded = app_data.join("runtime").join("ollama").join("ollama.exe");
    if downloaded.is_file() {
        return Some(downloaded);
    }
    if let Some(configured) = env::var_os("TEACHERLM_OLLAMA_EXE") {
        let executable = PathBuf::from(configured);
        if executable.is_file() {
            return Some(executable);
        }
    }
    if let Some(local_app_data) = env::var_os("LOCALAPPDATA") {
        let executable = PathBuf::from(local_app_data)
            .join("Programs")
            .join("Ollama")
            .join("ollama.exe");
        if executable.is_file() {
            return Some(executable);
        }
    }
    None
}

fn install_ollama_runtime(app_data: &Path) -> Result<PathBuf, String> {
    let runtime_root = app_data.join("runtime");
    let target = runtime_root.join("ollama");
    let archive = runtime_root.join(format!("ollama-{OLLAMA_VERSION}-windows-amd64.zip.part"));
    let staging = runtime_root.join("ollama-extracting");
    let progress = runtime_root.join("ollama-download-progress.txt");
    let error_marker = runtime_root.join("ollama-install-error.txt");
    fs::create_dir_all(&runtime_root).map_err(|error| error.to_string())?;
    let _ = fs::remove_file(&error_marker);
    let _ = fs::remove_file(&progress);
    let _ = fs::remove_file(&archive);
    remove_directory_if_present(&staging)?;

    let result = (|| -> Result<PathBuf, String> {
        download_ollama_archive(&archive, &progress)?;
        verify_sha256(&archive, OLLAMA_SHA256)?;
        extract_zip(&archive, &staging)?;
        let executable = find_named_file(&staging, "ollama.exe")
            .ok_or("the downloaded Ollama archive did not contain ollama.exe")?;
        let source = executable
            .parent()
            .ok_or("the downloaded Ollama executable had no parent directory")?
            .to_path_buf();
        remove_directory_if_present(&target)?;
        fs::rename(&source, &target)
            .map_err(|error| format!("could not install the downloaded Ollama runtime: {error}"))?;
        if staging.exists() {
            remove_directory_if_present(&staging)?;
        }
        let installed = target.join("ollama.exe");
        if !installed.is_file() {
            return Err("the installed Ollama runtime is incomplete".to_string());
        }
        Ok(installed)
    })();

    let _ = fs::remove_file(&archive);
    let _ = fs::remove_file(&progress);
    if let Err(error) = &result {
        let _ = fs::write(&error_marker, error.as_bytes());
    } else {
        let _ = fs::remove_file(&error_marker);
    }
    result
}

fn download_ollama_archive(destination: &Path, progress_path: &Path) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60 * 60 * 2))
        .build()
        .map_err(|error| format!("could not initialize the Ollama download: {error}"))?;
    let mut response = client
        .get(OLLAMA_DOWNLOAD_URL)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("could not download the local AI engine: {error}"))?;
    let total = response.content_length().unwrap_or(0);
    let mut output = File::create(destination)
        .map_err(|error| format!("could not create the Ollama download: {error}"))?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut downloaded = 0_u64;
    loop {
        let count = response
            .read(&mut buffer)
            .map_err(|error| format!("the Ollama download was interrupted: {error}"))?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| format!("could not save the Ollama download: {error}"))?;
        downloaded += count as u64;
        let _ = fs::write(progress_path, format!("{downloaded},{total}"));
    }
    output
        .sync_all()
        .map_err(|error| format!("could not finish the Ollama download: {error}"))?;
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let mut input = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        return Err("the local AI engine download failed SHA-256 verification".to_string());
    }
    Ok(())
}

fn extract_zip(archive_path: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
    let archive_file = File::open(archive_path).map_err(|error| error.to_string())?;
    let mut archive = zip::ZipArchive::new(archive_file).map_err(|error| error.to_string())?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|error| error.to_string())?;
        let relative = entry
            .enclosed_name()
            .ok_or("the Ollama archive contained an unsafe path")?
            .to_path_buf();
        let output = destination.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&output).map_err(|error| error.to_string())?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut file = File::create(&output).map_err(|error| error.to_string())?;
        std::io::copy(&mut entry, &mut file).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn find_named_file(root: &Path, filename: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(root).ok()?.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file()
            && entry
                .file_name()
                .to_string_lossy()
                .eq_ignore_ascii_case(filename)
        {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_named_file(&path, filename) {
                return Some(found);
            }
        }
    }
    None
}

fn remove_directory_if_present(path: &Path) -> Result<(), String> {
    if path.exists() {
        fs::remove_dir_all(path).map_err(|error| error.to_string())?;
    }
    Ok(())
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
            if let Err(error) = start_python_sidecar(&handle, handle.state::<RuntimeState>()) {
                eprintln!("{error}");
            }
            let ollama_handle = handle.clone();
            std::thread::spawn(move || {
                if let Err(error) =
                    start_ollama(&ollama_handle, ollama_handle.state::<RuntimeState>())
                {
                    eprintln!("{error}");
                    if let Ok(app_data) = ollama_handle.path().app_data_dir() {
                        let runtime = app_data.join("runtime");
                        let _ = fs::create_dir_all(&runtime);
                        let _ =
                            fs::write(runtime.join("ollama-install-error.txt"), error.as_bytes());
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![sidecar_health, app_data_dir])
        .run(tauri::generate_context!())
        .expect("error while running TeacherLM");
}
