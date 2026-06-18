use std::{
    env,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
};

use tauri::{Manager, State};

struct SidecarState(Mutex<Option<Child>>);

impl Drop for SidecarState {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
            }
        }
    }
}

#[tauri::command]
fn sidecar_health() -> Result<bool, String> {
    let response = reqwest::blocking::get("http://127.0.0.1:8765/api/health")
        .map_err(|error| error.to_string())?;
    Ok(response.status().is_success())
}

#[tauri::command]
fn app_data_dir(app: tauri::AppHandle) -> Result<String, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.to_string_lossy().to_string())
        .map_err(|error| error.to_string())
}

fn start_python_sidecar(app: &tauri::AppHandle, state: State<SidecarState>) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|error| error.to_string())?;
    if guard.is_some() {
        return Ok(());
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .ok_or("could not resolve project root")?
        .to_path_buf();
    let api_dir = project_root.join("python").join("local_api");
    let core_dir = project_root.join("python").join("teacherlm_core");
    let python_path = env::join_paths([api_dir.clone(), core_dir])
        .map_err(|error| error.to_string())?;
    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;

    let child = Command::new("python")
        .arg("-m")
        .arg("uvicorn")
        .arg("local_api.main:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("8765")
        .current_dir(api_dir)
        .env("PYTHONPATH", python_path)
        .env("TEACHERLM_APP_DATA_DIR", app_data)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("failed to start local API sidecar: {error}"))?;

    *guard = Some(child);
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(SidecarState(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();
            let state = handle.state::<SidecarState>();
            start_python_sidecar(&handle, state).map_err(|error| {
                Box::<dyn std::error::Error>::from(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    error,
                ))
            })?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![sidecar_health, app_data_dir])
        .run(tauri::generate_context!())
        .expect("error while running TeacherLM");
}
