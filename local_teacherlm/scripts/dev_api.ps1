$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = (Join-Path $Root "python\local_api") + ";" + (Join-Path $Root "python\teacherlm_core")
$ProjectPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $ProjectPython) { $ProjectPython } else { (Get-Command python).Source }
$AppDataDir = if ($env:TEACHERLM_APP_DATA_DIR) { $env:TEACHERLM_APP_DATA_DIR } else { Join-Path $env:APPDATA "TeacherLM" }
$env:TEACHERLM_APP_DATA_DIR = $AppDataDir
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_MODELS = Join-Path $AppDataDir "models\ollama"
$env:OLLAMA_NO_CLOUD = "1"

function Test-OllamaReady {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 1
        return [bool]$response.version
    }
    catch {
        return $false
    }
}

function Find-OllamaExecutable {
    if ($env:TEACHERLM_OLLAMA_EXE -and (Test-Path -LiteralPath $env:TEACHERLM_OLLAMA_EXE)) {
        return (Resolve-Path -LiteralPath $env:TEACHERLM_OLLAMA_EXE).Path
    }
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $standardPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path -LiteralPath $standardPath) {
        return $standardPath
    }
    return $null
}

$OllamaProcess = $null
if (-not (Test-OllamaReady)) {
    $OllamaExecutable = Find-OllamaExecutable
    if (-not $OllamaExecutable) {
        throw "Ollama is not installed. Install it or set TEACHERLM_OLLAMA_EXE, then run this script again."
    }
    $Logs = Join-Path $AppDataDir "logs"
    New-Item -ItemType Directory -Path $Logs -Force | Out-Null
    New-Item -ItemType Directory -Path $env:OLLAMA_MODELS -Force | Out-Null
    $OllamaProcess = Start-Process `
        -FilePath $OllamaExecutable `
        -ArgumentList "serve" `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Logs "ollama-dev.out.log") `
        -RedirectStandardError (Join-Path $Logs "ollama-dev.err.log") `
        -PassThru

    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if (Test-OllamaReady) {
            break
        }
        if ($OllamaProcess.HasExited) {
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-OllamaReady)) {
        $details = Get-Content (Join-Path $Logs "ollama-dev.err.log") -Tail 20 -ErrorAction SilentlyContinue
        if (-not $OllamaProcess.HasExited) {
            Stop-Process -Id $OllamaProcess.Id -Force
        }
        throw "Ollama could not start. $($details -join ' ')"
    }
    Write-Host "Ollama is ready on http://127.0.0.1:11434"
}

try {
    & $Python -m uvicorn local_api.main:app --host 127.0.0.1 --port 8765 --reload
}
finally {
    if ($OllamaProcess -and -not $OllamaProcess.HasExited) {
        Stop-Process -Id $OllamaProcess.Id -Force
    }
}
