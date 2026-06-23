[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$Hidden
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WindowStyle = if ($Hidden) { "Hidden" } else { "Normal" }

function Test-Endpoint([string]$Uri) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 1
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Wait-ForEndpoint([string]$Uri, [int]$TimeoutSeconds, [string]$Name) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Endpoint $Uri) {
            Write-Host "$Name is ready." -ForegroundColor Green
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not start within $TimeoutSeconds seconds. Check its terminal window for the detailed error."
}

function Start-ScriptTerminal([string]$Title, [string]$ScriptPath) {
    $command = "`$host.UI.RawUI.WindowTitle = '$Title'; & '$ScriptPath'"
    Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
        -WorkingDirectory $Root `
        -WindowStyle $WindowStyle `
        -PassThru
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

function Ensure-Ollama {
    if (Test-Endpoint "http://127.0.0.1:11434/api/version") {
        Write-Host "Ollama is ready." -ForegroundColor Green
        return
    }
    $executable = Find-OllamaExecutable
    if (-not $executable) {
        throw "Ollama is not installed. Install it or set TEACHERLM_OLLAMA_EXE, then run start_teacherlm.cmd again."
    }
    $appDataDir = if ($env:TEACHERLM_APP_DATA_DIR) { $env:TEACHERLM_APP_DATA_DIR } else { Join-Path $env:APPDATA "TeacherLM" }
    $logs = Join-Path $appDataDir "logs"
    $env:OLLAMA_HOST = "127.0.0.1:11434"
    $env:OLLAMA_MODELS = Join-Path $appDataDir "models\ollama"
    $env:OLLAMA_NO_CLOUD = "1"
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    New-Item -ItemType Directory -Path $env:OLLAMA_MODELS -Force | Out-Null
    Start-Process `
        -FilePath $executable `
        -ArgumentList "serve" `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "ollama-launcher.out.log") `
        -RedirectStandardError (Join-Path $logs "ollama-launcher.err.log") | Out-Null
    Wait-ForEndpoint "http://127.0.0.1:11434/api/version" 90 "Ollama"
}

Write-Host "Starting TeacherLM..." -ForegroundColor Cyan

Ensure-Ollama

if (-not (Test-Endpoint "http://127.0.0.1:8765/api/health")) {
    Start-ScriptTerminal "TeacherLM API" (Join-Path $PSScriptRoot "dev_api.ps1") | Out-Null
}
Wait-ForEndpoint "http://127.0.0.1:8765/api/health" 120 "TeacherLM API"

if (-not (Test-Endpoint "http://127.0.0.1:1420")) {
    Start-ScriptTerminal "TeacherLM UI" (Join-Path $PSScriptRoot "dev_desktop.ps1") | Out-Null
}
Wait-ForEndpoint "http://127.0.0.1:1420" 120 "TeacherLM UI"

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:1420"
}

Write-Host "TeacherLM is ready at http://127.0.0.1:1420" -ForegroundColor Green
