[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [string]$Python = "python",
    [string]$OllamaVersion = "v0.30.10",
    [switch]$SkipTests,
    [switch]$SkipOllamaDownload
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path (Join-Path $Root "..")).Path
$Work = Join-Path $Root ".packaging"
$Resources = Join-Path $Root "rust\tauri_shell\resources"
$ApiResources = Join-Path $Resources "api"
$OllamaResources = Join-Path $Resources "ollama"
$Release = Join-Path $Root "release"
$Desktop = Join-Path $Root "apps\desktop"
$Tauri = Join-Path $Root "rust\tauri_shell"

function Assert-WithinRoot([string]$Path, [string]$AllowedRoot) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullRoot = [System.IO.Path]::GetFullPath($AllowedRoot).TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside $AllowedRoot`: $fullPath"
    }
}

function Reset-Directory([string]$Path) {
    Assert-WithinRoot $Path $Root
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Invoke-Checked([scriptblock]$Command, [string]$FailureMessage) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

Write-Host "Building TeacherLM $Version for Windows"
New-Item -ItemType Directory -Path $Work -Force | Out-Null
New-Item -ItemType Directory -Path $Release -Force | Out-Null
Reset-Directory $ApiResources
Reset-Directory $OllamaResources
New-Item -ItemType File -Path (Join-Path $ApiResources ".gitkeep") -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $OllamaResources ".gitkeep") -Force | Out-Null

$BuildVenv = Join-Path $Work "venv"
if (-not (Test-Path -LiteralPath (Join-Path $BuildVenv "Scripts\python.exe"))) {
    Invoke-Checked { & $Python -m venv $BuildVenv } "Could not create the packaging virtual environment."
}
$BuildPython = Join-Path $BuildVenv "Scripts\python.exe"
Invoke-Checked { & $BuildPython -m pip install --upgrade pip } "Could not update pip."
Invoke-Checked {
    & $BuildPython -m pip install -r (Join-Path $Root "python\local_api\requirements.txt") -r (Join-Path $PSScriptRoot "requirements-build.txt")
} "Could not install the Windows build dependencies."

if (-not $SkipTests) {
    Invoke-Checked { & $BuildPython -m pytest (Join-Path $Root "python\local_api\tests") -q } "Python tests failed."
}

$PyInstallerWork = Join-Path $Work "pyinstaller"
Reset-Directory $PyInstallerWork
Invoke-Checked {
    & $BuildPython -m PyInstaller --noconfirm --clean --workpath (Join-Path $PyInstallerWork "work") --distpath (Join-Path $PyInstallerWork "dist") (Join-Path $PSScriptRoot "teacherlm-local-api.spec")
} "The local API executable could not be built."
Copy-Item -Path (Join-Path $PyInstallerWork "dist\teacherlm-local-api\*") -Destination $ApiResources -Recurse -Force

if (-not $SkipOllamaDownload) {
    $OllamaArchive = Join-Path $Work "ollama-windows-amd64.zip"
    $ExpectedHash = $env:TEACHERLM_OLLAMA_SHA256
    $CustomUrl = $env:TEACHERLM_OLLAMA_URL
    if ($CustomUrl) {
        if (-not $ExpectedHash) {
            throw "TEACHERLM_OLLAMA_SHA256 is required when TEACHERLM_OLLAMA_URL is customized."
        }
        $DownloadUrl = $CustomUrl
    } else {
        $Headers = @{ "User-Agent" = "TeacherLM-Windows-Builder" }
        $OllamaRelease = Invoke-RestMethod -Headers $Headers -Uri "https://api.github.com/repos/ollama/ollama/releases/tags/$OllamaVersion"
        $Asset = $OllamaRelease.assets | Where-Object { $_.name -eq "ollama-windows-amd64.zip" } | Select-Object -First 1
        if (-not $Asset) {
            throw "The official Ollama Windows standalone archive was not found."
        }
        $DownloadUrl = $Asset.browser_download_url
        if (-not $ExpectedHash -and $Asset.digest -match '^sha256:([a-fA-F0-9]{64})$') {
            $ExpectedHash = $Matches[1]
        }
    }
    if (-not $ExpectedHash) {
        throw "No SHA-256 digest was available for the Ollama archive. Set TEACHERLM_OLLAMA_SHA256 explicitly."
    }
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $OllamaArchive
    $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $OllamaArchive).Hash
    if ($ActualHash -ne $ExpectedHash) {
        throw "The Ollama archive failed SHA-256 verification."
    }
    $OllamaExtract = Join-Path $Work "ollama-extracted"
    Reset-Directory $OllamaExtract
    Expand-Archive -LiteralPath $OllamaArchive -DestinationPath $OllamaExtract -Force
    $OllamaExe = Get-ChildItem -Path $OllamaExtract -Recurse -Filter "ollama.exe" | Select-Object -First 1
    if (-not $OllamaExe) {
        throw "The downloaded Ollama archive did not contain ollama.exe."
    }
    Copy-Item -Path (Join-Path $OllamaExe.Directory.FullName "*") -Destination $OllamaResources -Recurse -Force
}

Push-Location $Desktop
try {
    Invoke-Checked { & npm.cmd ci } "npm dependencies could not be installed."
    if (-not $SkipTests) {
        Invoke-Checked { & npm.cmd test } "Desktop tests failed."
    }
    Invoke-Checked { & npm.cmd run build } "The desktop frontend could not be built."

}
finally {
    Pop-Location
}

$ReleaseConfig = Join-Path $Work "tauri-release.json"
$ReleaseConfigData = @{ version = $Version }
if ($env:TAURI_WINDOWS_CERTIFICATE_THUMBPRINT) {
    $ReleaseConfigData.bundle = @{
        windows = @{
            certificateThumbprint = $env:TAURI_WINDOWS_CERTIFICATE_THUMBPRINT
            digestAlgorithm = "sha256"
            timestampUrl = "http://timestamp.digicert.com"
        }
    }
}
$ReleaseConfigData | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ReleaseConfig -Encoding utf8
$TauriCli = Join-Path $Desktop "node_modules\.bin\tauri.cmd"
Push-Location $Tauri
try {
    Invoke-Checked {
        & $TauriCli build --config $ReleaseConfig
    } "The TeacherLM Windows installer could not be built."
}
finally {
    Pop-Location
}

$Installer = Get-ChildItem -Path (Join-Path $Tauri "target\release\bundle\nsis") -Filter "*.exe" | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
if (-not $Installer) {
    throw "Tauri completed without producing an NSIS installer."
}
$StableInstaller = Join-Path $Release "TeacherLM-Setup.exe"
$VersionedInstaller = Join-Path $Release "TeacherLM-Setup-$Version.exe"
Copy-Item -LiteralPath $Installer.FullName -Destination $StableInstaller -Force
Copy-Item -LiteralPath $Installer.FullName -Destination $VersionedInstaller -Force
$Manifest = @{
    version = $Version
    filename = "TeacherLM-Setup.exe"
    size_bytes = (Get-Item -LiteralPath $StableInstaller).Length
    sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $StableInstaller).Hash.ToLowerInvariant()
    created_at = [DateTime]::UtcNow.ToString("o")
}
$Manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Release "release-manifest.json") -Encoding utf8

Write-Host "Installer ready: $StableInstaller"
