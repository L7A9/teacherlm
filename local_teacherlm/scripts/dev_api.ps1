$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = (Join-Path $Root "python\local_api") + ";" + (Join-Path $Root "python\teacherlm_core")
$ProjectPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $ProjectPython) { $ProjectPython } else { (Get-Command python).Source }
& $Python -m uvicorn local_api.main:app --host 127.0.0.1 --port 8765 --reload

