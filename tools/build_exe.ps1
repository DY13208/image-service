$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $Root "tools\\launcher.py"
$Py = Join-Path $Root ".venv-ui\\Scripts\\python.exe"
$OutDir = Join-Path $Root "dist"
$BuildDir = Join-Path $Root "build"

if (!(Test-Path $Launcher)) { throw "Missing launcher: $Launcher" }
if (!(Test-Path $Py)) { throw "Missing UI venv python: $Py" }

$null = & $Py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
  Write-Host "PyInstaller not installed. Run:" -ForegroundColor Yellow
  Write-Host "  $Py -m pip install pyinstaller" -ForegroundColor Yellow
  exit 1
}

& $Py -m PyInstaller `
  --onefile `
  --name image-service `
  --clean `
  --distpath $OutDir `
  --workpath $BuildDir `
  --specpath $BuildDir `
  $Launcher

Write-Host "Built: $OutDir\\image-service.exe" -ForegroundColor Green
