$ErrorActionPreference = "Stop"

# ---- Config ----
$UiVenv = ".\.venv-ui\Scripts\Activate.ps1"
$LamaVenv = ".\.venv-lama\Scripts\Activate.ps1"

$LamaHost = "0.0.0.0"
$LamaPort = 8090
$LamaModel = "lama"
$PreferDevice = "cuda"   # cuda/cpu

$GradioPort = 7860
$AppFile = ".\app.py"
$OpenBrowser = $true

function Test-PortInUse([int]$port) {
  try { return (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue) -ne $null }
  catch { return $false }
}

function Start-NewWindow([string]$cmd) {
  Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd -WindowStyle Normal
}

if (-not (Test-Path $UiVenv)) { throw "Missing UI venv activate: $UiVenv" }
if (-not (Test-Path $LamaVenv)) { throw "Missing LAMA venv activate: $LamaVenv" }
if (-not (Test-Path $AppFile)) { throw "Missing app: $AppFile" }

# ---- Start lama-cleaner (in .venv-lama) ----
if (Test-PortInUse $LamaPort) {
  Write-Host "!! Port $LamaPort already in use, skip lama-cleaner" -ForegroundColor Yellow
} else {
  # CUDA check inside LAMA venv
  $checkCudaCmd = "cd `"$PWD`"; . `"$LamaVenv`"; python -c `"import torch; print('1' if torch.cuda.is_available() else '0')`""
  $cudaOk = (powershell -NoProfile -Command $checkCudaCmd).Trim()
  $deviceToUse = $PreferDevice
  if ($PreferDevice -eq "cuda" -and $cudaOk -ne "1") { $deviceToUse = "cpu" }

  Write-Host "==> Starting lama-cleaner on 0.0.0.0:$LamaPort (device=$deviceToUse)" -ForegroundColor Cyan
  $lamaCmd = "cd `"$PWD`"; . `"$LamaVenv`"; lama-cleaner --host $LamaHost --port $LamaPort --device=$deviceToUse --model=$LamaModel"
  Start-NewWindow $lamaCmd
}

# ---- Start Gradio (in .venv-ui) ----
if (Test-PortInUse $GradioPort) {
  Write-Host "!! Port $GradioPort already in use, skip Gradio" -ForegroundColor Yellow
} else {
  Write-Host "==> Starting Gradio on 0.0.0.0:$GradioPort" -ForegroundColor Cyan
  $gradioCmd = "cd `"$PWD`"; . `"$UiVenv`"; `$env:LAMA_SERVER='http://127.0.0.1:$LamaPort'; python `"$AppFile`""
  Start-NewWindow $gradioCmd
}

if ($OpenBrowser) {
  Start-Sleep -Seconds 2
  Start-Process "http://127.0.0.1:$GradioPort"
}
