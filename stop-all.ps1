$ErrorActionPreference = "SilentlyContinue"
$Ports = @(8090, 7860)

function Stop-ProcessByPort([int]$port) {
  $conns = Get-NetTCPConnection -LocalPort $port -State Listen
  if (-not $conns) {
    Write-Host "Port $port not in use." -ForegroundColor DarkGray
    return
  }
  foreach ($c in $conns) {
    $pid = $c.OwningProcess
    if ($pid -and $pid -ne 0) {
      $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
      if ($proc) {
        Write-Host "Stopping $($proc.ProcessName) (PID=$pid) on port $port" -ForegroundColor Yellow
        Stop-Process -Id $pid -Force
      }
    }
  }
}

foreach ($p in $Ports) { Stop-ProcessByPort $p }
Write-Host "All stopped." -ForegroundColor Green
