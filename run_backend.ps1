Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path ".run-logs" | Out-Null
"backend launcher started $(Get-Date -Format o)" | Out-File ".run-logs\backend.launcher.log" -Append
"python=$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" | Out-File ".run-logs\backend.launcher.log" -Append
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 *> ".run-logs\backend.combined.log"
"backend exited code=$LASTEXITCODE $(Get-Date -Format o)" | Out-File ".run-logs\backend.launcher.log" -Append
