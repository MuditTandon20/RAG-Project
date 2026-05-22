Set-Location $PSScriptRoot
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 *> ".run-logs\frontend.combined.log"
