$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location $projectRoot
try {
    & $pythonPath -m streamlit run frontend/app.py --server.port 8502 --theme.base dark --theme.primaryColor '#E5A93C' --theme.backgroundColor '#0B0D13' --theme.secondaryBackgroundColor '#151926' --theme.textColor '#EDEDED' --theme.font 'sans serif'
} finally {
    Pop-Location
}
