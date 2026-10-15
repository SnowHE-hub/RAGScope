param([ValidateRange(1024,65535)][int]$Port=8502, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'runtime_env.ps1')
$pythonPath = Join-Path $env:RAGSCOPE_HOME 'envs/models/Scripts/python.exe'
if (!(Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Verified models environment is missing.' }
$portProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
try { $portProbe.Start() }
catch { throw "Loopback port $Port is unavailable; no other process was stopped." }
finally { $portProbe.Stop() }
& $pythonPath -c 'import importlib.util; assert importlib.util.find_spec("streamlit"), "Streamlit is missing"'
if ($LASTEXITCODE -ne 0) { throw 'Streamlit dependency check failed; no automatic installation.' }
if ($CheckOnly) { Write-Output "UI_V3_CHECK_PASS: http://127.0.0.1:$Port ; models remain unloaded."; exit 0 }
Push-Location -LiteralPath $projectRoot
try {
    $appEntry = if (Test-Path -LiteralPath (Join-Path $projectRoot 'configs/v3/default.json')) { 'ui/full_workbench.py' } else { 'ui/chat_app_v3.py' }
    & $pythonPath -m streamlit run $appEntry --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
    $uiExitCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $uiExitCode
