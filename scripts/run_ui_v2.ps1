param([ValidateRange(1024,65535)][int]$Port=8501,
      [ValidateSet('auto','core','models')][string]$Environment='auto', [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'runtime_env.ps1')
$selectedEnvironment = $Environment
if ($selectedEnvironment -eq 'auto') {
    $selectedEnvironment = 'core'
    $modelsPython = Join-Path $env:RAGSCOPE_HOME 'envs/models/Scripts/python.exe'
    if (Test-Path -LiteralPath $modelsPython -PathType Leaf) {
        & $modelsPython -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("streamlit") else 2)'
        if ($LASTEXITCODE -eq 0) { $selectedEnvironment = 'models' }
    }
}
$pythonPath = Join-Path $env:RAGSCOPE_HOME "envs/$selectedEnvironment/Scripts/python.exe"
if (!(Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Requested $selectedEnvironment environment is missing; core bootstrap is scripts/bootstrap_v2.ps1."
}
$portProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
try { $portProbe.Start() }
catch { throw "Loopback port $Port is unavailable. No process was stopped and no alternate interface was used." }
finally { $portProbe.Stop() }
& $pythonPath -c 'import importlib.util; assert importlib.util.find_spec("streamlit"), "Streamlit is missing"'
if ($LASTEXITCODE -ne 0) { throw 'Streamlit dependency check failed.' }
if ($CheckOnly) {
    Write-Output "UI_CHECK_PASS: http://127.0.0.1:$Port ; $selectedEnvironment environment found; no model loaded."
    exit 0
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m streamlit run ui/chat_app.py --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
    $uiExitCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $uiExitCode
