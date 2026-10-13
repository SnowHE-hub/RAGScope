param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/runtime_env.ps1"
Set-Location (Split-Path $PSScriptRoot -Parent)
$uvCommand = Get-Command uv -ErrorAction Stop
$pythonExe = Join-Path $env:RAGSCOPE_HOME 'envs/core/Scripts/python.exe'
$environmentPath = Join-Path $env:RAGSCOPE_HOME 'envs/core'
if (Test-Path -LiteralPath $environmentPath) {
 if (!(Test-Path -LiteralPath $pythonExe)) { throw "Existing environment directory is incomplete; preserving it: $environmentPath" }
 # An existing environment is inspected, never silently recreated or synced.
 & $uvCommand.Source pip sync --dry-run --require-hashes --python $pythonExe requirements-v2.lock
 if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
 & $uvCommand.Source pip check --python $pythonExe
 exit $LASTEXITCODE
}
if ($CheckOnly) { throw 'Core environment is absent; omit -CheckOnly for first installation' }
& $uvCommand.Source python install 3.12.13
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $uvCommand.Source venv --python 3.12.13 $environmentPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $uvCommand.Source pip sync --require-hashes --python $pythonExe requirements-v2.lock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $uvCommand.Source pip check --python $pythonExe
exit $LASTEXITCODE
