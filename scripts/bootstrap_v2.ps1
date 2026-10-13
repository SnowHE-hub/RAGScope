$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/runtime_env.ps1"
Set-Location (Split-Path $PSScriptRoot -Parent)
uv python install 3.12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv venv --python 3.12 "$env:RAGSCOPE_HOME/envs/core"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv pip sync --python "$env:RAGSCOPE_HOME/envs/core/Scripts/python.exe" requirements-v2.lock
exit $LASTEXITCODE
