param([string]$Profile='full', [string]$Campaign='20261014_v2', [switch]$CpuOnly, [switch]$Resume, [string[]]$Task)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/runtime_env.ps1"
Set-Location (Split-Path $PSScriptRoot -Parent)
$arguments = @('-u','-m','ragscope.experiments.run','--profile',$Profile,'--campaign',$Campaign)
if ($CpuOnly) { $arguments += '--cpu-only' }
if ($Resume) { $arguments += '--resume' }
foreach ($item in $Task) { $arguments += @('--task',$item) }
& "$env:RAGSCOPE_HOME/envs/core/Scripts/python.exe" @arguments
exit $LASTEXITCODE
