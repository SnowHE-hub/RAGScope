param([string]$Profile='full', [string]$Campaign='20261014_v2', [switch]$CpuOnly, [switch]$Resume,
      [string[]]$Task, [switch]$Gpu, [switch]$ReconcileOnly, [string[]]$CoordinatedTask)
$ErrorActionPreference = 'Stop'
if ($CpuOnly -and $Gpu) { throw '-CpuOnly and -Gpu cannot be combined' }
. "$PSScriptRoot/runtime_env.ps1" -Gpu:$Gpu
Set-Location (Split-Path $PSScriptRoot -Parent)
$arguments = @('-u','-m','ragscope.experiments.run','--profile',$Profile,'--campaign',$Campaign)
if ($CpuOnly) { $arguments += '--cpu-only' }
if ($Resume) { $arguments += '--resume' }
if ($Gpu) { $arguments += '--allow-gpu-tasks' }
if ($ReconcileOnly) { $arguments += '--reconcile-only' }
foreach ($item in $Task) { $arguments += @('--task',$item) }
foreach ($item in $CoordinatedTask) { $arguments += @('--coordinated-task',$item) }
& "$env:RAGSCOPE_HOME/envs/core/Scripts/python.exe" @arguments
exit $LASTEXITCODE
