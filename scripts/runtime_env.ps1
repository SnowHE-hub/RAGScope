param([switch]$Gpu)
$ErrorActionPreference = 'Stop'
$env:RAGSCOPE_HOME = 'D:\RAGScope_Runtime'
$paths = @{
 HF_HOME='cache/hf'; HF_HUB_CACHE='cache/hf/hub'; HF_DATASETS_CACHE='cache/hf/datasets';
 TORCH_HOME='cache/torch'; SENTENCE_TRANSFORMERS_HOME='cache/hf/sentence_transformers';
 PIP_CACHE_DIR='cache/pip'; UV_CACHE_DIR='cache/uv'; UV_PYTHON_INSTALL_DIR='tools/python';
 TIKTOKEN_CACHE_DIR='cache/tiktoken'; NLTK_DATA='cache/nltk'; XDG_CACHE_HOME='cache';
 TRITON_CACHE_DIR='cache/triton'; CUDA_CACHE_PATH='cache/cuda'; RAGSCOPE_CACHE_DIR='cache/ragscope';
 TORCH_EXTENSIONS_DIR='cache/torch_extensions'; TORCHINDUCTOR_CACHE_DIR='cache/torchinductor';
 TEMP='tmp'; TMP='tmp'; TMPDIR='tmp'
}
foreach ($entry in $paths.GetEnumerator()) {
 $value = Join-Path $env:RAGSCOPE_HOME $entry.Value
 New-Item -ItemType Directory -Force -Path $value | Out-Null
 [Environment]::SetEnvironmentVariable($entry.Key,$value,'Process')
}
$env:PYTHONUTF8='1'; $env:PYTHONUNBUFFERED='1'; $env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'; $env:NUMEXPR_NUM_THREADS='4'
$env:TOKENIZERS_PARALLELISM='false'; $env:HF_HUB_DISABLE_TELEMETRY='1'; $env:DO_NOT_TRACK='1'
if (!$Gpu) { $env:CUDA_VISIBLE_DEVICES='' }
else {
 $leasePath = Join-Path $env:RAGSCOPE_HOME 'runs/20261014_v2/gpu_lease.json'
 $lease = Get-Content -LiteralPath $leasePath -Raw | ConvertFrom-Json
 if ($lease.status -ne 'ACQUIRED' -or $lease.safety_verified -ne $true) { throw 'GPU lease is not acquired' }
 $env:CUDA_VISIBLE_DEVICES = [string]$lease.gpu_device
}
