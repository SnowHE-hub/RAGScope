"""Process-local paths and CPU resource policy. No model imports or global changes."""
from __future__ import annotations
import os
import shutil
from pathlib import Path

CACHE_PATHS = {
    "HF_HOME": "cache/hf", "HF_HUB_CACHE": "cache/hf/hub", "HF_DATASETS_CACHE": "cache/hf/datasets",
    "TORCH_HOME": "cache/torch", "SENTENCE_TRANSFORMERS_HOME": "cache/hf/sentence_transformers",
    "PIP_CACHE_DIR": "cache/pip", "UV_CACHE_DIR": "cache/uv", "UV_PYTHON_INSTALL_DIR": "tools/python",
    "TIKTOKEN_CACHE_DIR": "cache/tiktoken", "NLTK_DATA": "cache/nltk", "XDG_CACHE_HOME": "cache",
    "TRITON_CACHE_DIR": "cache/triton", "CUDA_CACHE_PATH": "cache/cuda", "RAGSCOPE_CACHE_DIR": "cache/ragscope",
    "TORCH_EXTENSIONS_DIR": "cache/torch_extensions", "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "MPLCONFIGDIR": "cache/matplotlib",
    "TEMP": "tmp", "TMP": "tmp", "TMPDIR": "tmp",
}


def runtime_home() -> Path:
    return Path(os.environ.get("RAGSCOPE_HOME", "D:/RAGScope_Runtime" if os.name == "nt" else "/mnt/d/RAGScope_Runtime")).resolve()


def configure(cpu_only: bool = True) -> dict:
    root = runtime_home()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["RAGSCOPE_HOME"] = str(root)
    for name, relative in CACHE_PATHS.items():
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    threads = max(1, min(4, (os.cpu_count() or 1) // 4))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1")
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return {"home": str(root), "threads": threads, "cpu_only": cpu_only,
            "paths": {name: os.environ[name] for name in CACHE_PATHS}}


def require_disk(additional_bytes: int = 0) -> dict:
    usage = shutil.disk_usage(runtime_home())
    reserve = max(25 * 1024**3, usage.total // 10)
    if usage.free - additional_bytes < reserve:
        raise RuntimeError(f"Disk reserve: free={usage.free}, required={reserve + additional_bytes}")
    return {"free": usage.free, "total": usage.total, "reserve": reserve}


def require_ram(additional_bytes: int, reserve_bytes: int = 1024**3) -> dict:
    import psutil
    memory = psutil.virtual_memory()
    if memory.available < additional_bytes + reserve_bytes:
        raise RuntimeError(f"RAM budget: available={memory.available}, required={additional_bytes + reserve_bytes}")
    return {"available": memory.available, "reserved": reserve_bytes, "requested": additional_bytes}
