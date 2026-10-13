"""One pinned local BGE identity shared by materialization and online execution."""
from __future__ import annotations

import os
from pathlib import Path
import json

from ragscope.indexing.identity import artifact, file_hash, verify_artifacts

REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
WEIGHT_SHA256 = "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd"


def verified_checkpoint(home):
    directory = Path(home) / "models/bge-reranker-base" / REVISION
    manifest_path = directory / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"config.json", "model.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json",
                "tokenizer.json", "tokenizer_config.json"}
    items = manifest["files"]
    names = {Path(item["path"]).name for item in items}
    if (manifest.get("status") != "VERIFIED" or manifest.get("model") != "BAAI/bge-reranker-base"
            or manifest.get("revision") != REVISION or not required <= names or len(names) != len(items)
            or any(Path(item["path"]).resolve().parent != directory.resolve() for item in items)
            or not verify_artifacts(items)):
        raise ValueError("Pinned BGE reranker checkpoint identity/content mismatch")
    if file_hash(directory / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("BGE reranker weight SHA differs from audited official revision")
    return directory, artifact(manifest_path)


def load_verified_reranker(home, *, device="cpu"):
    """Load only after the caller owns resource handoff; never start any service."""
    directory, receipt = verified_checkpoint(home)
    if device != "cpu":
        raise ValueError("This audited loader currently permits CPU only")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only runtime environment is required")
    import torch
    if torch.version.cuda is not None:
        raise ValueError("Use the isolated CPU-only models environment")
    torch.set_num_threads(4)
    from ragscope.retrieval.reranker import Reranker
    reranker = Reranker(str(directory), revision=REVISION, device=device, max_length=512,
                        window_overlap=32, batch_size=2, local_files_only=True)
    reranker.identity.update(checkpoint_manifest=receipt, dtype=str(next(reranker.model.model.parameters()).dtype),
                             torch_version=torch.__version__, cpu_threads=4)
    if str(next(reranker.model.model.parameters()).device) != "cpu":
        raise ValueError("Reranker backend is not the declared CPU device")
    return reranker
