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


class CudaPairPredictor:
    """Local TF classifier with the CPU CrossEncoder's one-logit sigmoid contract."""
    max_length = 512

    def __init__(self, model, tokenizer, torch_module):
        self.model, self.tokenizer, self.torch = model, tokenizer, torch_module

    def predict(self, pairs, *, batch_size, show_progress_bar=False):
        torch = self.torch
        scores = []
        torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode():
            for offset in range(0, len(pairs), batch_size):
                batch = pairs[offset:offset + batch_size]
                encoded = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                         padding=True, truncation=False, return_tensors="pt")
                if encoded["input_ids"].shape[1] > self.max_length:
                    raise ValueError("GPU classifier pair exceeds the verified 512-token window")
                encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != 1 or logits.dtype != torch.float32:
                    raise ValueError("GPU BGE must produce one FP32 classification logit")
                scores.extend(torch.sigmoid(logits[:, 0]).cpu().tolist())
        return scores

    def resource_snapshot(self):
        torch = self.torch
        torch.cuda.synchronize(0)
        free, total = torch.cuda.mem_get_info(0)
        return {"device": "cuda:0", "allocated_bytes": torch.cuda.memory_allocated(0),
                "reserved_bytes": torch.cuda.memory_reserved(0),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "free_bytes": free, "total_bytes": total}


def load_verified_gpu_reranker(home):
    """Called only inside the campaign's current-boot lease and global GPU lock."""
    import importlib.metadata
    directory, receipt = verified_checkpoint(home)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("GPU R3 requires exactly the leased single visible device 0")
    import torch
    if torch.version.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("A single usable CUDA device is required")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    if torch.cuda.mem_get_info(0)[0] < 3 * 1024**3:
        raise MemoryError("GPU R3 requires 3 GiB free VRAM before weights")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager")
    if model.config.num_labels != 1:
        raise ValueError("Pinned BGE classifier must have exactly one label")
    model = model.to("cuda:0").eval()
    if any(p.dtype != torch.float32 or p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("GPU BGE parameters are not uniformly CUDA FP32")
    predictor = CudaPairPredictor(model, tokenizer, torch)
    from ragscope.retrieval.reranker import Reranker
    reranker = Reranker(str(directory), revision=REVISION, device="cuda", max_length=512,
                        window_overlap=32, batch_size=2, model=predictor, tokenizer=tokenizer)
    reranker.identity.update(checkpoint_manifest=receipt, dtype="torch.float32", cpu_threads=4,
                             torch_version=torch.__version__, transformers_version=importlib.metadata.version("transformers"),
                             cuda_version=torch.version.cuda, backend="transformers_sequence_classifier",
                             activation="sigmoid", attention_implementation="eager", tf32=False,
                             float32_matmul_precision="highest", autocast=False,
                             gpu_name=torch.cuda.get_device_name(0))
    return reranker
