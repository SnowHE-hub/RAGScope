"""Atomic complete training checkpoints with explicit project ownership."""
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path

from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts


class CheckpointStore:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_hash = signature(config)
        self.marker = self.root / "training_owner.json"
        identity = {"owner": "ragscope-v2-lora", "config_hash": self.config_hash}
        if self.marker.exists():
            if json.loads(self.marker.read_text(encoding="utf-8")) != identity:
                raise ValueError("Checkpoint directory belongs to different inputs")
        elif any(self.root.iterdir()):
            raise ValueError("Refuse to adopt a nonempty unowned checkpoint directory")
        else:
            atomic_json(self.marker, identity)

    def save(self, step, state, write_payload, *, best=False):
        required = {"step", "epoch", "data_position", "data_order", "rng", "optimizer", "scheduler"}
        if not required <= set(state) or state["step"] != step:
            raise ValueError("Incomplete training state")
        name = f"step-{step:06d}"
        target = self.root / name
        if target.exists():
            self.verify(target)
            return target
        temporary = self.root / (name+f".partial-{os.getpid()}")
        temporary.mkdir(exist_ok=False)
        # Payload writer saves adapter, optimizer/scheduler/RNG tensors and JSON progress.
        write_payload(temporary, state)
        files = [path for path in temporary.rglob("*") if path.is_file()]
        if not files or not (temporary / "progress.json").is_file():
            raise ValueError("Checkpoint payload lacks progress and training files")
        for path in files:
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        manifest = {"owner": "ragscope-v2-lora", "config_hash": self.config_hash, "step": step,
                    "artifacts": [{**artifact(path), "path": str(path.relative_to(temporary))} for path in files]}
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, target)
        self.verify(target)
        atomic_json(self.root / "latest.json", {"checkpoint": name, "step": step})
        if best:
            atomic_json(self.root / "best.json", {"checkpoint": name, "step": step})
        self._prune()
        return target

    def verify(self, path):
        path = Path(path).resolve()
        if path.parent != self.root:
            raise ValueError("Checkpoint path leaves owned run directory")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("owner") != "ragscope-v2-lora" or manifest["config_hash"] != self.config_hash:
            raise ValueError("Checkpoint identity mismatch")
        artifacts = []
        for item in manifest["artifacts"]:
            actual = (path / item["path"]).resolve()
            if not actual.is_relative_to(path):
                raise ValueError("Checkpoint artifact leaves owned directory")
            artifacts.append({**item, "path": str(actual)})
        if not verify_artifacts(artifacts):
            raise ValueError("Incomplete or corrupt checkpoint")
        return manifest

    def latest(self):
        pointer = self.root / "latest.json"
        if not pointer.exists():
            return None
        path = self.root / json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
        self.verify(path)
        return path

    def _prune(self):
        # Only verified, explicitly owned checkpoints created by this run may go.
        checkpoints = sorted(p for p in self.root.glob("step-*") if p.is_dir() and ".partial" not in p.name)
        protected = set()
        for name in ("latest.json", "best.json"):
            pointer = self.root / name
            if pointer.exists():
                protected.add(json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"])
        while len(checkpoints) > 3:
            candidate = next((p for p in checkpoints if p.name not in protected), None)
            if candidate is None:
                break
            self.verify(candidate)
            resolved = candidate.resolve()
            if resolved.parent != self.root or not resolved.name.startswith("step-"):
                raise ValueError("Unsafe checkpoint retention path")
            shutil.rmtree(resolved)
            checkpoints.remove(candidate)
