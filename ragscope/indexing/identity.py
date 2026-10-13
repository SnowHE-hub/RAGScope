"""Content identities shared by indexes, caches, configs and experiment records."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def signature(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def verify_artifacts(items: list[dict]) -> bool:
    return all(Path(a["path"]).is_file() and Path(a["path"]).stat().st_size == a["bytes"]
               and file_hash(a["path"]) == a["sha256"] for a in items)
