"""Bounded, hash checked downloads to the project's runtime only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from ragscope.experiments.runtime import configure, runtime_home, require_disk
from ragscope.indexing.identity import artifact, atomic_json, file_hash


def download(url: str, target: Path, expected_sha256: str | None = None, size: int | None = None) -> dict:
    target = Path(target).resolve()
    if not target.is_relative_to(runtime_home()):
        raise ValueError("Download outside runtime")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_sha256 and file_hash(target) != expected_sha256:
            raise ValueError(f"Existing asset hash mismatch: {target}; preserved for investigation")
        if size is not None and target.stat().st_size != size:
            raise ValueError("Existing asset size mismatch")
        return dict(artifact(target), url=url, cached=True)
    require_disk(size or 0)
    partial = target.with_suffix(target.suffix + ".part")
    errors = []
    for attempt in range(3):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={"User-Agent": "RAGScope-v2", **({"Range": f"bytes={start}-"} if start else {})})
            with urllib.request.urlopen(request, timeout=45) as response:
                resumed = start > 0 and response.status == 206
                mode = "ab" if resumed else "wb"
                count = start if resumed else 0
                tick = time.monotonic()
                with partial.open(mode) as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
                        count += len(block)
                        if size is not None and count > size:
                            raise ValueError("Server sent more bytes than frozen asset size")
                        if time.monotonic() - tick > 10:
                            print(json.dumps({"download": target.name, "bytes": count, "expected": size}), flush=True)
                            tick = time.monotonic()
                    handle.flush()
                    os.fsync(handle.fileno())
            if size is not None and partial.stat().st_size != size:
                raise ValueError("Incomplete asset size")
            observed = file_hash(partial)
            if expected_sha256 and observed != expected_sha256:
                raise ValueError("Asset SHA-256 mismatch")
            os.replace(partial, target)
            result = dict(artifact(target), url=url, attempts=attempt + 1, preceding_errors=errors)
            atomic_json(target.with_suffix(target.suffix + ".source.json"), result)
            return result
        except Exception as exc:
            errors.append({"attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(errors[-1]), flush=True)
            if attempt < 2:
                time.sleep(2**attempt)
    atomic_json(target.with_suffix(target.suffix + ".failed.json"), {"url": url, "errors": errors})
    raise RuntimeError(f"Download exhausted: {target.name}: {errors[-1]['error']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    configure(cpu_only=True)
    spec = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    for item in spec["assets"]:
        target = runtime_home() / item["relative_path"]
        print(json.dumps(download(item["url"], target, item.get("sha256"), item.get("bytes"))), flush=True)
        if item.get("extract_to"):
            destination = runtime_home() / item["extract_to"]
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target) as archive:
                for member in archive.infolist():
                    if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                        raise ValueError("Archive path escape")
                archive.extractall(destination)

if __name__ == "__main__":
    main()
