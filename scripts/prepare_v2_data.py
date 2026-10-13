"""Download and freeze the real v2 datasets with bounded memory, CPU only.

Run from the project root using the isolated core interpreter. Existing frozen
outputs are compared byte-for-byte, never silently replaced. Raw archives and
excluded annotations remain available for audit on RAGSCOPE_HOME.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ragscope.ingestion.qasper import (canonical_json, convert_question, digest,
    duplicate_groups, freeze_splits, make_document)
from ragscope.ingestion.technical import BASE_URL, CASE_SPECS, REVISION, build_technical_document


def sha_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def immutable_json(path: Path, value) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to change frozen file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(payload)
    temp.replace(path)


def immutable_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    if path.exists():
        if sha_file(path) != sha_file(temp):
            raise ValueError(f"Refusing to change frozen file: {path}")
        temp.unlink()  # this function's own reconstructible temporary file only
    else:
        temp.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_json_object(path: Path):
    """Incrementally parse a top-level JSON map without materializing the corpus."""
    decoder, buffer, eof = json.JSONDecoder(), "", False
    with path.open(encoding="utf-8") as handle:
        def refill():
            nonlocal buffer, eof
            chunk = handle.read(65536)
            buffer += chunk
            eof = not chunk

        refill()
        buffer = buffer.lstrip()
        if not buffer.startswith("{"):
            raise ValueError("Expected top-level object")
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                refill()
                continue
            if buffer.startswith("}"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    key, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
                    buffer = buffer.lstrip()
            buffer = buffer[end:].lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()
            if not buffer.startswith(":"):
                raise ValueError("Expected colon after JSON map key")
            buffer = buffer[1:].lstrip()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
                    buffer = buffer.lstrip()
            yield key, value
            buffer = buffer[end:]


def download(url: str, destination: Path, log_path: Path, attempts=3) -> dict:
    """Bounded network retries, source identity and SHA-256, Range resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(destination.suffix + ".source.json")
    if destination.exists():
        checksum = sha_file(destination)
        if sidecar.exists():
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            if saved["sha256"] != checksum or saved["url"] != url:
                raise ValueError(f"Raw snapshot integrity mismatch: {destination}")
            return saved
        record = {"url": url, "sha256": checksum, "bytes": destination.stat().st_size,
                  "path": str(destination), "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "http_headers": {}, "note": "preparation-session download; source assigned and hashed before processing"}
        immutable_json(sidecar, record)
        return record
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts+1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "RAGScope-local-research/2"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                response_headers = {k: response.headers[k] for k in ("ETag", "Last-Modified", "Content-Length") if k in response.headers}
                received = 0
                with partial.open("ab" if append else "wb") as handle:
                    while block := response.read(1024*1024):
                        handle.write(block)
                        received += len(block)
                expected = response.headers.get("Content-Length")
                if expected and received != int(expected):
                    raise IOError(f"Incomplete response: {received}/{expected}")
            partial.replace(destination)
            record = {"url": url, "sha256": sha_file(destination), "bytes": destination.stat().st_size,
                      "path": str(destination), "retrieved_at": datetime.now(timezone.utc).isoformat(),
                      "http_headers": response_headers}
            immutable_json(sidecar, record)
            print(f"Downloaded {destination.name}: {record['bytes']} bytes", flush=True)
            return record
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json({"url": url, "attempt": attempt, "error": repr(exc),
                                            "time": datetime.now(timezone.utc).isoformat()}) + "\n")
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def prepare(args) -> dict:
    project = Path(__file__).resolve().parents[1]
    home = Path(args.home).resolve()
    if os.name == "nt" and home.drive.upper() != "D:":
        raise ValueError("This campaign requires D: runtime storage")
    output = home / "corpora" / args.campaign
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(home / "cache" / "tiktoken")
    os.environ["TMP"] = os.environ["TEMP"] = str(home / "tmp")
    config = json.loads((project / "configs/v2/data.json").read_text(encoding="utf-8"))
    config_hash = digest(config)
    immutable_json(output / "FROZEN_CONFIG.json", {"config": config, "config_hash": config_hash})
    log_path, sources = output / "download_attempts.jsonl", []
    for name in ("qasper-train-dev-v0.3.tgz", "qasper-test-and-evaluator-v0.3.tgz"):
        archive = raw / name
        sources.append(download("https://qasper-dataset.s3.us-west-2.amazonaws.com/" + name, archive, log_path))
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.name in {f"qasper-{split}-v0.3.json" for split in ("train", "dev", "test")}:
                    target = raw / member.name
                    if not target.exists():
                        with tar.extractfile(member) as source, target.with_suffix(".json.part").open("wb") as dest:
                            shutil.copyfileobj(source, dest, 1024*1024)
                        target.with_suffix(".json.part").replace(target)
    # Keep versioned dataset metadata/license with the raw snapshots.
    sources.append(download("https://huggingface.co/datasets/allenai/qasper/raw/main/README.md", raw / "QASPER_DATA_CARD.md", log_path))
    all_docs_path, all_queries_path = output / "all_documents.jsonl", output / "all_queries_audit.jsonl"
    thin_docs, thin_queries, source_counts = [], [], Counter()

    def source_docs():
        for split in ("train", "dev", "test"):
            for paper_id, paper in iter_json_object(raw / f"qasper-{split}-v0.3.json"):
                doc = make_document(paper_id, paper, split)
                source_counts[f"{split}_documents"] += 1
                thin_docs.append({"doc_id": doc["doc_id"], "split": split})
                yield doc

    immutable_jsonl(all_docs_path, source_docs())

    def source_queries():
        for split in ("train", "dev", "test"):
            for paper_id, paper in iter_json_object(raw / f"qasper-{split}-v0.3.json"):
                doc = make_document(paper_id, paper, split)
                for question in paper.get("qas", []):
                    query, reasons = convert_question(question, doc)
                    source_counts[f"{split}_queries"] += 1
                    thin_queries.append({"query_id": query["query_id"], "doc_id": query["doc_id"],
                                         "split": split, "answerability": query["answerability"],
                                         "metadata": {"quarantine_reasons": reasons}})
                    yield query

    immutable_jsonl(all_queries_path, source_queries())
    print("Source conversion complete: " + canonical_json(source_counts), flush=True)
    groups = duplicate_groups(read_jsonl(all_docs_path))
    immutable_json(output / "duplicate_groups.json", groups)
    chosen = freeze_splits(thin_docs, thin_queries, groups, config["targets"], config["max_selected_documents"], config["seed"])
    doc_ids = {d["doc_id"] for d in chosen["documents"]}
    query_ids = {q["query_id"] for q in chosen["queries"]}
    import tiktoken
    encoding = tiktoken.get_encoding(config["reference_tokenizer"])
    lengths = Counter()

    def length_documents():
        for document in read_jsonl(all_docs_path):
            if document["doc_id"] in doc_ids:
                count = len(encoding.encode(document["text"], disallowed_special=()))
                group = "short" if count <= 512 else "medium" if count <= 4096 else "long"
                document["metadata"].update(reference_tokenizer="cl100k_base", reference_n_tokens=count, length_group=group)
                lengths[group] += 1
                yield document

    immutable_jsonl(output / "documents.jsonl", length_documents())
    for split in ("train", "dev", "test"):
        immutable_jsonl(output / f"queries_{split}.jsonl", (q for q in read_jsonl(all_queries_path)
                         if q["query_id"] in query_ids and q["split"] == split))
    globals_by_id = {q["query_id"]: q for q in chosen["global_queries"]}

    def global_rows():
        for query in read_jsonl(output / "queries_test.jsonl"):
            if query["query_id"] in globals_by_id:
                query["scope"] = {"kind": "global", "corpus": "frozen_qasper", "corpus_doc_ids_hash": digest(sorted(doc_ids))}
                query["metadata"]["gold_doc_id_evaluation_only"] = query.pop("doc_id")
                yield query

    immutable_jsonl(output / "queries_global_test.jsonl", global_rows())
    immutable_jsonl(output / "quarantine.jsonl", (q for q in read_jsonl(all_queries_path) if q["metadata"]["quarantine_reasons"]))
    tech_docs, tech_cases = [], []
    for name in CASE_SPECS:
        source_path = raw / "technical" / f"{name}.rst"
        sources.append(download(BASE_URL + name + ".rst", source_path, log_path))
        doc, cases = build_technical_document(name, source_path.read_text(encoding="utf-8"))
        tech_docs.append(doc)
        tech_cases.extend(cases)
    sources.append(download(f"https://raw.githubusercontent.com/python/cpython/{REVISION}/LICENSE", raw / "technical" / "LICENSE", log_path))
    immutable_jsonl(output / "technical_documents.jsonl", tech_docs)
    immutable_jsonl(output / "technical_cases.jsonl", tech_cases)
    legacy = []
    for folder in ("data/raw", "data/processed"):
        for path in sorted((project / folder).glob("*.json*")):
            relative = path.relative_to(project)
            target = output / "legacy_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
            original_hash, snapshot_hash = sha_file(path), sha_file(target)
            if original_hash != snapshot_hash:
                raise ValueError(f"Legacy mismatch: {relative}")
            legacy.append({"source": str(path), "snapshot": str(target), "bytes": path.stat().st_size,
                           "sha256": original_hash})
    immutable_json(output / "legacy_manifest.json", {"files": legacy,
        "arxiv_limit": "legacy arxiv abstracts have no evaluation questions; not a labeled long-text QA benchmark"})
    immutable_json(output / "source_manifest.json", sources)
    quarantine_counts = Counter(reason for q in read_jsonl(output / "quarantine.jsonl") for reason in q["metadata"]["quarantine_reasons"])
    frozen = {"campaign": args.campaign, "config_hash": config_hash, "seed": config["seed"],
              "split_policy": config["split_policy"], "selection_rule": chosen["selection_rule"],
              "source_counts": dict(source_counts), "counts": chosen["counts"], "target_counts": config["targets"],
              "selected_document_count": len(doc_ids), "document_ids": sorted(doc_ids),
              "query_ids": {split: sorted(q["query_id"] for q in chosen["queries"] if q["split"] == split)
                            for split in ("train", "dev", "test")}, "global_test_query_ids": sorted(globals_by_id),
              "length_groups": dict(lengths), "reference_tokenizer": {"id": "cl100k_base", "package_version": tiktoken.__version__},
              "quarantine_reason_counts_all_source_queries": dict(quarantine_counts),
              "aliases": chosen["aliases"], "cross_split_quarantined_doc_ids": chosen["cross_split_quarantined_doc_ids"],
              "duplicate_groups": len(groups), "technical_documents": len(tech_docs), "technical_cases": len(tech_cases),
              "technical_authorship": "agent_authored_not_human_validated", "cpu_only": True,
              "files": {p.name: {"sha256": sha_file(p), "bytes": p.stat().st_size}
                        for p in sorted(output.glob("*.jsonl")) if p.name != "download_attempts.jsonl"}}
    immutable_json(output / "SPLIT_MANIFEST.json", frozen)
    print(canonical_json({"status": "PASS", "manifest": str(output / "SPLIT_MANIFEST.json"),
                          "counts": chosen["counts"], "documents": len(doc_ids),
                          "global_test": len(globals_by_id), "technical_cases": len(tech_cases)}), flush=True)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default=os.environ.get("RAGSCOPE_HOME", "D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    prepare(parser.parse_args())
