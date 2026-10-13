"""Reconstruct the legacy corpus with a known CPU model, preserving provenance.

Historical Chroma vectors have no recoverable weight revision and are not queried.
The explicitly audited legacy 512-token truncation is excluded from v2 fairness claims.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from filelock import FileLock

from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from ragscope.experiments.ledger import Campaign, RecordStore

MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_SHA = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
PREFIX = "Represent this sentence for searching relevant passages: "


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixed_config(folder, model_dir):
    provenance = read(folder / "PROVENANCE.json")
    manifest = read(model_dir / "model_manifest.json")
    if (manifest.get("status") != "VERIFIED" or manifest.get("revision") != MODEL_REVISION
            or not verify_artifacts(manifest["artifacts"])):
        raise ValueError("Pinned local BGE-small checkpoint is incomplete or changed")
    weight = next(item for item in manifest["artifacts"] if item["relative_path"] == "model.safetensors")
    if weight["sha256"] != MODEL_SHA or not verify_artifacts([provenance["chunks"], provenance["bm25_queries"]]):
        raise ValueError("Model/corpus identity mismatch")
    return {"label": "legacy_reconstructed_fixed_revision", "original_weight_revision": "UNKNOWN",
            "corpus": provenance["chunks"], "queries": provenance["bm25_queries"],
            "model_id": "BAAI/bge-small-en-v1.5", "revision": MODEL_REVISION, "weight_sha256": MODEL_SHA,
            "model_manifest_sha256": file_hash(model_dir / "model_manifest.json"), "dimensions": 384,
            "pooling": "official_CLS", "normalization": "L2", "dtype": "float32", "device": "cpu",
            "query_instruction": PREFIX, "max_length": 512,
            "input_policy": "legacy_truncation_explicitly_counted_not_v2_fairness_comparison",
            "engine": "exact_numpy_cosine", "hybrid": {"fetch_each": 20, "final_k": 10, "alpha": .5, "rrf_k": 60},
            "source_archive_sha256": provenance["original_source_zip"]["sha256"], "seed": 42}


def load_inputs(folder):
    chunks = [json.loads(line) for line in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = read(folder / "queries_bm25_top20.json")
    if len({row["chunk_id"] for row in chunks}) != len(chunks) or len(queries) != 200:
        raise ValueError("Legacy chunk/query identity or frozen N mismatch")
    return chunks, queries


def validate_vectors(vectors, count, dimensions=384):
    if (vectors.shape != (count, dimensions) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)):
        raise ValueError("Invalid or unnormalized reconstructed vectors")


def merge_rrf(dense, sparse, k=10):
    # Original source's insertion-order tie behavior and 20-per-branch depth.
    scores, chunks = {}, {}
    for results in (dense[:20], sparse[:20]):
        for rank, item in enumerate(results, 1):
            key = item["chunk_id"]
            scores[key] = scores.get(key, 0.) + .5 / (60 + rank)
            chunks.setdefault(key, item)
    return [dict(chunks[key], score=value) for key, value in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def _resource_check(minimum):
    import psutil
    available = psutil.virtual_memory().available
    if available < minimum:
        raise MemoryError(f"Host memory reserve: available={available}, required={minimum}")
    return available


def _sample_resources(path, stop):
    """Observe this actual worker during load/encode; never act on other processes."""
    import psutil
    process = psutil.Process(os.getpid())
    with path.open("a", encoding="utf-8") as writer:
        while not stop.is_set():
            writer.write(json.dumps({"time":time.time(), "pid":process.pid,
                                     "rss":process.memory_info().rss,
                                     "host_available":psutil.virtual_memory().available})+"\n")
            writer.flush()
            stop.wait(1)


def encode(folder, model_dir, *, reserve_bytes=1280 * 1024**2):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Source the CPU-only runtime environment before E0 reconstruction")
    _resource_check(1792 * 1024**2)
    config = fixed_config(folder, model_dir)
    chunks, queries = load_inputs(folder)
    directory = folder / ("embeddings_" + signature(config)[:16])
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "writer.lock"), timeout=0):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = read(manifest_path)
            if manifest["config"] != config or not verify_artifacts(manifest["artifacts"]):
                raise ValueError("Completed reconstruction changed; preserve it")
            return directory
        state_path = directory / "state.json"
        state = read(state_path) if state_path.exists() else {"config": config, "completed_batches": [], "attempts": 0}
        if state["config"] != config:
            raise ValueError("Partial reconstruction identity mismatch")
        if state.get("backend_failures", 0) >= 2:
            raise RuntimeError("The two permitted model/backend attempts are exhausted")
        state["initial_available_bytes"] = _resource_check(1792 * 1024**2)
        state.update(status="RUNNING", attempts=state["attempts"]+1, pid=os.getpid(), argv=sys.argv,
                     started_at=time.time(), model_loaded=False)
        atomic_json(state_path, state)
        started = time.perf_counter()
        monitor_stop = threading.Event()
        monitor = threading.Thread(target=_sample_resources, args=(directory/"resource_samples.jsonl", monitor_stop), daemon=True)
        monitor.start()
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            import psutil
            if torch.version.cuda is not None:
                raise ValueError("E0 requires the isolated CPU-only PyTorch environment")
            torch.set_num_threads(4)
            model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True, trust_remote_code=False)
            if model.max_seq_length != 512 or model.get_sentence_embedding_dimension() != 384:
                raise ValueError("The official BGE-small tokenizer/model contract changed")
            state.update(model_loaded=True, load_seconds=time.perf_counter()-started,
                         parameter_count=sum(p.numel() for p in model.parameters()), torch_version=torch.__version__)
            atomic_json(state_path, state)
            texts = [item["text"] for item in chunks] + [PREFIX+item["query"]["question"] for item in queries]
            batch_count = math.ceil(len(texts) / 128)
            for number in range(batch_count):
                start, stop = number*128, min((number+1)*128, len(texts))
                batch_file = directory / f"batch_{number:04d}.json"
                if batch_file.exists():
                    batch = read(batch_file)
                    if (batch["config_hash"] != signature(config) or batch["start"] != start or batch["stop"] != stop
                            or batch["texts_hash"] != signature(texts[start:stop]) or not verify_artifacts([batch["array"]])):
                        raise ValueError("Partial batch changed; no overwrite permitted")
                    continue
                state["last_available_bytes"] = _resource_check(reserve_bytes)
                token_counts = [len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
                                for text in texts[start:stop]]
                tick = time.perf_counter()
                vectors = model.encode(texts[start:stop], batch_size=8, show_progress_bar=False,
                                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
                validate_vectors(vectors, stop-start)
                array = directory / f"batch_{number:04d}.npy"
                temporary = array.with_suffix(".tmp")
                with temporary.open("wb") as writer:
                    np.save(writer, vectors, allow_pickle=False); writer.flush(); os.fsync(writer.fileno())
                os.replace(temporary, array)
                batch = {"config_hash": signature(config), "start": start, "stop": stop,
                         "texts_hash": signature(texts[start:stop]), "array": artifact(array),
                         "input_tokens_before_legacy_truncation": token_counts,
                         "encoded_tokens": [min(count, 512) for count in token_counts],
                         "truncated_indices": [start+i for i,count in enumerate(token_counts) if count>512],
                         "seconds": time.perf_counter()-tick, "real_model": True}
                atomic_json(batch_file, batch)
                state["completed_batches"] = sorted(set(state["completed_batches"]+[number]))
                rss = psutil.Process(os.getpid()).memory_info().rss
                available = psutil.virtual_memory().available
                state.update(completed_texts=stop, elapsed_seconds=time.perf_counter()-started,
                             current_process_rss=rss, peak_sampled_process_rss=max(state.get("peak_sampled_process_rss",0),rss),
                             current_available_bytes=available,
                             minimum_available_bytes=min(state.get("minimum_available_bytes",available),available))
                atomic_json(state_path, state)
                print(json.dumps({"batch":number+1,"batches":batch_count,"texts_completed":stop,
                                  "seconds":batch["seconds"],"truncated":len(batch["truncated_indices"]),
                                  "actual_model_process_rss":rss,"host_available_bytes":available}),flush=True)
                _resource_check(reserve_bytes)  # Commit the batch first, then leave training its RAM reserve.
            arrays = [np.load(directory/f"batch_{i:04d}.npy",allow_pickle=False) for i in range(batch_count)]
            vectors = np.concatenate(arrays)
            validate_vectors(vectors, len(texts))
            full = directory / "vectors.npy"
            with full.open("wb") as writer:
                np.save(writer,vectors,allow_pickle=False)
            batches = [read(directory/f"batch_{i:04d}.json") for i in range(batch_count)]
            artifacts = [artifact(full)]+[artifact(directory/f"batch_{i:04d}.json") for i in range(batch_count)]
            manifest = {"config":config,"artifacts":artifacts,"status":"COMPLETE","real_model":True,
                        "n_chunks":len(chunks),"n_queries":len(queries),"batch_count":batch_count,
                        "truncated_chunks":sum(index<len(chunks) for batch in batches for index in batch["truncated_indices"]),
                        "truncated_queries":sum(index>=len(chunks) for batch in batches for index in batch["truncated_indices"]),
                        "elapsed_seconds":time.perf_counter()-started}
            atomic_json(manifest_path,manifest)
            state.update(status="COMPLETE",model_loaded=False,manifest=artifact(manifest_path))
            atomic_json(state_path,state)
            return directory
        except Exception as error:
            if not isinstance(error, MemoryError):
                state["backend_failures"] = state.get("backend_failures", 0) + 1
            state.update(status="PENDING_RESOURCE" if isinstance(error,MemoryError) else "FAILED",
                         error={"type":type(error).__name__,"message":str(error)},elapsed_seconds=time.perf_counter()-started)
            atomic_json(state_path,state)
            raise
        finally:
            monitor_stop.set()
            monitor.join(timeout=2)


def evaluate(folder, model_dir, home):
    config = fixed_config(folder, model_dir)
    directory=folder/("embeddings_"+signature(config)[:16])
    manifest=read(directory/"manifest.json")
    if manifest["config"]!=config or manifest["status"]!="COMPLETE" or not verify_artifacts(manifest["artifacts"]):
        raise ValueError("Full real encoding must complete before retrieval evaluation")
    chunks,queries=load_inputs(folder)
    vectors=np.load(directory/"vectors.npy",allow_pickle=False)
    validate_vectors(vectors,len(chunks)+len(queries))
    # Import only the pure metric file, not the optional generation metric package.
    source=Path(__file__).resolve().parents[1]/"evaluation/retrieval_metrics.py"
    spec=importlib.util.spec_from_file_location("e0_metrics",source)
    metrics_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics_module)
    with Campaign(home,"20261014_v2_legacy_reconstructed") as campaign:
        for method in ("dense","hybrid"):
            run_id="E0.RECONSTRUCTED_"+method.upper()
            run_config={**config,"method":method,"embedding_manifest":artifact(directory/"manifest.json"),
                        "query_ids":[item["query"]["query_id"] for item in queries],"metrics_source_sha256":file_hash(source)}
            if not campaign.begin(run_id,run_config,[sys.executable,"-m",__name__,"--stage","evaluate"]):continue
            store=RecordStore(campaign.path/(method+".jsonl"),run_config,real_only=True,max_records=200)
            for i,item in enumerate(queries):
                query=item["query"]
                if query["query_id"] in store.ids:continue
                start=time.perf_counter()
                try:
                    scores=vectors[:len(chunks)]@vectors[len(chunks)+i]
                    order=np.argsort(-scores,kind="stable")[:20]
                    dense=[dict(chunks[int(index)],score=float(scores[index])) for index in order]
                    hits=dense[:10] if method=="dense" else merge_rrf(dense,item["bm25_top20"])
                    ids=[hit["doc_id"] for hit in hits]; gold={query["gold_doc_id"]}
                    metrics={"recall5":metrics_module.recall_at_k(ids,gold,5),"hit5":metrics_module.hit_rate_at_k(ids,gold,5),
                             "recall10":metrics_module.recall_at_k(ids,gold,10),"mrr":metrics_module.mrr(ids,gold),
                             "ndcg10":metrics_module.ndcg_at_k(ids,gold,10)}
                    error=None
                except Exception as exc:
                    hits=[]; metrics={key:0. for key in ("recall5","hit5","recall10","mrr","ndcg10")}; error=str(exc)
                store.add(query["query_id"],{"query_id":query["query_id"],"question":query["question"],"doc_id":query["gold_doc_id"],
                         "raw_retrieval":hits,"metrics":metrics,"error":error,"seconds":time.perf_counter()-start,
                         "real_model":True,"model_identity":config,"experiment_type":"legacy_reconstructed_fixed_revision",
                         "timing_scope":"search_and_fusion_only; cached query encoding separately audited in embedding batches"})
            summary={"status":"COMPLETE","n":len(store.records),"failures":sum(bool(row['error']) for row in store.records),
                     "method":method,"label":config["label"],"original_weight_revision":"UNKNOWN",
                     "truncated_chunks":manifest["truncated_chunks"],"truncated_queries":manifest["truncated_queries"],
                     "metrics":{key:sum(row['metrics'][key] for row in store.records)/len(store.records) for key in store.records[0]['metrics']},
                     "same200_query_ids":run_config["query_ids"],"deviations":["pinned current official weights replace unknown original revision",
                      "exact cosine search replaces original unmanifested HNSW", "legacy512 truncation explicitly audited; excluded from v2 fairness comparisons",
                      "current validated document metrics are source-hashed; old reports remain unchanged"]}
            output=campaign.path/"summaries"/(method+".json"); atomic_json(output,summary)
            campaign.finish(run_id,"PASS",[store.path,store.path.with_suffix('.config.json'),output],summary)
            print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home",default="D:/RAGScope_Runtime")
    parser.add_argument("--stage",choices=("encode","evaluate","all"),default="all")
    args=parser.parse_args()
    home=Path(args.home); folder=home/"runs/20261014_v2_legacy/reconstructed"
    model_dir=home/"models/bge-small-en-v1.5"/MODEL_REVISION
    if args.stage in {"encode","all"}:
        try:encode(folder,model_dir)
        except MemoryError as error:
            print(json.dumps({"status":"PENDING_RESOURCE","reason":str(error),"own_model_process_exiting":True}),flush=True)
            return 2
    if args.stage in {"evaluate","all"}:evaluate(folder,model_dir,home)
    return 0


if __name__=="__main__":raise SystemExit(main())
