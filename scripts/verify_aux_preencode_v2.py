"""Read back completed auxiliary indexes without model calls or main-ledger writes."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ragscope.experiments.aux_preencode import frozen_job
from ragscope.experiments.ledger import now, read_records
from ragscope.experiments.retrieval_campaign import chunker_for
from ragscope.experiments.runtime import configure
from ragscope.indexing.array_index import ArrayCache
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from ragscope.indexing.local_embedding import HFJsonTokenizer
from ragscope.indexing.scoped_index import ENGINE
from ragscope.ingestion.chunkers import TokenizerAdapter


def main():
    configure(cpu_only=True)
    home, campaign_id = Path("D:/RAGScope_Runtime"), "20261014_v2"
    run = home / "runs" / (campaign_id + "_aux_preencode")
    identity = json.loads((home / "runs" / campaign_id / "embedding_identity.json").read_text(encoding="utf-8"))
    cache = ArrayCache(home / "cache/ragscope/v2_embeddings", identity)
    validated = []
    for dataset in ("squad", "qasper"):
        job = frozen_job(home, campaign_id, dataset)
        cfg = job["chunking"]
        tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
        assert file_hash(tokenizer_path) == cfg["tokenizer_file_sha256"]
        backend = HFJsonTokenizer(tokenizer_path, cfg["tokenizer_id"], cfg["tokenizer_revision"])
        tokenizer = TokenizerAdapter(backend, cfg["tokenizer_id"], cfg["tokenizer_revision"])
        chunk_identity = {**job["base_identity"], "group": job["group"],
                          "chunking_signature": chunker_for(job["group"], tokenizer).chunking_signature}
        directory = home / "indexes" / campaign_id / job["area"] / signature(chunk_identity)
        chunk_manifest = json.loads((directory / "chunk_manifest.json").read_text(encoding="utf-8"))
        assert chunk_manifest["identity"] == chunk_identity and verify_artifacts(chunk_manifest["artifacts"])
        expected = {**chunk_identity, "engine": ENGINE if dataset == "qasper" else "ArrayIndex-v2",
                    "bm25": job["config"]["bm25"], "embedding": identity,
                    "embedding_input": job["config"]["embedding_input"]}
        dense = directory / ("dense_" + signature(identity)[:16])
        index = job["index_type"].load(dense, expected)
        assert index.vectors.shape == (chunk_manifest["count"], 1024)
        assert index.vectors.dtype == np.float32 and np.isfinite(index.vectors).all()
        assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1, atol=1e-5, rtol=0)
        for chunk, vector in zip(index.chunks, index.vectors):
            assert np.array_equal(vector, cache.get(chunk["text"]))
            assert len(backend.encode(chunk["text"])) <= 512
        observations = read_records(dense / "embedding_first_observations.jsonl")
        assert len(observations) == len(index.chunks)
        assert {r["chunk_id"] for r in observations} == {c["chunk_id"] for c in index.chunks}
        validated.append({"dataset": dataset, "count": len(index.chunks), "index_signature": index.signature,
            "manifest": artifact(dense / "manifest.json"), "chunk_manifest": artifact(directory / "chunk_manifest.json"),
            "exact_cache_vector_matches": len(index.chunks), "dtype": "float32", "dimensions": 1024,
            "all_finite_unit_norm": True, "all_body_tokens_le_512": True,
            "first_observations": artifact(dense / "embedding_first_observations.jsonl")})
        del index, job, backend, tokenizer
    output = {"status": "PASS", "validated_at": now(), "model_calls": 0,
        "main_campaign_opened": False, "readback_after_save": True, "indexes": validated,
        "clarification": "Aux task index_loaded_and_validated covered the constructed index. This separate receipt confirms actual persisted reload and cache alignment.",
        "source": artifact(Path(__file__))}
    atomic_json(run / "INDEPENDENT_INDEX_READBACK.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
