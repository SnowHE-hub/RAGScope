"""Prepare frozen train/dev LoRA examples with a locally available tokenizer."""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from ragscope.indexing.identity import atomic_json, file_hash, signature, verify_artifacts
from ragscope.training.data import prepare_examples
from ragscope.generation.prompts import PROMPT_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--max-seq-len", type=int, choices=(512, 1024), default=1024)
    args = parser.parse_args()
    corpus, out = Path(args.corpus_dir), Path(args.output_dir)
    manifest = json.loads((corpus / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    names = [f"queries_{args.split}.jsonl", "documents.jsonl"]
    for name in names:
        if file_hash(corpus / name) != manifest["files"][name]["sha256"]:
            raise ValueError("Frozen corpus artifact changed: "+name)
    queries = [json.loads(line) for line in (corpus/names[0]).read_text(encoding="utf-8").splitlines() if line]
    if set(q["query_id"] for q in queries) != set(manifest["query_ids"][args.split]):
        raise ValueError("Frozen query membership changed")
    documents = [json.loads(line) for line in (corpus/"documents.jsonl").read_text(encoding="utf-8").splitlines() if line]
    allowed = {d["doc_id"] for d in documents if d.get("split") == args.split}
    tokenizer_directory = Path(args.tokenizer_dir)
    tokenizer_manifest = json.loads((tokenizer_directory/'manifest.json').read_text(encoding='utf-8')) if (tokenizer_directory/'manifest.json').exists() else json.loads((tokenizer_directory/'model_manifest.json').read_text(encoding='utf-8'))
    tokenizer_artifacts = [row for row in tokenizer_manifest['artifacts'] if Path(row['path']).name in
                          {'tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json'}]
    if len(tokenizer_artifacts)<3 or not verify_artifacts(tokenizer_artifacts):
        raise ValueError('Pinned tokenizer artifacts missing or changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=False)
    accepted, rejected = prepare_examples(queries, documents, tokenizer, allowed,
        max_seq_len=args.max_seq_len, max_examples=1000 if args.split == "train" else 60, split=args.split)
    out.mkdir(parents=True, exist_ok=True)
    for tag, rows in (("examples", accepted), ("rejected", rejected)):
        path = out / f"{args.split}_{tag}.jsonl"
        content = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows)
        if path.exists():
            if path.read_text(encoding='utf-8') != content:
                raise ValueError('Immutable prepared examples differ; use a new output directory')
        else:
            with path.open('x',encoding='utf-8',newline='\n') as handle:
                handle.write(content)
    summary = {"split": args.split, "accepted": len(accepted), "rejected": len(rejected),
               "max_seq_len": args.max_seq_len, "tokenizer_dir": str(Path(args.tokenizer_dir).resolve()),
               "chat_template_hash": signature(tokenizer.chat_template), "examples_hash": signature(accepted),
               "tokenizer_identity": {'model':tokenizer_manifest['model'],'revision':tokenizer_manifest['revision'],
                                      'artifacts':tokenizer_artifacts},
               "rejection_counts": dict(Counter(row['reason'].split(':',1)[0] for row in rejected)),
               "answerability_counts": dict(Counter(row['answerability'] for row in accepted)),
               "max_actual_tokens": max((row['total_tokens'] for row in accepted),default=0),
               "min_supervised_tokens": min((row['supervised_tokens'] for row in accepted),default=0),
               "complete_evidence_and_answer": True, "truncation": False,
               "prompt_version":PROMPT_VERSION,"torch_imported":'torch' in sys.modules,
               "cuda_visible_devices":os.environ.get('CUDA_VISIBLE_DEVICES'),
               "cuda_initialized":bool(sys.modules['torch'].cuda.is_initialized()) if 'torch' in sys.modules else False,
               "source_manifest_sha256": file_hash(corpus / "SPLIT_MANIFEST.json")}
    atomic_json(out / f"{args.split}_manifest.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
