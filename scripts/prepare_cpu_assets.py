"""Explicit tokenizer/stopword preparation; never invoked on import or UI opening."""
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.experiments.assets import download
from ragscope.indexing.identity import atomic_json, artifact

configure(cpu_only=True)
root = runtime_home()
revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
records = []
for filename in ("tokenizer.json", "tokenizer_config.json", "config.json"):
    records.append(download(f"https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/{revision}/{filename}",
                            root / "models/qwen3-embedding-0.6b" / filename))
stopwords = root / "downloads/nltk-stopwords.zip"
records.append(download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip", stopwords))
with zipfile.ZipFile(stopwords) as archive:
    content = archive.read("stopwords/english")
    target = Path(__file__).resolve().parents[1] / "ragscope/indexing/stopwords_en.txt"
    if not target.exists():
        target.write_bytes(content)
    elif target.read_bytes() != content:
        raise ValueError("Stopword snapshot changed; preserve existing and explicitly version new snapshot")
    records.append(artifact(target))
atomic_json(root / "models/qwen3-embedding-0.6b/tokenizer_manifest.json", {"model_id": "Qwen/Qwen3-Embedding-0.6B", "revision": revision, "artifacts": records})
print(json.dumps({"status": "PASS", "files": len(records)}, ensure_ascii=False))
