"""Read-only model source identities and sizes; no weight downloads or model requests."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import urllib.request
import urllib.error
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.indexing.identity import atomic_json, signature

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return {"url": url, "status": response.status, "body": json.load(response)}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"url": url, "error": str(exc)}

configure(cpu_only=True)
results = []
for model in ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "LiquidAI/LFM2.5-1.2B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"):
    response = fetch("https://huggingface.co/api/models/" + model)
    data = response.pop("body", {})
    response.update(model=model, revision=data.get("sha"), license=data.get("cardData", {}).get("license"), files=[i["rfilename"] for i in data.get("siblings", [])])
    results.append(response)
    print(json.dumps(response), flush=True)
for tag in ("2b", "4b", "9b"):
    response = fetch("https://registry.ollama.ai/v2/library/qwen3.5/manifests/" + tag)
    response["model"] = "qwen3.5:" + tag
    results.append(response)
    print(json.dumps(response), flush=True)
atomic_json(runtime_home() / "runs/20261014_v2/model_sources.json", {"observations": results, "identity_hash": signature(results), "model_loaded": False})
