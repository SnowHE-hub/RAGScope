"""Anonymous, bounded official model-card reads; never download generator weights."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ragscope.indexing.identity import artifact, atomic_json


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'RAGScope-public-metadata-audit'})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(512*1024+1)
    if len(body) > 512*1024:
        raise ValueError('Public metadata exceeds bounded allowance')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path('D:/RAGScope_Runtime').resolve()
    run = Path(args.campaign).resolve()
    if run.parent != runtime/'runs' or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    receipt = run/'GENERATION_MODEL_CARDS.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text('utf-8'))
        if not args.resume or saved['runner'] != artifact(__file__) or any(artifact(r['artifact']['path']) != r['artifact'] for r in saved['models'] if r.get('artifact')):
            raise ValueError('Frozen model-card identity changed')
        print(json.dumps({'status': 'REUSED_PINNED_MODEL_CARDS'})); return 0
    disk = shutil.disk_usage(runtime)
    if disk.free < 20*1024**3 or disk.free/disk.total < .10:
        raise RuntimeError('Required D free-space download guard not met')
    models = []
    for size in ('2B', '4B', '9B'):
        manifest = json.loads((runtime/f'models/tokenizers/Qwen3.5-{size}/manifest.json').read_text('utf-8'))
        repo, revision = manifest['model'], manifest['revision']
        url = f'https://huggingface.co/{repo}/resolve/{revision}/README.md'
        row = {'model': repo, 'tokenizer_revision': revision, 'requested_source': url, 'attempts': []}
        try:
            body = fetch(url)
            row['attempts'].append({'url': url, 'status': 'FOUND'})
        except Exception as exc:
            row['attempts'].append({'url': url, 'status': 'FAILED', 'error': str(exc)})
            # One source-compatible fallback: current official card pinned to its actual repo commit.
            try:
                info_url = f'https://huggingface.co/api/models/{repo}'
                info = json.loads(fetch(info_url))
                revision = info['sha']
                url = f'https://huggingface.co/{repo}/resolve/{revision}/README.md'
                body = fetch(url)
                row['attempts'].append({'url': url, 'status': 'FOUND_CURRENT_CARD_PINNED', 'metadata_url': info_url})
            except Exception as fallback:
                row.update(status='BLOCKED_PUBLIC_CARD', error=str(fallback))
                models.append(row); continue
        folder = runtime/'models/v3_metadata'/repo.replace('/', '__')/revision
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'README.md'
        if path.exists() and path.read_bytes() != body:
            raise ValueError('Pinned model card differs; do not overwrite')
        if not path.exists():
            path.write_bytes(body)
        row.update(status='FOUND', card_revision=revision, source=url, artifact=artifact(path))
        models.append(row)
    output = {'status': 'PASS' if all(r['status'] == 'FOUND' for r in models) else 'PARTIAL_PUBLIC_SOURCE_BLOCK',
        'runner': artifact(__file__), 'models': models, 'generator_weights_downloaded': False,
        'anonymous_public_requests_only': True, 'new_llm_requests': 0}
    atomic_json(receipt, output)
    print(json.dumps(output)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
