"""Bounded anonymous downloads of pinned official semantic metric weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_disk, require_ram
from ragscope.indexing.identity import artifact, atomic_json

MODELS = ['FacebookAI/roberta-large', 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli']
SMALL_FILES = {'README.md', 'LICENSE', 'config.json', 'tokenizer_config.json',
               'special_tokens_map.json', 'tokenizer.json', 'vocab.json', 'merges.txt',
               'spm.model', 'tokenizer.model', 'added_tokens.json'}


def metadata(repo):
    url = 'https://huggingface.co/api/models/' + repo + '?blobs=true'
    errors = []
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=45) as handle:
                data = handle.read(2 * 1024**2 + 1)
            if len(data) > 2 * 1024**2:
                raise ValueError('Model metadata exceeds bound')
            value = json.loads(data)
            return value, {'url': url, 'attempts': attempt, 'preceding_errors': errors}
        except Exception as exc:
            errors.append({'attempt': attempt, 'exception_type': type(exc).__name__, 'message': str(exc)[:500]})
    raise RuntimeError('Official model metadata unavailable after three bounded attempts: ' + repo)


def download_file(url, destination, expected):
    errors = []
    size = expected.get('size') or expected.get('lfs', {}).get('size')
    sha = expected.get('lfs', {}).get('sha256')
    if destination.exists():
        saved = artifact(destination)
        if (size is not None and saved['bytes'] != size) or (sha and saved['sha256'] != sha):
            raise ValueError('Existing pinned file differs; never overwrite')
        return {**saved, 'url': url, 'attempts': 0, 'reused': True, 'official_lfs_sha256': sha}
    maximum = int(size) if size is not None else 64 * 1024**2
    require_disk(maximum)
    for attempt in range(1, 4):
        part = destination.with_suffix(destination.suffix + f'.attempt{attempt}.partial')
        if part.exists():
            errors.append({'attempt': attempt, 'exception_type': 'PreservedPriorPartial',
                           'message': 'Earlier partial retained; this attempt is not restarted'})
            continue
        try:
            digest, total, last = hashlib.sha256(), 0, time.monotonic()
            with urllib.request.urlopen(url, timeout=45) as response, part.open('xb') as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > maximum:
                        raise ValueError('Download exceeds official/declaration bound')
                    handle.write(block)
                    digest.update(block)
                    if time.monotonic() - last >= 20:
                        print(json.dumps({'event': 'DOWNLOAD_PROGRESS', 'file': destination.name,
                                          'bytes': total, 'expected_bytes': size}), flush=True)
                        last = time.monotonic()
                handle.flush()
                os.fsync(handle.fileno())
            if size is not None and total != size or sha and digest.hexdigest() != sha:
                raise ValueError('Official model byte count/LFS hash mismatch')
            part.rename(destination)
            return {**artifact(destination), 'url': url, 'attempts': attempt,
                    'preceding_errors': errors, 'official_lfs_sha256': sha, 'git_blob_id': expected.get('blobId')}
        except Exception as exc:
            errors.append({'attempt': attempt, 'exception_type': type(exc).__name__, 'message': str(exc)[:500]})
            print(json.dumps({'event': 'DOWNLOAD_ATTEMPT_FAILED', 'file': destination.name,
                              'attempt': attempt, 'exception_type': type(exc).__name__}), flush=True)
    raise RuntimeError('Official file unavailable after bounded attempts: ' + destination.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if not run.is_relative_to(runtime / 'runs') or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2 * 1024**3)
    with FileLock(str(run / 'writer.lock'), timeout=0):
        all_models = []
        for repo in MODELS:
            api_file = run / 'source_package' / (repo.replace('/', '__') + '.hf_api.json')
            if api_file.exists():
                api = json.loads(api_file.read_text('utf-8'))
                meta_receipt = {'status': 'PINNED_API_REUSED', 'source': artifact(api_file)}
            else:
                api, meta_receipt = metadata(repo)
                atomic_json(api_file, api)
            revision = api['sha']
            target = runtime / 'models/v3_metrics' / repo.replace('/', '__') / revision
            target.mkdir(parents=True, exist_ok=True)
            siblings = {s['rfilename']: s for s in api['siblings']}
            weight = 'model.safetensors' if 'model.safetensors' in siblings else 'pytorch_model.bin'
            if weight not in siblings or not siblings[weight].get('lfs', {}).get('sha256'):
                raise ValueError('A single official hashed model weight is required')
            names = sorted((SMALL_FILES & set(siblings)) | {weight})
            files = []
            for name in names:
                file = download_file('https://huggingface.co/' + repo + '/resolve/' + revision + '/' + name,
                                     target / name, siblings[name])
                files.append(file)
                atomic_json(target / 'DOWNLOAD_PROGRESS.json', {'model': repo, 'revision': revision, 'files': files,
                    'status': 'DOWNLOADING', 'new_llm_requests': 0})
                print(json.dumps({'event': 'OFFICIAL_FILE_VERIFIED', 'model': repo,
                                  'file': name, 'bytes': file['bytes']}), flush=True)
            entry = {'model': repo, 'revision': revision, 'directory': str(target),
                     'api_source': artifact(api_file), 'metadata_request': meta_receipt,
                     'artifacts': files, 'trust_remote_code': False, 'source': 'official_anonymous_huggingface',
                     'status': 'DOWNLOAD_VERIFIED_NOT_LOADED', 'new_llm_requests': 0}
            atomic_json(target / 'model_manifest.json', entry)
            all_models.append(entry)
        atomic_json(run / 'METRIC_MODEL_MANIFEST.json', {'models': all_models, 'auditor': artifact(__file__),
            'bert_score_configuration': {'implementation': 'bert-score==0.3.13', 'model': MODELS[0], 'num_layers': 17,
                'idf': False, 'rescale_with_baseline': False, 'use_fast_tokenizer': False,
                'selection': 'Official English default; fixed before scoring, not selected for higher results'},
            'nli_configuration': {'model': MODELS[1], 'id2label_policy': 'read_actual_config_before_scoring'},
            'status': 'DOWNLOADS_COMPLETE_METRICS_NOT_RUN', 'new_llm_requests': 0})
        print(json.dumps({'status': 'METRIC_DOWNLOADS_VERIFIED', 'model_n': len(all_models), 'new_llm_requests': 0}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
