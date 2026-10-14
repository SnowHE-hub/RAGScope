"""Verify actual reused model bytes, environments and E8 denominator on CPU."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json
from ragscope.v3.audit import journal_rows


def artifacts_in(value):
    if isinstance(value, dict):
        if all(k in value for k in ('path', 'bytes', 'sha256')):
            yield value
        else:
            for item in value.values():
                yield from artifacts_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from artifacts_in(item)


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
        output = run / 'S0_IDENTITY_E8_AUDIT.json'
        if output.exists():
            if not args.resume:
                raise FileExistsError('Identity audit already exists')
            saved = json.loads(output.read_text('utf-8'))
            if saved['auditor'] != artifact(__file__):
                raise ValueError('Identity auditor changed')
            print(json.dumps({'status': 'EXISTING_IDENTITY_AUDIT', 'new_llm_requests': 0}))
            return 0
        models = list((runtime / 'models/generation').glob('*.json'))
        models += list((runtime / 'models/generation/text_only').glob('*.json'))
        models += list((runtime / 'models').rglob('*manifest.json'))
        models = sorted(set(models))
        verified, dedup, total = [], {}, 0
        for manifest_path in models:
            contents = json.loads(manifest_path.read_text('utf-8'))
            for expected in artifacts_in(contents):
                path = Path(expected['path']).resolve()
                if not path.is_relative_to(runtime) and not path.is_relative_to(PROJECT):
                    raise ValueError('Model artifact outside verified runtime/project')
                record = {'source_manifest': str(manifest_path), 'path': str(path), 'status': 'MISSING',
                          'location_role': 'runtime' if path.is_relative_to(runtime) else 'preserved_project_source'}
                if path.is_file():
                    stat = path.stat()
                    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    if key not in dedup:
                        digest = hashlib.sha256()
                        with path.open('rb') as handle:
                            for block in iter(lambda: handle.read(1024*1024), b''):
                                digest.update(block)
                        dedup[key] = digest.hexdigest()
                        total += stat.st_size
                    record.update(bytes=stat.st_size, sha256=dedup[key],
                        status='FOUND' if stat.st_size == expected['bytes'] and dedup[key] == expected['sha256'] else 'HASH_MISMATCH')
                verified.append(record)
            print(json.dumps({'event': 'MODEL_MANIFEST_AUDITED', 'manifest': str(manifest_path),
                              'unique_bytes_hashed': total}), flush=True)
        envs = []
        for name in ['core', 'core_acceptance_20261014', 'models', 'reports', 'train']:
            path = runtime / 'envs' / name
            packages = []
            for dist in sorted((path / 'Lib/site-packages').glob('*.dist-info')):
                meta = dist / 'METADATA'
                if not meta.is_file():
                    continue
                headers = {}
                with meta.open(encoding='utf-8', errors='replace') as handle:
                    for line in handle:
                        if line in ('\n', '\r\n'):
                            break
                        if line.startswith(('Name: ', 'Version: ')):
                            k, v = line.rstrip().split(': ', 1)
                            headers[k] = v
                packages.append({'name': headers.get('Name'), 'version': headers.get('Version'), 'metadata': artifact(meta)})
            envs.append({'name': name, 'pyvenv': artifact(path / 'pyvenv.cfg'),
                         'python': artifact(path / 'Scripts/python.exe'), 'packages': packages})
        global_path = runtime / 'corpora/20261014_v2/queries_global_test.jsonl'
        with global_path.open(encoding='utf-8') as handle:
            global_queries = {q['query_id']: q for q in (json.loads(line) for line in handle)}
        e8 = []
        for path in sorted((runtime / 'runs/20261014_v2_retrieval/records').glob('E8_*.jsonl')):
            rows = [r for _, r in journal_rows(path)]
            values = [r['scores']['full_reference_coverage'] for r in rows if r['scores'].get('full_reference_coverage') is not None]
            e8.append({'source': artifact(path), 'n': len(rows),
                'answerability_counts': Counter(r['answerability'] for r in rows),
                'scope_counts': Counter(r['scope']['kind'] for r in rows),
                'applicable_n': len(values), 'fully_covered_n': sum(v == 1 for v in values),
                'full_reference_coverage': sum(values)/len(values) if values else None,
                'failures': sum(bool(r.get('error')) for r in rows),
                'missing_frozen_questions_n': sum(r['query_id'] not in global_queries for r in rows),
                'note': 'Partial CPU and complete GPU sources stay separate; never silently concatenate.'})
        missing = [r for r in verified if r['status'] != 'FOUND']
        result = {'status': 'PASS' if not missing else 'FAIL', 'auditor': artifact(__file__),
            'created_utc': datetime.now(timezone.utc).isoformat(), 'model_sources': [artifact(p) for p in models],
            'verified_artifacts': verified, 'unique_model_bytes_hashed': total,
            'artifact_status_counts': Counter(r['status'] for r in verified), 'environments': envs,
            'global_queries_source': artifact(global_path), 'E8': e8, 'new_llm_requests': 0,
            'verified_without_model_imports': True, 'cpu_only': True,
            'missing_global_inventory_entry_resolution': 'queries_global.jsonl is absent; preserved actual source is queries_global_test.jsonl'}
        atomic_json(output, result)
        print(json.dumps({'status': result['status'], 'verified_model_artifacts': len(verified),
                          'unique_bytes_hashed': total, 'existing_environments': len(envs), 'E8_groups': len(e8)}))
        return 0 if not missing else 2


if __name__ == '__main__':
    raise SystemExit(main())
