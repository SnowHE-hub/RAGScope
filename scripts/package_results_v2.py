"""Create a local, allowlisted, weight-free result bundle after acceptance.

No upload, repository archive, user-file scan, or original-job snapshot is used.
Portable JSON exports explicitly keep original hashes and name their path-only
redaction; they are not substituted for the original immutable journals.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragscope.experiments.analysis_report import validated_rows
from ragscope.experiments.generation_campaign import load_campaign, iter_record_sources
from ragscope.experiments.runtime import runtime_home
from ragscope.indexing.identity import artifact, atomic_json


def portable(value, project, home):
    """Redact known path prefixes without inspecting any credential locations."""
    if isinstance(value, dict):
        return {key: portable(item, project, home) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, project, home) for item in value]
    if isinstance(value, str):
        for path, label in ((project, '<PROJECT>'), (home, '<RUNTIME>'),
                            (Path.home(), '<USER_HOME>')):
            for spelling in {str(path), str(path).replace('\\', '/')}:
                value = value.replace(spelling, label)
        return value
    return value


def build_bundle(campaign, report_dir, output):
    project, home = Path(__file__).resolve().parents[1], runtime_home()
    campaign, report_dir, output = campaign.resolve(), report_dir.resolve(), output.resolve()
    if not campaign.is_relative_to(home/'runs') or not output.is_relative_to(home):
        raise ValueError('Campaign and exported bundle must remain under the project runtime')
    if report_dir != project/'reports/v2' and not report_dir.is_relative_to(home/'runs'):
        raise ValueError('Report directory must be project reports/v2 or a runtime run')
    if output.exists():
        raise FileExistsError('Existing bundles are preserved; choose a new output name')
    acceptance = json.loads((report_dir/'ACCEPTANCE.json').read_text(encoding='utf-8-sig'))
    criteria = acceptance['criteria']
    if len(criteria) != 15 or len({row['criterion_id'] for row in criteria}) != 15:
        raise ValueError('All15 acceptance criteria must be present')
    if any(row['status'] in {'PENDING', 'RUNNING', 'NOT_RUN'} for row in criteria):
        raise ValueError('Unfinished acceptance cannot be packaged as a final result')
    required = ('criterion_id', 'status', 'evidence_paths', 'command', 'exit_code', 'observed_value', 'limitations')
    if any(any(key not in row for key in required) for row in criteria):
        raise ValueError('Acceptance lacks required evidence fields')
    if not (report_dir/'FINAL_REPORT.md').is_file():
        raise ValueError('Final report is required')
    manifest, inputs = load_campaign(campaign)
    records, sources = validated_rows(campaign, manifest, inputs)
    allowed = ['FINAL_REPORT.md', 'ACCEPTANCE.json', 'GENERATION_TABLES.md', 'ANALYSIS.json',
               'ERROR_CASES.md', 'ERROR_CASES.json', 'ROUTING_REPORT.md', 'TRAINING_REPORT.md',
               'RESOURCE_REPORT.md', 'RETRIEVAL_TABLES.md']
    members = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        def add(name, data, original=None, transformation='none'):
            bundle.writestr(name, data)
            members.append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                            'original': original, 'transformation': transformation})
        def encode(value):
            return (json.dumps(portable(value, project, home), ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
        for name in allowed:
            path = report_dir/name
            if not path.is_file():
                continue
            if path.is_symlink() or path.stat().st_size > 16*1024**2:
                raise ValueError('Report member is a symlink or exceeds16MiB: '+name)
            if path.suffix == '.json':
                payload = encode(json.loads(path.read_text(encoding='utf-8-sig')))
            else:
                payload = portable(path.read_text(encoding='utf-8-sig'), project, home).encode('utf-8')
            add('reports/'+name, payload, portable(artifact(path), project, home), 'known path prefixes redacted')
        for path in sorted((report_dir/'figures').glob('*')):
            if path.suffix not in {'.png', '.svg', '.json'} or path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_size > 16*1024**2:
                raise ValueError('Figure exceeds16MiB')
            payload = path.read_bytes() if path.suffix == '.png' else portable(path.read_text(encoding='utf-8'), project, home).encode('utf-8')
            add('reports/figures/'+path.name, payload, portable(artifact(path), project, home),
                'none' if path.suffix == '.png' else 'known path prefixes redacted')
        for number, source in enumerate(iter_record_sources(campaign, manifest, inputs)):
            add(f'provenance/campaign_{number}/EXPERIMENT_MANIFEST.json', encode(source['manifest']),
                portable(artifact(source['directory']/'EXPERIMENT_MANIFEST.json'), project, home), 'known path prefixes redacted')
        add('results/FROZEN_INPUTS.portable.json', encode(inputs), transformation='known path prefixes redacted')
        # Exports are clearly named, retain original record_hash, and never rewrite
        # or impersonate the original RecordStore/config identity.
        for model in sorted(manifest['models']):
            selected = [row for row in records if row['model_id'] == model]
            payload = b''.join(encode({'original_record_hash': row['record_hash'], 'exported_record': row}) for row in selected)
            add(f'results/{model}.portable.jsonl', payload, transformation='validated analysis projection; known paths redacted')
        for name in ('analysis.json', 'retrieval_statistics.json', 'routing.json', 'lora.json', 'chart_contracts.json'):
            path = project/'configs/v2'/name
            add('protocols/'+name, path.read_bytes(), portable(artifact(path), project, home))
        readme = ('# Local RAGScope v2 result bundle\n\n'
                  'This package contains public benchmark result exports and final reports, not weights, environments, '
                  'credentials, original-job snapshots, or private project files. Nothing was uploaded.\n\n'
                  'Portable exports redact known local path prefixes. Their hashes differ from immutable original journals; '
                  'original_record_hash and source artifacts identify those originals. Do not load these projections as RecordStore journals. '
                  'Failure rows and incomplete/blocked statuses are retained. Full canonical public corpora are not bundled.\n')
        add('README.md', readme.encode())
        export = {'version': 'local-results-package-v1', 'records': len(records), 'planned_tasks': len(manifest['tasks']),
                  'members': members.copy(), 'original_record_sources': portable(sources, project, home),
                  'contains_weights': False, 'uploaded': False, 'compression_limit_bytes': 64*1024**2}
        add('PACKAGE_MANIFEST.json', encode(export))
    receipt = {'status': 'PASS' if output.stat().st_size <= 64*1024**2 else 'OVERSIZE_REVIEW_REQUIRED',
               'artifact': artifact(output), 'members': len(members), 'observed_records': len(records),
               'planned_tasks': len(manifest['tasks']), 'uploaded': False}
    atomic_json(output.with_suffix('.receipt.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--report-dir', type=Path, default=Path('reports/v2'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build_bundle(args.campaign, args.report_dir, args.output)
    print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
