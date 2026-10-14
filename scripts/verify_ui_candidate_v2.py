"""Exercise the real offline UI candidate; no model calls, server, or publication."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, verify_artifacts


@contextmanager
def offline_transport():
    import requests
    attempts = []
    def denied(*args, **kwargs):
        attempts.append('HTTP request refused by offline UI QA')
        raise RuntimeError(attempts[-1])
    # Windows asyncio needs a local socketpair to create its event loop. Blocking
    # that framework self-pipe does not test model isolation; block both HTTP
    # transports actually used by the application instead.
    with patch.object(requests.sessions.Session, 'request', denied), patch.object(urllib.request.OpenerDirector, 'open', denied):
        yield attempts


def verify(candidate, receipt, output):
    """Readiness is the captured candidate snapshot, never a new live claim."""
    from streamlit.testing.v1 import AppTest
    from ui.chat_app import read_ui_config, ready_options, validate_index_entry
    configure(cpu_only=True)
    require_ram(512*1024**2, reserve_bytes=1280*1024**2)
    candidate, receipt, output = Path(candidate), Path(receipt), Path(output)
    if output.exists():
        raise FileExistsError('Preserve the previous UI QA receipt')
    previous = json.loads(receipt.read_text(encoding='utf-8'))
    if previous['candidate'] != artifact(candidate) or not verify_artifacts(previous['sources']):
        raise ValueError('Candidate or its frozen provenance changed')
    config = read_ui_config(candidate)
    if (config['default_model'] != 'qwen_2b' or config['default_router'] != 'fixed'
            or config['routers'] or config['reference_budget'] != 2048):
        raise ValueError('Offline QA requires the existing fixed dev-only candidate')
    indexes = ready_options(config['indexes'])
    if len(indexes) != 1 or indexes[0]['method'] != 'bm25' or ready_options(config['models']):
        raise ValueError('This offline QA is for the services-unavailable candidate')
    manifest = validate_index_entry(indexes[0])
    if manifest['count'] != 6551 or len(manifest['identity']['selected_doc_ids']) != 509:
        raise ValueError('Expected the completed all509 structured index')
    published = artifact(ROOT/'configs/v2/ui.json')
    question = 'What problem does this paper address?'
    with patch.dict(os.environ, {'RAGSCOPE_UI_CONFIG': str(candidate)}), offline_transport() as attempts:
        app = AppTest.from_file(str(ROOT/'ui/chat_app.py'), default_timeout=60).run()
        if app.exception or app.error:
            raise ValueError('Initial page failed to render')
        choices = {element.label: element.value for element in app.selectbox}
        if choices.get('生成模型') != '仅检索证据' or choices.get('检索配置') != indexes[0]['id']:
            raise ValueError('Offline application exposed an unavailable model/path')
        next(item for item in app.text_input if item.label == '问题').set_value(question)
        next(item for item in app.button if item.label == '运行').click().run()
        if app.exception or app.error:
            raise ValueError('Actual scoped BM25 UI request failed')
        result = app.session_state['live_result']
        bundle = result['context_bundle']
        if (not bundle['pieces'] or any(p['doc_id'] != config['default_scope'] for p in bundle['pieces'])
                or bundle['budget']['reference_tokens'] > 2048):
            raise ValueError('Real scoped context or budget invalid')
        next(item for item in app.button if item.label == '加载并校验记录').click().run()
        if app.exception or app.error or not any('已验证 100 条记录' in e.value for e in app.caption):
            raise ValueError('Completed dev100 record replay failed')
        if attempts:
            raise ValueError('Offline UI attempted a network request')
        result_path = output.with_suffix('.retrieval.json')
        atomic_json(result_path, result)
        report = {'status': 'PASS_OFFLINE_UI_CANDIDATE', 'time': datetime.now(timezone.utc).isoformat(),
            'candidate': artifact(candidate), 'candidate_verification': artifact(receipt),
            'implementation': [artifact(__file__), artifact(ROOT/'ui/chat_app.py')],
            'index_signature': manifest['signature'], 'documents': 509, 'chunks': 6551,
            'question_source': 'agent-authored engineering smoke question; not a quality label',
            'question': question, 'scope': config['default_scope'], 'initial_choices': choices,
            'actual_retrieval_result': artifact(result_path), 'dev_replay_rows': 100,
            'http_attempts': attempts, 'model_calls': 0, 'server_started': False,
            'torch_imported': 'torch' in sys.modules, 'published_config_unchanged': artifact(ROOT/'configs/v2/ui.json') == published,
            'readiness_scope': 'historical candidate snapshot; not fresh service availability',
            'limitations': ['No browser pixel inspection', 'No neural retrieval or generation request',
                            'No nonfixed routing enabled', 'Final publication remains coordinator-owned']}
        if report['torch_imported'] or not report['published_config_unchanged']:
            raise ValueError('Offline isolation or project configuration preservation failed')
        atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = 'D:/RAGScope_Runtime/runs/20261014_v2/ui_candidate/'
    parser.add_argument('--candidate', default=directory+'ui.json')
    parser.add_argument('--receipt', default=directory+'ui.verification.json')
    parser.add_argument('--output', default=directory+'OFFLINE_STARTUP_QA.json')
    args = parser.parse_args()
    result = verify(args.candidate, args.receipt, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'documents', 'chunks', 'dev_replay_rows', 'model_calls')}))


if __name__ == '__main__':
    main()
