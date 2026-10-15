"""Independently recount uncached online requests; never fill missing latency with zero."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import numpy as np
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.audit import overlap, reference_texts
from ragscope.v3.splits import read_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    a = p.parse_args(); run = Path(a.campaign).resolve()
    if run.parent != Path('D:/RAGScope_Runtime/runs').resolve():
        raise ValueError('Wrong campaign')
    path = run/'S6_ONLINE_COST.jsonl'
    config = json.loads(path.with_suffix('.config.json').read_text('utf8'))
    rows = read_records(path); warm = read_records(run/'S6_ONLINE_WARMUPS.jsonl')
    ids = config['query_ids']; names = config['conditions']
    expected = {(qid, c, repeat) for qid in ids for c in names for repeat in range(3)}
    actual = set(); grouped = defaultdict(list); paired = defaultdict(dict)
    labels = {q['query_id']: q for q in read_split(run, 'D_confirm', purpose='development')}
    events = [json.loads(l) for l in (run/'ONLINE_EMBEDDING_REQUESTS.jsonl').read_text('utf8').splitlines()]
    embeds = defaultdict(list)
    for e in events: embeds[e['request_id']].append(e)
    for row in rows:
        key = (row['question_id'], row['condition'], row['repeat'])
        if key not in expected or key in actual or row['config_hash'] != signature(config):
            raise ValueError('Online population or immutable config differs')
        actual.add(key)
        if row['answer_cache'] or row['embedding_cache']:
            raise ValueError('Main online measurements cannot contain application cache hits')
        if [e['event'] for e in embeds[row['budget_ticket']]] != ['RESERVED', 'COMPLETED']:
            raise ValueError('Every measured request needs an actual uncached query encoding')
        resident = row['residency_before']
        if len(resident) != 2 or any(m['size_vram'] <= 0 for m in resident):
            raise ValueError('Encoder and generator residency not established')
        if row['success']:
            parsed = RAGPipeline.parse_output(row['raw_response']['content'])
            if {k: parsed[k] for k in ('answer', 'citations', 'abstain')} != row['parsed_response']:
                raise ValueError('Parser/raw response mismatch')
        q = labels[row['question_id']]; response = row['parsed_response']
        can = q['answerability'] == 'answerable'
        f1 = (overlap(response['answer'], reference_texts(q))['legacy_token_f1']
              if row['success'] and response and not response['abstain'] else 0.0) if can else None
        item = {'query_id': q['query_id'], 'paper_id': q['doc_id'], 'repeat': row['repeat'],
                'condition': row['condition'], 'seconds': row['wall_time_ms']/1000,
                'word_f1': f1, 'failure': not row['success'],
                'answerable_abstention': can and bool(response and response['abstain']),
                'unanswerable_correct_abstention': not can and bool(response and response['abstain']),
                'source_record_hash': row['record_hash']}
        grouped[row['condition']].append(item)
        paired[(q['query_id'], row['repeat'])][row['condition']] = item
    summaries = {}
    for name, items in grouped.items():
        latencies = [i['seconds'] for i in items]; can = [i for i in items if i['word_f1'] is not None]
        summaries[name] = {'formal_n': len(items), 'planned_n': 90,
            'failures_n': sum(i['failure'] for i in items), 'answerable_request_n': len(can),
            'answerable_abstentions_n': sum(i['answerable_abstention'] for i in items),
            'unanswerable_correct_abstentions_n': sum(i['unanswerable_correct_abstention'] for i in items),
            'legacy_token_f1': statistics.mean(i['word_f1'] for i in can) if can else None,
            'online_p50_seconds': float(np.quantile(latencies, .5)),
            'online_p95_seconds': float(np.quantile(latencies, .95)),
            'online_p99_exploratory_seconds': float(np.quantile(latencies, .99)),
            'claim_f1_proxy': None, 'context_support_proxy': None, 'citation_completeness_proxy': None}
    complete_pairs = [v for v in paired.values() if set(v) == set(names)]
    by_paper = defaultdict(list)
    for pair in complete_pairs: by_paper[pair[names[0]]['paper_id']].append(pair)
    keys = sorted(by_paper); rng = np.random.default_rng(42); p50_ratios = []; p95_ratios = []
    for _ in range(2000):
        sample = [pair for pid in rng.choice(keys, len(keys), replace=True) for pair in by_paper[pid]]
        b = [v[names[0]]['seconds'] for v in sample]; c = [v[names[1]]['seconds'] for v in sample]
        p50_ratios.append(float(np.quantile(c, .5)/np.quantile(b, .5)))
        p95_ratios.append(float(np.quantile(c, .95)/np.quantile(b, .95)))
    result = {'status': 'PASS_180_INDEPENDENT_ONLINE_AUDIT' if actual == expected else 'PARTIAL_ONLINE_MISSING_REQUESTS',
        'formal_n': len(rows), 'warmup_n': len(warm), 'missing_requests': [list(x) for x in sorted(expected-actual)],
        'groups': summaries, 'paired_requests_n': len(complete_pairs), 'papers_n': len(keys),
        'p50_candidate_baseline_ratio_ci95': np.quantile(p50_ratios, [.025, .975]).tolist(),
        'p95_candidate_baseline_ratio_ci95': np.quantile(p95_ratios, [.025, .975]).tolist(),
        'bootstrap_repeats': 2000, 'source': artifact(path), 'warmups': artifact(run/'S6_ONLINE_WARMUPS.jsonl'),
        'quality_noninferiority': 'NOT_PROVED_PRIMARY_UNKNOWN', 'candidate_admitted': False,
        'cost_scope': config['interpretation'], 'first_token_seconds': None,
        'human_review_status': 'NOT_REVIEWED', 'baseline_qualification': config['baseline_qualification'],
        'limitations': ['Development efficiency screen, not unseen quality proof.',
            'Both policies use the qualified GPU query-encoder bridge, not the old CPU online baseline.',
            'Backend internal prompt reuse may exist; application answer and embedding caches are disabled.',
            'Missing requests have no invented latency or quality. Repeats are clustered by source paper.']}
    atomic_json(run/'S6_ONLINE_COST_ANALYSIS.json', result)
    print(canonical_json({k: result[k] for k in ('status','formal_n','warmup_n','groups','candidate_admitted')}))
    return 0 if actual == expected else 3


if __name__ == '__main__': raise SystemExit(main())
