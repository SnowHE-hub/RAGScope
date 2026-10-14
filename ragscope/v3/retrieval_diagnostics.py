"""Offline interval coverage and alternative aggregation of preserved scores."""
from collections import defaultdict


def coverage(pieces, references, units):
    intervals = defaultdict(list)
    for piece in pieces:
        for span in piece.get('unit_spans', []):
            intervals[span['unit_id']].append((span['start'],span['end']))
    values = []
    for reference in references:
        fractions = []
        for uid in reference:
            unit = units[uid]
            lo,hi = unit['canonical_start'],unit['canonical_end']
            clipped = sorted((max(lo,a),min(hi,b)) for a,b in intervals[uid] if max(lo,a)<min(hi,b))
            covered, end = 0,lo
            for a,b in clipped:
                covered += max(0,b-max(a,end))
                end = max(end,b)
            fractions.append(covered/(hi-lo))
        values.append({'continuous_coverage':sum(fractions)/len(fractions) if fractions else 0.0,
            'full_unit_recall':sum(x==1.0 for x in fractions)/len(fractions) if fractions else 0.0,
            'full_reference_coverage':float(bool(fractions) and all(x==1.0 for x in fractions))})
    return {key:max((x[key] for x in values),default=0.0) for key in
        ('continuous_coverage','full_unit_recall','full_reference_coverage')}


def top2_ranking(candidates, windows, k=10):
    scores = defaultdict(list)
    for window in windows:
        scores[window['chunk_id']].append(window['score'])
    if set(scores) != {p['chunk_id'] for p in candidates}:
        raise ValueError('Window scores do not cover exact candidate IDs')
    aggregated = {cid:sum(sorted(s,reverse=True)[:2])/min(2,len(s)) for cid,s in scores.items()}
    ranked = sorted(candidates,key=lambda p:(-aggregated[p['chunk_id']],p['chunk_id']))[:k]
    return [{**p,'rerank_score':aggregated[p['chunk_id']],'aggregation':'top2_mean'} for p in ranked]
