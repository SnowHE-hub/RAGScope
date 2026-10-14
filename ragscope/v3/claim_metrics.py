"""Evidence-bound claim proxies; uncertainty is never silently scored as truth."""
from __future__ import annotations
import re

SUBJECT = r'(they|he|she|it|we|you|the (?:model|method|experiment|system|paper|authors|approach|dataset|study))'
AUX = r'(do|does|did|is|are|was|were|can|could|will|would|has|have|had)'
VERB = re.compile(r'\b(?:is|are|was|were|be|been|has|have|had|can|could|will|would|do|does|did|'
                  r'uses?|used|contains?|contained|includes?|included|evaluates?|evaluated|'
                  r'outperforms?|outperformed|achieves?|achieved|improves?|improved|'
                  r'reduces?|reduced|increases?|increased|decreases?|decreased|'
                  r'trains?|trained|reports?|reported|requires?|required|purchased|bought)\b', re.I)


def proposition(question, text, *, kind='claim'):
    """Only mechanical Q+answer completion; never receives a reference or context.

    All raw text remains separately available. Complex/incomplete statements
    are UNKNOWN. The narrow QA completion requires its own real calibration.
    """
    raw = text.strip()
    q = question.strip().rstrip('?').strip()
    if raw.lower().rstrip('.') in {'yes', 'no'}:
        match = re.fullmatch(AUX + r'\s+' + SUBJECT + r'\s+(.+)', q, re.I)
        if match:
            auxiliary, subject, remainder = match.groups()
            # Negated questions and disjunctions need interpretation, not this rule.
            if re.search(r'\b(?:not|never|or|unless)\b|n\x27t\b', q, re.I):
                return {'text': raw, 'eligible': False, 'reason': 'ambiguous_polar_question'}
            negative = ' not' if raw.lower().rstrip('.') == 'no' else ''
            value = f'{subject[0].upper() + subject[1:]} {auxiliary.lower()}{negative} {remainder}.'
            return {'text': value, 'eligible': True, 'reason': 'mechanical_polar_qa_v1', 'raw_text': text}
    numeric = re.fullmatch(r'(\d+(?:\.\d+)?)\s*\.?', raw)
    quantity = re.fullmatch(r'How many ([a-z]+) (do|does|did) ' + SUBJECT + r' (use|contain|include|have)', q, re.I)
    if numeric and quantity:
        noun, auxiliary, subject, verb = quantity.groups()
        return {'text': f'{subject[0].upper() + subject[1:]} {auxiliary.lower()} {verb} {numeric.group(1)} {noun}.',
                'eligible': True, 'reason': 'mechanical_quantity_qa_v1', 'raw_text': text}
    if kind != 'claim':
        return {'text': raw, 'eligible': False, 'reason': 'segmentation_not_independent', 'raw_text': text}
    if len(raw.split()) < 3 or not VERB.search(raw):
        return {'text': raw, 'eligible': False, 'reason': 'uncalibrated_fragment_or_grammar', 'raw_text': text}
    return {'text': raw, 'eligible': True, 'reason': 'original_declarative_candidate', 'raw_text': text}


def reference_units(text):
    """Keep original spans; do not split conditions on commas or semicolons."""
    bounds = [0] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])|\n+', text)] + [len(text)]
    units = []
    for left, right in zip(bounds, bounds[1:]):
        quote = text[left:right]
        if quote.strip():
            start = left + len(quote) - len(quote.lstrip())
            end = right - len(quote) + len(quote.rstrip())
            units.append({'start': start, 'end': end, 'source_quote': text[start:end], 'kind': 'claim'})
    return units


def bounds(verdicts):
    n = len(verdicts)
    if n == 0:
        return {'value': None, 'lower': None, 'upper': None, 'n': 0, 'supported': 0, 'unknown': 0}
    supported = verdicts.count('SUPPORTED_PROXY')
    unknown = verdicts.count('UNKNOWN')
    low, high = supported/n, (supported+unknown)/n
    return {'value': low if unknown == 0 else None, 'lower': low, 'upper': high,
            'n': n, 'supported': supported, 'unknown': unknown}


def union_relation(relations):
    if 'SUPPORTED_PROXY' in relations:
        return 'SUPPORTED_PROXY'
    if not relations or 'UNKNOWN' in relations:
        return 'UNKNOWN'
    if 'CONTRADICTED_PROXY' in relations:
        return 'CONTRADICTED_PROXY'
    return 'NOT_ENTAILED_PROXY'


def harmonic(a, b):
    if a is None or b is None:
        return None
    return 2*a*b/(a+b) if a+b else 0.0


def reference_match(question, answer, units, references, classify):
    predictions = [proposition(question, u['source_quote'], kind=u['kind']) for u in units]
    alternatives = []
    for ref in references:
        keys = [proposition(question, u['source_quote']) for u in reference_units(ref)]
        precision = bounds([classify(proposition(question, ref), p)['verdict'] for p in predictions])
        recall = bounds([classify(proposition(question, answer), k)['verdict'] for k in keys])
        alternatives.append({'precision': precision, 'recall': recall,
            'f1_lower': harmonic(precision['lower'], recall['lower']),
            'f1_upper': harmonic(precision['upper'], recall['upper'])})
    if not alternatives:
        return {'claim_reference_precision_proxy': bounds(['UNKNOWN']*len(predictions)),
                'key_fact_recall_proxy': bounds(['UNKNOWN']), 'claim_f1_proxy': None,
                'claim_f1_proxy_lower': 0.0, 'claim_f1_proxy_upper': 1.0,
                'reference_results': [], 'best_reference_index': None}
    selected = max(range(len(alternatives)), key=lambda i: (alternatives[i]['f1_lower'] or 0, alternatives[i]['f1_upper'] or 0, -i))
    best = alternatives[selected]
    return {'claim_reference_precision_proxy': best['precision'], 'key_fact_recall_proxy': best['recall'],
            'claim_f1_proxy': best['f1_lower'] if best['f1_lower'] == best['f1_upper'] else None,
            'claim_f1_proxy_lower': best['f1_lower'], 'claim_f1_proxy_upper': best['f1_upper'],
            'reference_results': alternatives, 'best_reference_index': selected}


def score_claims(question, answer, units, references, pieces, citations, classify):
    output = reference_match(question, answer, units, references, classify)
    full_context = '\n\n'.join(p['text'] for p in pieces)
    by_id = {p['chunk_id']: p['text'] for p in pieces}
    context, completeness, cited_claims, extra, flags = [], [], [], [], []
    per_citation = {cid: [] for cid in citations}
    for index, unit in enumerate(units):
        candidate = proposition(question, unit['source_quote'], kind=unit['kind'])
        result = classify({'text': full_context, 'eligible': True}, candidate)
        context.append(result['verdict'])
        ref_result = union_relation([classify(proposition(question, r), candidate)['verdict'] for r in references])
        if ref_result == 'NOT_ENTAILED_PROXY' and result['verdict'] == 'SUPPORTED_PROXY':
            extra.append(index)
        if result['verdict'] == 'CONTRADICTED_PROXY' or ref_result == 'CONTRADICTED_PROXY':
            flags.append({'unit_index': index, 'flag': 'NLI_CONTRADICTION_PROXY', 'human_review_status': 'NOT_REVIEWED'})
        relations = []
        for cid in citations:
            if cid not in by_id:
                relation = 'NOT_ENTAILED_PROXY'
            else:
                relation = classify({'text': by_id[cid], 'eligible': True}, candidate)['verdict']
            relations.append(relation); per_citation[cid].append(relation)
            cited_claims.append({'unit_index': index, 'citation': cid, 'verdict': relation})
        completeness.append(union_relation(relations) if citations else 'NOT_ENTAILED_PROXY')
    output.update(context_support_precision_proxy=bounds(context),
        citation_support_precision_proxy=bounds([union_relation(v) for v in per_citation.values()]),
        citation_completeness_proxy=bounds(completeness), supported_extra=extra,
        critical_error_flags=flags, citation_claim_relations=cited_claims,
        citation_association_rule='legacy answer-level citations; support at least one claim, not fabricated inline links',
        prediction_units=units, proposition_auxiliaries=[proposition(question, u['source_quote'], kind=u['kind']) for u in units])
    return output


def retained_failure_metrics(reason):
    """Applicable QA failure/abstention stays zero, factual assertions are N/A."""
    zero = bounds(['NOT_ENTAILED_PROXY'])
    return {'claim_reference_precision_proxy': zero, 'key_fact_recall_proxy': zero,
            'claim_f1_proxy': 0.0, 'claim_f1_proxy_lower': 0.0, 'claim_f1_proxy_upper': 0.0,
            'context_support_precision_proxy': bounds([]), 'citation_support_precision_proxy': bounds([]),
            'citation_completeness_proxy': bounds([]), 'critical_error_flags': [],
            'supported_extra': [], 'status': reason}
