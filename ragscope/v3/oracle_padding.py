"""Same-paper, gold-excluded diagnostic padding with intact original evidence."""
from dataclasses import asdict
import copy
import re
from ragscope.generation.context import pack_context, reference_token_count, render_context
from ragscope.indexing.identity import signature

STOP = set('a an the is are was were do does did they their this that it of in on to for and or how what which why with use used'.split())


def padded_context(question, clean, retrieval, document, gold_ids, scope, *, counter=reference_token_count):
    target = max(counter(render_context(clean)), counter(render_context(retrieval)))
    core = copy.deepcopy(clean['pieces'])
    if not core:
        raise ValueError('Only answerable nonempty Oracle cases are eligible')
    qwords = set(re.findall(r'[a-z]+', question.lower()))-STOP
    units = [u for u in document['evidence_units'] if u['unit_id'] not in gold_ids and u['kind'] == 'paragraph']
    units.sort(key=lambda u: (len(qwords & set(re.findall(r'[a-z]+', u['text'].lower())))/max(1, len(qwords)), signature([42, u['unit_id']])))
    noise = []
    for unit in units:
        candidate = {'chunk_id': 'oracle:'+unit['unit_id'], 'doc_id': document['doc_id'], 'text': unit['text'],
            'canonical_start': unit['canonical_start'], 'canonical_end': unit['canonical_end'],
            'unit_spans': [{'doc_id': document['doc_id'], 'unit_id': unit['unit_id'],
                            'start': unit['canonical_start'], 'end': unit['canonical_end']}]}
        current = {'pieces': core+noise}
        if counter(render_context(current)) >= target:
            break
        if counter(render_context({'pieces': core+noise+[candidate]})) > target:
            low, high = 0, len(candidate['text'])
            while low < high:
                middle = (low+high+1)//2
                shortened = {**candidate, 'text': candidate['text'][:middle]}
                if counter(render_context({'pieces': core+noise+[shortened]})) <= target:
                    low = middle
                else:
                    high = middle-1
            if low < 40:
                continue
            # Keep complete source sentences; do not clip away a condition or negation.
            boundaries = [m.end() for m in re.finditer(r'[.!?](?:["\u201d\u2019\)]*)(?=\s|$)', candidate['text'][:low])]
            low = boundaries[-1] if boundaries else 0
            if low < 40:
                continue
            candidate['text'] = candidate['text'][:low]
            candidate['canonical_end'] = candidate['canonical_start']+low
            candidate['unit_spans'][0]['end'] = candidate['canonical_end']
        noise.append(candidate)
    # Match the location of the first annotated evidence in retrieval when available.
    first = next((i for i,p in enumerate(retrieval['pieces']) if any(u['unit_id'] in gold_ids for u in p.get('unit_spans', []))), None)
    desired = counter(render_context({'pieces': retrieval['pieces'][:first]})) if first is not None else target//2
    position = min(range(len(noise)+1), key=lambda i: abs(counter(render_context({'pieces': noise[:i]}))-desired))
    ordered = noise[:position]+core+noise[position:]
    # Account for token merges/separators after reordering without losing Oracle text.
    allowance = max(target, counter(render_context({'pieces': ordered})))
    bundle = asdict(pack_context(ordered, max_tokens=allowance, token_counter=counter, scope=scope))
    actual = {p['chunk_id']:p['text'] for p in bundle['pieces']}
    if any(actual.get(p['chunk_id']) != p['text'] for p in core):
        raise ValueError('Padding changed or dropped original Oracle evidence')
    return bundle, {'target_reference_tokens': target, 'actual_reference_tokens': counter(render_context(bundle)),
        'original_evidence_characters_unchanged': True, 'noise_unit_ids': [p['unit_spans'][0]['unit_id'] for p in noise],
        'first_evidence_retrieval_index': first, 'padding_pieces_before_evidence': position,
        'length_difference_reference_tokens': counter(render_context(bundle))-target,
        'padding_uses_complete_source_sentences': True,
        'noise_selection': 'same-paper non-annotated paragraphs, smallest lexical overlap; semantic unrelatedness is proxy, not human verified',
        'scope_changed': False}
