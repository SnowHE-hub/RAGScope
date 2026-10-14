from dataclasses import asdict
from ragscope.generation.context import pack_context, render_context
from ragscope.v3.oracle_padding import padded_context


def test_padding_keeps_gold_evidence_and_actual_canonical_noise_coordinates():
    source = 'The answer is Delta. Unrelated background is here. Additional history is here.'
    strings = ['The answer is Delta.', 'Unrelated background is here.', 'Additional history is here.']
    units = []
    for i, text in enumerate(strings):
        start = source.index(text)
        units.append({'unit_id': 'u'+str(i), 'kind': 'paragraph', 'text': text,
                      'canonical_start': start, 'canonical_end': start+len(text)})
    doc = {'doc_id': 'd', 'text': source, 'evidence_units': units}
    core = {'chunk_id': 'oracle:u0', 'doc_id': 'd', 'text': strings[0], 'canonical_start': 0,
            'canonical_end': len(strings[0]), 'unit_spans': [{'unit_id': 'u0', 'doc_id': 'd', 'start': 0, 'end': len(strings[0])}]}
    scope = {'kind': 'document', 'doc_ids': ['d']}
    clean = asdict(pack_context([core], token_counter=len, scope=scope))
    retrieval = {'pieces': [{'chunk_id': 'r', 'doc_id': 'd', 'text': source}]}
    padded, audit = padded_context('What is the answer?', clean, retrieval, doc, {'u0'}, scope, counter=len)
    assert next(p['text'] for p in padded['pieces'] if p['chunk_id'] == 'oracle:u0') == strings[0]
    assert 'u0' not in audit['noise_unit_ids']
    assert all(source[p['canonical_start']:p['canonical_end']] == p['text'] for p in padded['pieces'])
