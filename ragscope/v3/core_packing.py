"""Preserve ranked cores before optional same-document neighbor expansion."""
from dataclasses import asdict
from ragscope.generation.context import pack_context,reference_token_count,render_context


def pack_core_first(chunks, *, max_tokens=2048, scope=None, chunk_lookup=None, token_counter=None):
    count=token_counter or reference_token_count
    lookup=chunk_lookup or {}
    # Up to the entire budget can serve cores; neighbors never displace a core.
    core=asdict(pack_context(chunks,max_tokens=max_tokens,scope=scope,token_counter=count))
    core_tokens=count(render_context(core))
    neighbor_cap=min(max_tokens//4,max_tokens-core_tokens)
    candidates=[]
    for piece in core['pieces']:
        original=lookup.get(piece.get('source_chunk_id',piece['chunk_id']),piece)
        for key in (original.get('prev_chunk_id'),original.get('next_chunk_id')):
            neighbor=lookup.get(key)
            if neighbor is not None and neighbor['doc_id']==piece['doc_id']:
                candidates.append(neighbor)
    combined=asdict(pack_context(core['pieces']+candidates,max_tokens=core_tokens+neighbor_cap,
        scope=scope,token_counter=count))
    bundle=pack_context(combined['pieces'],max_tokens=max_tokens,scope=scope,token_counter=count)
    final=asdict(bundle)
    byid={p['chunk_id']:p['text'] for p in final['pieces']}
    if any(byid.get(p['chunk_id'])!=p['text'] for p in core['pieces']):
        raise ValueError('Neighbor packing changed a selected core')
    used=count(render_context(final))-core_tokens
    if used>neighbor_cap:
        raise ValueError('Neighbor exceeded predeclared 25 percent cap')
    return bundle,{'core_budget_allocated':max_tokens,'core_reference_tokens':core_tokens,
        'neighbor_budget_allocated':neighbor_cap,'neighbor_incremental_reference_tokens':used,
        'core_piece_n':len(core['pieces']),'final_piece_n':len(final['pieces']),
        'core_unchanged':True,'packing_policy':'ranked full cores first; unused budget at most 25% for neighbors',
        'core_truncations':core['truncations'],'neighbor_truncations':combined['truncations']}
