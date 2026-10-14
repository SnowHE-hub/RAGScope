"""Preserve raw results; repair one proven reused-envelope hash in a new journal."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import RecordStore,read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    first={x['record_id']:x for x in read_records(run/'S2_SUPPLEMENT_RESULTS.jsonl')}
    source=run/'S2_SUPPLEMENT_RESULTS_v2.jsonl';out=run/'S2_SUPPLEMENT_RESULTS_verified.jsonl'
    if out.exists():raise ValueError('Verified projection exists; do not overwrite')
    rows=[];fixed=[]
    with source.open('rb') as handle:
        for number,line in enumerate(handle,1):
            if not line.endswith(b'\n'):raise ValueError('Incomplete source line')
            row=json.loads(line);saved=row.pop('record_hash')
            if signature(row)!=saved:
                previous=first.get(row['record_id'])
                if previous is None or not row.get('reused_actual_result'):raise ValueError('Unexplained corrupt row')
                # RecordStore.add hashed the still-present old record_hash before replacing it.
                if signature({**row,'record_hash':previous['record_hash']})!=saved:raise ValueError('Hash mismatch is not the documented wrapper defect')
                unchanged=[k for k in previous if k not in {'record_hash','config_hash','recorded_at'}]
                if any(row[k]!=previous[k] for k in unchanged):raise ValueError('Original inference fields changed')
                fixed.append({'source_line':number,'record_id':row['record_id'],'old_record_hash':previous['record_hash'],
                    'invalid_envelope_hash':saved,'raw_inference_fields_exactly_equal':True})
            rows.append((number,row,saved))
    if len(rows)!=42 or len(fixed)!=1:raise ValueError('Unexpected repair scope')
    config={'source_journal':artifact(source),'source_config':artifact(source.with_suffix('.config.json')),
        'original_first_attempt':artifact(run/'S2_SUPPLEMENT_RESULTS.jsonl'),'runner':artifact(__file__),
        'repair_scope':'one reused envelope only; no raw output, request, score or inference identity modifications'}
    store=RecordStore(out,config,max_records=42)
    for number,row,saved in rows:
        store.add(row['record_id'],{**row,'source_envelope_hash':saved,'source_line':number,
            'source_config_hash':row['config_hash'],'source_recorded_at':row['recorded_at'],
            'raw_inference_unchanged':True})
    if len(read_records(out))!=42:raise ValueError('Projection hash validation failed')
    result={'status':'PASS_VERIFIED_PROJECTION','repaired_envelopes':fixed,'record_n':42,
        'source':artifact(source),'verified':artifact(out),'new_llm_calls':0,'originals_overwritten':False}
    atomic_json(run/'S2_SUPPLEMENT_PROJECTION_VERIFICATION.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
