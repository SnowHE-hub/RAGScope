"""Read sealed auxiliary public questions without opening evaluator-only labels."""
import json
from pathlib import Path
from ragscope.indexing.identity import artifact
from ragscope.v3.splits import assert_inference_safe


def read_auxiliary_public(run, split):
    run=Path(run)
    if split=='H_scope':
        receipt=json.loads((run/'AUXILIARY_SPLIT_INPUTS.json').read_text('utf8'))
        name='H_scope.questions.jsonl'
        if artifact(receipt['parent_split_manifest']['path'])!=receipt['parent_split_manifest']:
            raise ValueError('Parent primary split changed')
    elif split=='H_tasks':
        receipt=json.loads((run/'H_TASKS_SOURCE_REPAIR.json').read_text('utf8'))
        name='H_tasks.questions_v2.jsonl'
    else:raise ValueError('Unknown auxiliary split')
    refs=[ref for ref in receipt['outputs'] if Path(ref['path']).name==name]
    if len(refs)!=1 or artifact(refs[0]['path'])!=refs[0]:raise ValueError('Public auxiliary seal differs')
    rows=[json.loads(l) for l in Path(refs[0]['path']).read_text('utf8').splitlines()]
    for row in rows:assert_inference_safe(row)
    return rows
