import json
from scripts.v3.run_lora_runtime_diagnostics import selected_inputs


def test_runtime_inputs_are_fixed_dev_only_and_do_not_read_gold(tmp_path):
    rows=[{'dataset_role':'D_confirm','condition':'B0_policy_neighbor_bridge','query_id':f'q{i:02d}'} for i in range(30)]
    rows += [{'dataset_role':'H_QA','condition':'B0_policy_neighbor_bridge','query_id':'sealed'},
             {'dataset_role':'D_confirm','condition':'C1_policy_core_first_bridge','query_id':'alternate'}]
    (tmp_path/'S3_SCOPED_CONTEXTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reversed(rows)),encoding='utf8')
    selected=selected_inputs(tmp_path)
    assert [r['query_id'] for r in selected]==[f'q{i:02d}' for i in range(24)]
    assert all(r['dataset_role']=='D_confirm' for r in selected)
    assert list(tmp_path.iterdir())==[tmp_path/'S3_SCOPED_CONTEXTS.jsonl']
