"""A separately frozen E2 journal keeps its actual immutable provenance."""
from pathlib import Path
import pytest

from ragscope.indexing.identity import artifact
from ragscope.routing.campaign import route_record_artifacts


def sources(directory, experiment='E2', external=True):
    paths = [directory/'records'/f'{experiment}_qwen_2b.jsonl',
             directory/'records'/f'{experiment}_qwen_2b.config.json']
    if external:
        paths += [directory/'EXPERIMENT_MANIFEST.json', directory/'FROZEN_INPUTS.json']
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n', encoding='utf-8')
    return [artifact(path) for path in paths]


def test_external_e2_owner_is_used_without_fabricating_main_journal(tmp_path):
    items = sources(tmp_path/'e2')
    result = route_record_artifacts(tmp_path/'main', items, 'E2', 'qwen_2b')
    assert result == items
    assert not (tmp_path/'main/records/E2_qwen_2b.jsonl').exists()


def test_existing_train_dev_provenance_does_not_change(tmp_path):
    items = sources(tmp_path/'main', 'ROUTE_TRAIN', external=False)
    assert route_record_artifacts(tmp_path/'main', items, 'ROUTE_TRAIN', 'qwen_2b') == items


@pytest.mark.parametrize('corruption', ['missing', 'ambiguous', 'config', 'manifest', 'inputs'])
def test_missing_or_ambiguous_owner_is_rejected(tmp_path, corruption):
    items = sources(tmp_path/'e2')
    if corruption == 'missing': items.pop(0)
    elif corruption == 'ambiguous': items += sources(tmp_path/'other')
    else:
        suffix = {'config': '.config.json', 'manifest': 'EXPERIMENT_MANIFEST.json', 'inputs': 'FROZEN_INPUTS.json'}[corruption]
        items = [item for item in items if not Path(item['path']).name.endswith(suffix)]
    with pytest.raises(ValueError): route_record_artifacts(tmp_path/'main', items, 'E2', 'qwen_2b')
