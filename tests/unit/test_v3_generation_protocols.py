import pytest
from ragscope.v3.generation_protocols import card_sampling, protocol_messages, protocol_schema
from ragscope.generation.prompts import RAG_SYSTEM_PROMPT


def test_g2_changes_constraint_without_changing_g1_messages():
    bundle = {'pieces': []}
    assert protocol_messages('A question?', bundle, 'G1') == protocol_messages('A question?', bundle, 'G2')
    assert protocol_messages('A question?', bundle, 'G0')[0]['content'] == RAG_SYSTEM_PROMPT
    assert protocol_schema(bundle, 'G2') is None
    assert protocol_schema(bundle, 'G1')['properties']['citations']['maxItems'] == 0


def test_model_card_text_recommendation_and_backend_parameter_mapping(tmp_path):
    path = tmp_path/'README.md'
    path.write_text('**Non-thinking mode for text tasks**:\n'
        '`temperature=1.0`, `top_p=1.00`, `top_k=20`, `min_p=0.0`, `presence_penalty=2.0`, `repetition_penalty=1.0`\n'
        '**Non-thinking mode for VL tasks**:\n'
        '`temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`', encoding='utf-8')
    result = card_sampling(path, 'Qwen/Qwen3.5-2B')
    assert result['temperature'] == 1.0 and result['options']['presence_penalty'] == 2.0
    assert result['options']['repeat_penalty'] == 1.0 and 'repetition_penalty' not in result['options']
    path.write_text('Non-thinking mode for text tasks: `temperature=1`', encoding='utf-8')
    with pytest.raises(ValueError):
        card_sampling(path, 'Qwen/Qwen3.5-2B')


def test_adjacent_reasoning_advice_does_not_overwrite_general_sampling(tmp_path):
    path = tmp_path/'README.md'
    path.write_text('Instruct (or non-thinking) mode for general tasks: '
        '`temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`\n'
        'Instruct (or non-thinking) mode for reasoning tasks: '
        '`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`', encoding='utf-8')
    result = card_sampling(path, 'Qwen/Qwen3.5-9B')
    assert result['temperature'] == .7 and result['options']['top_p'] == .8
