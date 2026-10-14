import json
from types import SimpleNamespace
from ragscope.generation.llm_client import LocalLLMClient
from ragscope.v3.generation_protocols import ProtocolClient
from ragscope.v3.audit import overlap


def test_g2_has_no_grammar_and_g3_sends_every_verified_sampling_option(monkeypatch, tmp_path):
    captured = []
    def request(self, path, payload):
        captured.append(payload)
        return {'message': {'content': json.dumps({'answer': 'Delta', 'citations': [], 'abstain': False})},
                'prompt_eval_count': 10, 'eval_count': 5, 'done_reason': 'stop'}
    monkeypatch.setattr(LocalLLMClient, '_request', request)
    original = LocalLLMClient('fixture', prompt_token_counter=lambda messages: 10)
    service = SimpleNamespace(base_url='http://127.0.0.1:11435', record={'backend_version': 'fixture'}, path=tmp_path/'service.json')
    # Call base chat to avoid resource threads; this is only a transport function fixture.
    free = ProtocolClient(original, service, protocol='G2')
    LocalLLMClient.chat(free, [], output_schema={'type': 'object'})
    assert 'format' not in captured[-1]
    sampling = {'options': {'top_p': .8, 'top_k': 20, 'min_p': 0.0,
                'presence_penalty': 1.5, 'repeat_penalty': 1.0}, 'model_card': {'sha256': 'fixture'}}
    sampled = ProtocolClient(original, service, protocol='G3', seed=44, sampling=sampling)
    LocalLLMClient.chat(sampled, [], temperature=.7, output_schema={'type': 'object'})
    assert captured[-1]['options']['seed'] == 44
    assert all(captured[-1]['options'][k] == v for k, v in sampling['options'].items())
    assert sampled.raw_response['message']['content']


def test_word_scoring_receives_all_references_as_a_list():
    assert overlap('Delta', ['Sigma', 'Delta'])['legacy_token_f1'] == 1
