"""Pinned text-only Qwen3.5/Ollama chat rendering for real token budgeting."""
from pathlib import Path
from tokenizers import Tokenizer
from ragscope.indexing.identity import file_hash,signature

class Qwen35PromptCounter:
    """Restricted to the audited system+user RAG request, with thinking disabled.

    Matches Ollama v0.34.4 model/renderers/qwen35.go; pilot metadata must still
    compare this count against actual prompt_eval_count before formal use.
    """
    def __init__(self, directory, renderer_source):
        directory=Path(directory)
        self.tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
        self.identity={'tokenizer_sha256':file_hash(directory/'tokenizer.json'),
                       'official_template_sha256':file_hash(directory/'chat_template.jinja'),
                       'renderer_sha256':file_hash(renderer_source),
                       'implementation':'qwen35_two_message_no_think_v1'}
        self.template_hash=signature(self.identity)

    def render(self,messages):
        if len(messages)!=2 or [m['role'] for m in messages]!=['system','user']:
            raise ValueError('Audited renderer supports exactly system and user messages')
        if any(not isinstance(m.get('content'),str) or set(m)-{'role','content'} for m in messages):
            raise ValueError('Only plain text messages are supported')
        return ''.join('<|im_start|>'+m['role']+'\n'+m['content'].strip()+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __call__(self,messages):
        return len(self.tokenizer.encode(self.render(messages),add_special_tokens=False).ids)
