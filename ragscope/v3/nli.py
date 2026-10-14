"""Pinned NLI with full premise windows and explicit uncertainty.

NLI is a proxy. Neutral is not factual falsehood, and separately inconclusive
windows are not proof that their combined evidence is insufficient.
"""
from __future__ import annotations
import gc


def label_map(id2label):
    result = {int(k): str(v).strip().casefold() for k, v in id2label.items()}
    if set(result.values()) != {'entailment', 'neutral', 'contradiction'} or len(result) != 3:
        raise ValueError('NLI id2label is not an explicit three-relation mapping')
    return result


def aggregate_windows(windows, threshold=0.8):
    if not windows:
        return 'UNKNOWN'
    entailed = any(w['probabilities']['entailment'] >= threshold for w in windows)
    contradicted = any(w['probabilities']['contradiction'] >= threshold for w in windows)
    if entailed and contradicted:
        return 'UNKNOWN'
    if entailed:
        return 'SUPPORTED_PROXY'
    if contradicted:
        return 'CONTRADICTED_PROXY'
    if len(windows) == 1 and windows[0]['probabilities']['neutral'] >= threshold:
        return 'NOT_ENTAILED_PROXY'
    return 'UNKNOWN'


class WindowedNLI:
    def __init__(self, directory, *, device='cpu', batch_size=4, maximum=512, stride=64, threshold=0.8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(4)
        self.torch = torch
        self.device, self.batch_size = device, batch_size
        self.maximum, self.stride, self.threshold = maximum, stride, threshold
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.float32).to(device).eval()
        self.labels = label_map(self.model.config.id2label)
        if maximum > self.model.config.max_position_embeddings:
            raise ValueError('NLI window exceeds actual model capacity')

    def classify(self, premise, hypothesis):
        hyp_tokens = len(self.tokenizer.encode(hypothesis, add_special_tokens=False, truncation=False))
        capacity = self.maximum - hyp_tokens - self.tokenizer.num_special_tokens_to_add(pair=True)
        if not premise.strip() or not hypothesis.strip() or capacity <= 32:
            return {'verdict': 'UNKNOWN', 'reason': 'empty_input_or_hypothesis_too_long',
                    'windows': [], 'encoder_tokens': 0, 'hypothesis_truncated': False}
        encoded = self.tokenizer(premise, hypothesis, truncation='only_first', padding=True,
            max_length=self.maximum, stride=min(self.stride, capacity//2),
            return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors='pt')
        count = len(encoded['input_ids'])
        if count > 64:
            return {'verdict': 'UNKNOWN', 'reason': 'bounded_window_limit',
                    'windows': [], 'encoder_tokens': 0, 'hypothesis_truncated': False}
        windows, encoder_tokens = [], 0
        for start in range(0, count, self.batch_size):
            ids = {k: encoded[k][start:start+self.batch_size].to(self.device)
                   for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in encoded}
            with self.torch.inference_mode():
                probs = self.model(**ids).logits.float().softmax(-1).cpu().tolist()
            for offset, values in enumerate(probs):
                index = start + offset
                scores = {self.labels[i]: value for i, value in enumerate(values)}
                offsets = encoded['offset_mapping'][index].tolist()
                sequences = encoded.sequence_ids(index)
                source_offsets = [pair for pair, sequence in zip(offsets, sequences) if sequence == 0 and pair[1] > pair[0]]
                active = int(encoded['attention_mask'][index].sum())
                encoder_tokens += active
                windows.append({'window_index': index, 'premise_char_start': min(p[0] for p in source_offsets),
                    'premise_char_end': max(p[1] for p in source_offsets), 'encoder_tokens': active,
                    'probabilities': scores, 'argmax': max(scores, key=scores.get)})
            del ids
        return {'verdict': aggregate_windows(windows, self.threshold), 'reason': 'windowed_nli_proxy',
                'windows': windows, 'encoder_tokens': encoder_tokens, 'hypothesis_truncated': False,
                'id2label_observed': self.labels}

    def close(self):
        del self.model
        gc.collect()
        if self.device.startswith('cuda'):
            self.torch.cuda.empty_cache()
