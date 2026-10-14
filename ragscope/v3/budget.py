"""Durable reservations count failed/warmup/judge calls and protect final work."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time

from filelock import FileLock
from ragscope.indexing.identity import canonical_json, signature

STAGES = {'S0_S1_evidence_metrics': 800, 'S2_generation_protocol': 1000,
          'S3_retrieval_packing': 600, 'S4_task_workflows_router': 800,
          'S5_optional_lora': 500, 'S6_final_validation': 1800, 'technical_reserve': 500}


class Budget:
    def __init__(self, run):
        self.path = Path(run) / 'BUDGET.jsonl'
        self.lock = FileLock(str(self.path.with_suffix('.lock')), timeout=0)

    def _state(self):
        rows = []
        with self.path.open('rb') as handle:
            for line in handle:
                if not line.endswith(b'\n'):
                    raise ValueError('Partial budget tail; preserve and reconcile before another call')
                row = json.loads(line)
                if row.get('event') != 'INITIALIZED':
                    digest = row.pop('event_hash', None)
                    if signature(row) != digest:
                        raise ValueError('Budget event hash mismatch')
                    row['event_hash'] = digest
                rows.append(row)
        initial = rows[0]
        if (initial['event'] != 'INITIALIZED' or initial['request_limit'] != 6000 or
                initial['output_token_limit'] != 4000000 or initial['protected_final_requests'] != 1800 or
                initial['stage_request_limits'] != STAGES):
            raise ValueError('Budget differs from approved frozen caps')
        reserved, finalized = {}, {}
        for row in rows[1:]:
            if row['event'] == 'RESERVED':
                if row['request_id'] in reserved:
                    raise ValueError('Duplicate reservation')
                reserved[row['request_id']] = row
            elif row['event'] == 'FINALIZED':
                rid = row['request_id']
                if rid not in reserved or rid in finalized:
                    raise ValueError('Unreserved or duplicate finalization')
                finalized[rid] = row
            else:
                raise ValueError('Unknown budget event')
        stages = {s: sum(r['stage'] == s for r in reserved.values()) for s in STAGES}
        output = sum(finalized[rid]['charged_output_tokens'] if rid in finalized else r['max_output_tokens']
                     for rid, r in reserved.items())
        return {'requests_reserved': len(reserved), 'stage_requests': stages,
                'output_tokens_charged_or_reserved': output, 'reserved': reserved, 'finalized': finalized}

    def _append(self, event):
        event = dict(event, unix=time.time())
        event['event_hash'] = signature(event)
        with self.path.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(canonical_json(event) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def reserve(self, *, request_id, stage, max_output_tokens, purpose, model_identity):
        if stage not in STAGES or not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens < 1:
            raise ValueError('Invalid stage or token allowance')
        if not request_id or not purpose or not model_identity:
            raise ValueError('Reservation requires request/purpose/model identity')
        with self.lock:
            state = self._state()
            if any(row.get('output_limit_violation') for row in state['finalized'].values()):
                raise RuntimeError('Recorded backend output cap violation; no further model calls')
            if request_id in state['reserved']:
                raise RuntimeError('Already reserved; reconcile existing outcome, do not call model again')
            if state['stage_requests'][stage] >= STAGES[stage] or state['requests_reserved'] >= 6000:
                raise RuntimeError('Request budget exhausted')
            protected = max(0, 1800 - state['stage_requests']['S6_final_validation'])
            if stage != 'S6_final_validation' and 6000 - state['requests_reserved'] - 1 < protected:
                raise RuntimeError('Final S6 request reserve is protected')
            if state['output_tokens_charged_or_reserved'] + max_output_tokens > 4000000:
                raise RuntimeError('Output token budget exhausted')
            return self._append({'event': 'RESERVED', 'request_id': request_id, 'stage': stage,
                'max_output_tokens': max_output_tokens, 'purpose': purpose, 'model_identity': model_identity})

    def finalize(self, request_id, *, output_tokens, outcome):
        with self.lock:
            state = self._state()
            if request_id not in state['reserved'] or request_id in state['finalized']:
                raise RuntimeError('Missing reservation or already finalized')
            allowance = state['reserved'][request_id]['max_output_tokens']
            if output_tokens is not None and (not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0):
                raise ValueError('Invalid observed output token count')
            charged = allowance if output_tokens is None else output_tokens
            event = self._append({'event': 'FINALIZED', 'request_id': request_id,
                'observed_output_tokens': output_tokens, 'charged_output_tokens': charged,
                'outcome': outcome, 'output_limit_violation': charged > allowance})
            if charged > allowance:
                raise RuntimeError('Backend exceeded reserved output cap; recorded, stop new calls')
            return event

    def snapshot(self):
        with self.lock:
            state = self._state()
            return {k: v for k, v in state.items() if k not in ('reserved', 'finalized')}
