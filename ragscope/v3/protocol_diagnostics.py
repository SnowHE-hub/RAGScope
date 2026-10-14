"""Conservative JSON-envelope diagnosis without inventing missing model fields."""
import json
from ragscope.generation.pipeline import RAGPipeline


def recover_envelope(raw):
    decoder, candidates = json.JSONDecoder(), []
    # Only syntactically complete original JSON objects are eligible. No repair
    # inserts fields, quotes, boolean values, citations, or answer characters.
    for start, char in enumerate(raw):
        if char != '{':
            continue
        try:
            value, consumed = decoder.raw_decode(raw[start:])
            if not isinstance(value, dict):
                continue
            parsed = RAGPipeline.parse_output(raw[start:start+consumed])
            candidates.append({'original_start': start, 'original_end': start+consumed,
                               'parsed': parsed, 'original_object': value})
        except (ValueError, TypeError):
            continue
    if len(candidates) == 1:
        return {'status': 'RECOVERABLE_ENVELOPE_DIAGNOSTIC_ONLY', **candidates[0]}
    return {'status': 'AMBIGUOUS_MULTIPLE_OBJECTS' if candidates else 'NOT_RECOVERABLE_WITHOUT_CONTENT_CHANGE',
            'eligible_objects': len(candidates)}
