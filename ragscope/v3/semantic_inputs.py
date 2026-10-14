"""Evidence-preserving blinded inputs and unambiguous calibration contrasts."""
from __future__ import annotations
import copy


def validated_calibration_groups(groups):
    """Version the construction fixes before any judge has scored these cases.

    Buying and selling can both occur, so a purchase alone cannot contradict a
    later sale. Explicitly negate the purchase instead. Likewise a sample count
    is stated as exact to prevent interpreting an alternate count as a subset.
    """
    corrected = copy.deepcopy(groups)
    for group in corrected:
        category = group['category']
        if category == 'paraphrase':
            entity = group['premise'].split(' purchased', 1)[0]
            group['negative'] = f'{entity} did not purchase that car.'
        elif category == 'paper_and_citation':
            group['premise'] = group['premise'].replace(' used ', ' used exactly ')
        group.update(construction_version='calibration-v2-before-judge',
                     construction_review='RULE_AUDIT; NOT_HUMAN_REVIEWED',
                     judge_result='NOT_RUN_YET')
    return corrected


def blinded_context_pieces(bundle):
    """Keep source text/coordinates while hiding generator and retrieval names."""
    fields = ('chunk_id', 'doc_id', 'section_id', 'text', 'canonical_start',
              'canonical_end', 'unit_spans', 'source_unit_ids')
    return [{key: piece[key] for key in fields if key in piece} for piece in bundle['pieces']]
