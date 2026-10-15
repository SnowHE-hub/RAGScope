"""Transparent lexical fact checks. UNKNOWN is retained, never a correctness vote."""
import re


def numbers(text):
    return set(re.findall(r'(?<!\w)\d+(?:[.,]\d+)*%?(?!\w)',text))


def polarity(text):
    match=re.match(r'^\s*(yes|no)\b',text,re.I)
    return match.group(1).lower() if match else None


def fact_checks(question, answer, references, pieces, citations):
    cited='\n'.join(p['text'] for p in pieces if p['chunk_id'] in set(citations))
    context='\n'.join(p['text'] for p in pieces)
    values=numbers(answer);cited_numbers=numbers(cited)
    entities=set(re.findall(r'\b[A-Z][A-Z0-9_-]{1,}\b',answer))
    missing_entities=sorted(e for e in entities if e.casefold() not in cited.casefold())
    ref_polarities={p for r in references if (p:=polarity(r)) and len(r.split())<8}
    answer_polarity=polarity(answer)
    conflict=(answer_polarity not in ref_polarities) if answer_polarity and len(ref_polarities)==1 else None
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',answer) if len(s.split())>=4]
    literal=[s.casefold() in cited.casefold() for s in sentences]
    comparison=bool(re.search(r'\b(better|worse|higher|lower|outperform\w*|improv\w*|surpass\w*)\b',answer,re.I))
    verdict=('REFERENCE_POLARITY_CONFLICT' if conflict else
             'LITERAL_SENTENCES_SUPPORTED' if sentences and all(literal) else
             'REFERENCE_POLARITY_MATCH' if conflict is False else
             'LITERAL_DETAILS_MATCH' if (values or entities) and values<=cited_numbers and not missing_entities else 'UNKNOWN')
    return {'rule_verdict':verdict,'number_check':'LITERAL_MATCH' if values and values<=cited_numbers else ('NOT_FOUND_IN_CITED_EVIDENCE' if values else 'UNKNOWN'),
        'numbers_not_in_citations':sorted(values-cited_numbers),'numbers_not_in_context':sorted(values-numbers(context)),
        'entity_check':'LITERAL_MATCH' if entities and not missing_entities else ('NOT_FOUND_IN_CITED_EVIDENCE' if entities else 'UNKNOWN'),
        'entities_not_in_citations':missing_entities,
        'yes_no_reference_conflict':conflict,'negation_check':'REFERENCE_POLARITY_CONFLICT' if conflict else ('REFERENCE_POLARITY_MATCH' if conflict is False else 'UNKNOWN'),
        'comparison_check':'LITERAL_SENTENCE_MATCH' if comparison and sentences and all(literal) else 'UNKNOWN',
        'literal_supported_sentences':sum(literal),'sentence_n':len(sentences),
        'factual_correctness':'UNKNOWN','interpretation':'Literal checks flag omissions or mismatches; they are not entailment or human factual accuracy.'}
