"""Source-stable SQuAD2 dev conversion for a separate short-text regression."""
from __future__ import annotations

from .qasper import digest


SOURCE = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def convert_squad(payload, token_counter):
    documents, queries, quarantined = [], [], []
    for article_index, article in enumerate(payload["data"]):
        title = article["title"]
        article_id = f"squad2:dev:article:{article_index}:{digest(title)[:12]}"
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            text = paragraph["context"]  # identity normalization preserves answer_start
            doc_id = f"{article_id}:p{paragraph_index}"
            unit_id = f"{doc_id}:u:{digest(text)[:12]}"
            unit = {"unit_id": unit_id, "doc_id": doc_id, "section_id": "paragraph",
                    "paragraph_index": paragraph_index, "text": text, "canonical_start": 0,
                    "canonical_end": len(text), "kind": "paragraph"}
            tokens = token_counter(text)
            documents.append({"doc_id": doc_id, "title": title, "text": text, "source": "SQuAD2.0",
                "source_revision": "official-dev-v2.0", "language": "en", "sections": [],
                "split": "test", "raw_checksum": digest(paragraph), "canonical_checksum": digest(text),
                "evidence_units": [unit], "metadata": {"source_url": SOURCE, "license": "CC-BY-SA-4.0",
                    "official_split": "dev", "evaluation_partition": "E4_frozen_100",
                    "article_cluster_id": article_id, "source_paragraph_index": paragraph_index,
                    "normalization": "identity_unicode_v1", "reference_tokenizer": "cl100k_base",
                    "reference_n_tokens": tokens, "length_group": "short" if tokens <= 512 else "medium" if tokens <= 4096 else "long"}})
            for raw in paragraph["qas"]:
                annotations, bad = [], []
                for answer in raw.get("answers", []):
                    start, answer_text = answer["answer_start"], answer["text"]
                    if text[start:start+len(answer_text)] != answer_text:
                        bad.append("answer_offset_mismatch")
                    annotations.append({"text": answer_text, "raw_start": start, "canonical_start": start,
                                        "canonical_end": start+len(answer_text)})
                answerable = not raw["is_impossible"]
                if answerable and not annotations:
                    bad.append("missing_answer")
                query = {"query_id": "squad2:" + raw["id"], "question": raw["question"], "split": "test",
                         "scope": {"kind": "global", "corpus": "official_squad2_dev_all_paragraphs"},
                         "answerability": "answerable" if answerable else "unanswerable",
                         "answer_refs": list(dict.fromkeys(a["text"] for a in annotations)),
                         "evidence_reference_sets": [[unit_id]] if answerable else [], "annotation_disagreements": [],
                         "metadata": {"gold_doc_id_evaluation_only": doc_id, "article_cluster_id": article_id,
                                      "source_url": SOURCE, "official_split": "dev", "raw_question_id": raw["id"],
                                      "reference_n_tokens": tokens, "answer_annotations": annotations,
                                      "quarantine_reasons": sorted(set(bad)),
                                      "unanswerable_scope": "original_paragraph_only"}}
                queries.append(query)
                if bad:
                    quarantined.append(query)
    return documents, queries, quarantined


def select_squad_short(queries, n=100, seed=42):
    if n <= 0:
        raise ValueError("n must be positive")
    eligible = [q for q in queries if q["answerability"] == "answerable"
                and q["metadata"]["reference_n_tokens"] <= 512 and not q["metadata"]["quarantine_reasons"]]
    return sorted(eligible, key=lambda q: digest(f"{seed}:E4:{q['query_id']}"))[:n]
