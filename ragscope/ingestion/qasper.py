"""QASPER v0.3 conversion with source-stable evidence and explicit quarantine.

No imports here download data or allocate accelerators. Coordinates always refer
to ``Document.text`` (canonical Unicode text), never to the original JSON bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any


NORMALIZATION_VERSION = "qasper_nfc_structure_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    text = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Retain paragraph/code line structure; normalize only Unicode/newlines."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def match_key(text: str) -> str:
    return " ".join(normalize_text(text).split())


def make_document(paper_id: str, paper: dict, split: str, revision: str = "v0.3") -> dict:
    """Build stable units BEFORE any chunking, preserving every source paragraph."""
    doc_id = f"qasper:{paper_id}"
    units, parts, sections = [], [], []
    position = 0

    def add(text: str, section: str, paragraph: int, kind: str) -> None:
        nonlocal position
        text = normalize_text(text)
        if not text:
            return
        if parts:
            position += 2
        unit_id = f"{doc_id}:{section}:p{paragraph}:{digest(text)[:12]}"
        units.append({"unit_id": unit_id, "doc_id": doc_id, "section_id": section,
                      "paragraph_index": paragraph, "text": text,
                      "canonical_start": position, "canonical_end": position + len(text),
                      "kind": kind, "coordinate_system": "canonical_document_char"})
        parts.append(text)
        position += len(text)

    add(paper.get("title", ""), "title", 0, "title")
    add(paper.get("abstract", ""), "abstract", 0, "paragraph")
    for section_index, section in enumerate(paper.get("full_text", [])):
        sid = f"s{section_index}"
        title = section.get("section_name") or ""
        add(title, sid, -1, "heading")
        sections.append({"section_id": sid, "title": title, "source_section_index": section_index})
        for paragraph_index, paragraph in enumerate(section.get("paragraphs", [])):
            add(paragraph, sid, paragraph_index, "paragraph")
    text = "\n\n".join(parts)
    return {"doc_id": doc_id, "text": text, "source": "allenai/qasper", "source_revision": revision,
            "language": "en", "title": paper.get("title", ""), "sections": sections,
            "raw_checksum": digest(paper), "canonical_checksum": digest(text), "split": split,
            "evidence_units": units, "metadata": {"paper_id": paper_id, "official_split": split,
            "normalization_version": NORMALIZATION_VERSION, "license": "CC-BY-4.0",
            "source_url": "https://huggingface.co/datasets/allenai/qasper",
            "coordinate_system": "canonical_document_char", "raw_to_canonical_mapping":
            "source section_index/paragraph_index retained; byte offsets are not claimed"}}


def convert_question(raw: dict, document: dict) -> tuple[dict, list[str]]:
    """Return the query plus reasons excluding it from the text-only core.

    A mixed annotation query remains available in the audit and does not silently
    lose its inconvenient annotation. Duplicate paragraph matches are quarantined
    because the source annotation supplies text but no unambiguous location.
    """
    by_text = defaultdict(list)
    for unit in document["evidence_units"]:
        if unit["kind"] == "paragraph":
            by_text[match_key(unit["text"])].append(unit["unit_id"])
    reasons, references, evidence_sets, annotations, unanswerables = [], [], [], [], []
    for annotation in raw.get("answers", []):
        answer = annotation.get("answer", {})
        unanswerable = answer.get("unanswerable")
        if not isinstance(unanswerable, bool):
            reasons.append("malformed_answerability")
            continue
        unanswerables.append(unanswerable)
        annotation_units, unmatched, ambiguous, floats = [], [], [], []
        if unanswerable:
            answer_text, answer_type = "", "unanswerable"
        elif answer.get("extractive_spans"):
            answer_text = ", ".join(answer["extractive_spans"])
            answer_type = "extractive"
        elif answer.get("free_form_answer"):
            answer_text, answer_type = answer["free_form_answer"], "free_form"
        elif isinstance(answer.get("yes_no"), bool):
            answer_text, answer_type = ("Yes" if answer["yes_no"] else "No"), "yes_no"
        else:
            answer_text, answer_type = "", "malformed"
            reasons.append("missing_answer")
        for text in answer.get("evidence", []):
            if "FLOAT SELECTED" in text:
                floats.append(text)
                reasons.append("float_evidence")
                continue
            matches = by_text.get(match_key(text), [])
            if not matches:
                unmatched.append(text)
                reasons.append("unmatched_evidence")
            elif len(matches) != 1:
                ambiguous.append({"text": text, "candidate_units": matches})
                reasons.append("ambiguous_evidence_location")
            else:
                annotation_units.append(matches[0])
        if not unanswerable and not annotation_units:
            reasons.append("missing_text_evidence")
        if not unanswerable and answer_text:
            references.append(normalize_text(answer_text))
        if annotation_units and not (unmatched or ambiguous or floats):
            evidence_sets.append(list(dict.fromkeys(annotation_units)))
        annotations.append({"annotation_id": annotation.get("annotation_id"),
                            "answer_type": answer_type, "answer": answer_text,
                            "unanswerable": unanswerable, "unit_ids": annotation_units,
                            "unmatched_evidence": unmatched, "ambiguous_evidence": ambiguous,
                            "float_evidence": floats, "raw_answer": answer})
    if not unanswerables:
        answerability = "unknown"
        reasons.append("missing_annotations")
    elif len(set(unanswerables)) > 1:
        answerability = "conflict"
        reasons.append("answerability_conflict")
    else:
        answerability = "unanswerable" if unanswerables[0] else "answerable"
    question = normalize_text(raw.get("question", ""))
    if not question or not raw.get("question_id"):
        reasons.append("malformed_question")
    disagreements = []
    if len(set(references)) > 1:
        disagreements.append("multiple_answer_texts")
    if len({tuple(sorted(x)) for x in evidence_sets}) > 1:
        disagreements.append("multiple_evidence_sets")
    if answerability == "conflict":
        disagreements.append("answerability_conflict")
    query = {"query_id": f"qasper:{raw.get('question_id', digest(raw)[:16])}",
             "question": question, "split": document["split"],
             "scope": {"kind": "document", "doc_ids": [document["doc_id"]]},
             "doc_id": document["doc_id"], "answerability": answerability,
             "answer_refs": list(dict.fromkeys(references)),
             "evidence_reference_sets": list({tuple(x): x for x in evidence_sets}.values()),
             "annotation_disagreements": disagreements,
             "metadata": {"source": "allenai/qasper", "source_revision": document["source_revision"],
                          "annotations": annotations, "evidence_aggregation": "best_legal_reference_set",
                          "unanswerable_scope": "specified_paper_only", "quarantine_reasons": sorted(set(reasons))}}
    return query, sorted(set(reasons))


def duplicate_groups(documents: list[dict], threshold: float = 0.9) -> list[dict]:
    """Exact duplicates and conservative bottom-128 trigram near-duplicate audit.

    Candidate filtering avoids an all-pairs full-document quadratic comparison.
    Only candidates sharing >=90% of their bottom sketches receive exact Jaccard
    verification. This is a declared approximate near-duplicate screen, not a proof
    of absence of all semantic paraphrases.
    """
    parents, by_id = {}, {}
    reasons = []

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(left, right, reason, similarity=1.0):
        parents[find(right)] = find(left)
        reasons.append({"left": left, "right": right, "reason": reason, "similarity": similarity})

    exact, titles, sketches, inverted, bodies = {}, {}, {}, defaultdict(list), {}
    for document in documents:
        key = document["doc_id"]
        parents[key] = key
        by_id[key] = {"split": document["split"]}
        body = " ".join(u["text"] for u in document["evidence_units"] if u["kind"] == "paragraph")
        body = match_key(body).lower()
        bodies[key] = body
        checksum = digest(body)
        title = re.sub(r"\W+", "", document["title"].lower())
        if checksum in exact:
            join(exact[checksum], key, "exact_body")
        else:
            exact[checksum] = key
        if title and title in titles:
            join(titles[title], key, "same_normalized_title_conservative")
        elif title:
            titles[title] = key
        words = body.split()
        hashes = {hashlib.blake2b(" ".join(words[i:i+3]).encode(), digest_size=8).digest()
                  for i in range(max(0, len(words)-2))}
        sketch = set(sorted(hashes)[:128])
        candidates = Counter(other for h in sketch for other in inverted[h])
        for other, shared in candidates.items():
            if shared / max(1, min(len(sketch), len(sketches[other]))) >= threshold:
                other_words = bodies[other].split()
                other_hashes = {hashlib.blake2b(" ".join(other_words[i:i+3]).encode(), digest_size=8).digest()
                                for i in range(max(0, len(other_words)-2))}
                similarity = len(hashes & other_hashes) / max(1, len(hashes | other_hashes))
                if similarity >= threshold:
                    join(other, key, "trigram_jaccard", similarity)
        sketches[key] = sketch
        for h in sketch:
            inverted[h].append(key)
    groups = defaultdict(list)
    for key in parents:
        groups[find(key)].append(key)
    return [{"group_id": digest(sorted(ids))[:16], "doc_ids": sorted(ids),
             "splits": sorted({by_id[x]["split"] for x in ids}),
             "matches": [r for r in reasons if r["left"] in ids and r["right"] in ids]}
            for ids in groups.values() if len(ids) > 1]


def freeze_splits(documents: list[dict], queries: list[dict], groups: list[dict],
                  targets: dict[str, int], max_docs: int = 600, seed: int = 42) -> dict:
    """Document-level seeded selection; cross-split duplicate groups quarantined."""
    if max_docs <= 0 or any(n < 0 for n in targets.values()):
        raise ValueError("invalid selection budget")
    blocked = {doc_id for group in groups if len(group["splits"]) > 1 for doc_id in group["doc_ids"]}
    aliases = {doc_id: min(group["doc_ids"]) for group in groups if len(group["splits"]) == 1
               for doc_id in group["doc_ids"] if doc_id != min(group["doc_ids"])}
    # Keep duplicate source documents available in raw snapshots. Exclude their
    # queries rather than remapping evidence across structurally distinct copies.
    excluded = blocked | set(aliases)
    by_doc = defaultdict(list)
    for query in queries:
        if query["doc_id"] not in excluded and not query["metadata"]["quarantine_reasons"]:
            by_doc[query["doc_id"]].append(query)
    selected_docs, selected_queries = [], []
    counts = Counter()
    # Evaluation quotas take priority under the explicit 600-document cap.
    for split in ("dev", "test", "train"):
        candidates = sorted((d for d in documents if d["split"] == split and d["doc_id"] in by_doc),
                            key=lambda d: digest(f"{seed}:doc:{d['doc_id']}"))
        for document in candidates:
            if counts[split] >= targets.get(split, 0) or len(selected_docs) >= max_docs:
                break
            selected_docs.append(document)
            remaining = targets[split] - counts[split]
            chosen = sorted(by_doc[document["doc_id"]], key=lambda q: digest(f"{seed}:query:{q['query_id']}"))[:remaining]
            selected_queries.extend(chosen)
            counts[split] += len(chosen)
    assert len({q["query_id"] for q in selected_queries}) == len(selected_queries)
    global_queries = []
    for query in sorted((q for q in selected_queries if q["split"] == "test" and q["answerability"] == "answerable"),
                        key=lambda q: digest(f"{seed}:global:{q['query_id']}"))[:100]:
        clone = json.loads(canonical_json(query))
        clone["scope"] = {"kind": "global", "corpus": "frozen_qasper"}
        clone["metadata"]["gold_doc_id_evaluation_only"] = clone.pop("doc_id")
        global_queries.append(clone)
    return {"documents": selected_docs, "queries": selected_queries, "global_queries": global_queries,
            "counts": dict(counts), "cross_split_quarantined_doc_ids": sorted(blocked),
            "aliases": aliases, "selection_rule": "seeded_sha256_document_then_query; dev,test,train quota priority",
            "seed": seed, "targets": targets, "max_docs": max_docs}
