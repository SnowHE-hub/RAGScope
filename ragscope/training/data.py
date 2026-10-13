"""Train-only, full-evidence examples with assistant-only supervision."""
from __future__ import annotations
import json

from ragscope.evaluation.generation_metrics import answer_reference_texts
from ragscope.evaluation.evidence import evidence_metrics
from ragscope.generation.context import pack_context, render_context
from ragscope.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from ragscope.indexing.identity import signature


def encode_assistant_only(tokenizer, messages, max_seq_len=1024):
    """Use actual chat-template tokens; reject, never truncate answer/evidence.

    Requiring an exact token prefix avoids guessed assistant boundaries for
    tokenizers whose chat template changes with conversation termination.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Exactly one final assistant response is required")
    if any(message.get("role") == "assistant" for message in messages[:-1]):
        raise ValueError("Only the final assistant turn may be supervised")
    options = {"tokenize": True, "enable_thinking": False, "return_dict": False}
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    if not isinstance(prompt, list) or not isinstance(full, list) or any(not isinstance(i, int) for i in full):
        raise ValueError("Chat template must return a flat token ID list")
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat-template assistant boundary is not an exact token prefix")
    if len(full) <= len(prompt):
        raise ValueError("No assistant supervision tokens")
    if len(full) > max_seq_len:
        raise ValueError(f"over_budget_full_example:{len(full)}>{max_seq_len}; no truncation")
    return {"input_ids": full, "attention_mask": [1]*len(full),
            "labels": [-100]*len(prompt)+full[len(prompt):],
            "prompt_tokens": len(prompt), "supervised_tokens": len(full)-len(prompt),
            "total_tokens": len(full), "messages_hash": signature(messages)}


def prepare_examples(queries, documents, tokenizer, train_doc_ids, *, max_seq_len=1024, max_examples=1000, split="train"):
    if not 1 <= max_examples <= 1000 or max_seq_len not in {512, 1024}:
        raise ValueError("Training preparation must respect frozen 1000-example /512-or-1024-token limits")
    if split not in {"train", "dev"} or (split == "dev" and max_examples > 60):
        raise ValueError("Preparation accepts train or at most 60 dev loss examples, never test")
    docs = {doc["doc_id"]: doc for doc in documents}
    accepted, rejected = [], []
    allowed = set(train_doc_ids)
    selected = sorted(queries, key=lambda q: q["query_id"])[:max_examples]
    for query in selected:
        qid = query["query_id"]
        try:
            doc_id = query.get("doc_id") or query.get("scope", {}).get("doc_id")
            if query.get("split") != split or doc_id not in allowed:
                raise ValueError("split_leakage_or_not_train")
            document = docs[doc_id]
            if document.get("split") not in {None, "", split}:
                raise ValueError("document_not_train")
            units = {u["unit_id"]: u for u in document["evidence_units"]}
            answerability = query.get("answerability")
            if answerability not in {"answerable", "unanswerable"}:
                raise ValueError("ambiguous_answerability")
            unanswerable = answerability == "unanswerable"
            annotation_id = None
            if unanswerable:
                # Real annotated negative with an actual paragraph, no fabricated evidence.
                selected_units = [uid for uid, unit in units.items() if unit.get("kind", "paragraph") == "paragraph"][:1]
                answer = "I cannot answer based on the provided context."
            else:
                references = query.get("evidence_reference_sets", [])
                if not references or not references[0]:
                    raise ValueError("missing_reference_evidence")
                selected_units = references[0]
                answers = answer_reference_texts(query.get("answer_refs", []))
                if not answers:
                    raise ValueError("missing_reference_answer")
                answer = answers[0]
                annotations = query.get("metadata", {}).get("annotations", [])
                if annotations:
                    paired = [annotation for annotation in annotations
                              if not annotation.get("unanswerable") and annotation.get("answer")
                              and set(annotation.get("unit_ids", [])) == set(selected_units)
                              and not any(annotation.get(key) for key in
                                          ("float_evidence", "unmatched_evidence", "ambiguous_evidence"))]
                    if not paired:
                        raise ValueError("complete_answer_evidence_annotation_pair_unavailable")
                    answer = answer_reference_texts([paired[0]["answer"]])[0]
                    annotation_id = paired[0].get("annotation_id")
            pieces = []
            for uid in dict.fromkeys(selected_units):
                unit = units[uid]
                a, b = unit["canonical_start"], unit["canonical_end"]
                if document["text"][a:b] != unit["text"] or unit["doc_id"] != doc_id:
                    raise ValueError("evidence_text_coordinate_mismatch")
                pieces.append({"chunk_id": "train:"+uid, "doc_id": doc_id, "text": unit["text"],
                               "canonical_start": a, "canonical_end": b, "title": document.get("title", ""),
                               "unit_spans": [{"unit_id": uid, "doc_id": doc_id, "start": a, "end": b}]})
            if not pieces:
                raise ValueError("no_real_context_units")
            bundle = pack_context(pieces, max_tokens=10**9, token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                                  reference_tokenizer="training_actual_tokenizer", scope=doc_id)
            if (not unanswerable and evidence_metrics([selected_units],bundle.unit_spans,'answerable',units)
                    ['full_reference_coverage'] != 1.0):
                raise ValueError('packing_removed_required_evidence; no partial supervision example')
            output = {"answer": answer, "citations": [] if unanswerable else [piece["chunk_id"] for piece in bundle.pieces],
                      "abstain": unanswerable}
            messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": RAG_USER_TEMPLATE.format(question=query["question"], context=render_context(bundle))},
                        {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}]
            encoded = encode_assistant_only(tokenizer, messages, max_seq_len)
            accepted.append({"query_id": qid, "doc_id": doc_id, "split": split, **encoded,
                             "context_hash": bundle.context_hash, "messages": messages,
                             "annotation_id": annotation_id, "answerability": answerability,
                             "evidence_policy": "first_real_paragraph" if unanswerable else "first_legal_reference_set_with_paired_annotation"})
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"query_id": qid, "reason": str(exc), "error_type": type(exc).__name__})
    return accepted, rejected
