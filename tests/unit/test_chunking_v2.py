"""Offline fixtures exercise coverage, identity and budget rather than snapshots."""
import json
import random

import pytest

from ragscope.schema import (Chunk, ContextBundle, Document, EvidenceUnit,
                             GenerationRecord, Query, signature)
from ragscope.ingestion.cleaner import (clean_documents, clean_documents_legacy,
                                       clean_text, dedup_documents, evidence_units, remap_query_aliases)
from ragscope.ingestion.chunkers import (FixedSizeChunker, ParentChildChunker,
    SemanticChunker, SentenceAwareChunker, StructureRecursiveChunker, TokenizerAdapter)


# 26 hand-authored engineering fixtures, not a Chinese QA benchmark.
FIXTURES = [
    "中", "无空格的中文文本需要严格切开但不能破坏Unicode字符。" * 5,
    "第一句。第二句！第三句？重复。重复。", "你好，世界！\n\n下一段。",
    "🙂🚀🧑‍💻" * 12, "e\u0301éÅ中文𠮷" * 13, "x" * 301,
    "a " * 130, "x", "", "   \n\t", "a\n\n" * 30,
    "  leading\tspaces\ntrailing  ", "a" * 31, "a" * 32, "a" * 33,
    "# Heading\n\nParagraph.\n\n- first\n- second\n\n## Next\nBody.",
    "```python\nif x < 3:\n    print('<T>')\n```\n\nText after code.",
    "~~~sql\nSELECT *\nFROM table_name;\n~~~\n", "列表：\n1. 项目甲\n2. 项目乙",
    "A sentence. Another sentence that exceeds the budget significantly. Last one.",
    "repeat\n\nrepeat\n\nrepeat", "a\r\nb\rc", "<T> x < y && y > z",
    "Long URL https://example.test/" + "a/b/" * 60, "\n\nHEAD\n" + " " * 70 + "TAIL\n",
]


def verify_chunks(text, chunker, budget):
    doc = {"doc_id": "fixture", "text": text, "source_revision": "fixture-v1"}
    chunks = chunker.chunk(doc)
    if not text.strip():
        assert chunks == []
        return
    assert chunks[0].canonical_start == 0
    assert chunks[-1].canonical_end == len(text)
    covered = [False] * len(text)
    for i, chunk in enumerate(chunks):
        assert chunk.text == text[chunk.canonical_start:chunk.canonical_end]
        assert 0 < chunk.n_tokens == chunker.count_tokens(chunk.text) <= budget
        assert "\ufffd" not in chunk.text
        assert chunk.chunking_signature == chunker.chunking_signature
        assert chunk.prev_chunk_id == (chunks[i - 1].chunk_id if i else None)
        assert chunk.next_chunk_id == (chunks[i + 1].chunk_id if i + 1 < len(chunks) else None)
        if i:
            assert chunk.canonical_start > chunks[i - 1].canonical_start
            assert chunk.canonical_start <= chunks[i - 1].canonical_end
            assert chunk.canonical_end > chunks[i - 1].canonical_end
        for span in chunk.unit_spans:
            assert text[span["start"]:span["end"]] == chunk.text[span["chunk_start"]:span["chunk_end"]]
        covered[chunk.canonical_start:chunk.canonical_end] = [True] * len(chunk.text)
    assert all(covered)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


@pytest.mark.parametrize("text", FIXTURES)
@pytest.mark.parametrize("method", ["fixed", "sentence", "structure", "parent_child"])
def test_fixture_coverage_and_hard_budget(text, method):
    chunker = {"fixed": lambda: FixedSizeChunker(32, 7),
               "sentence": lambda: SentenceAwareChunker(32),
               "structure": lambda: StructureRecursiveChunker(32),
               "parent_child": lambda: ParentChildChunker(32, 7, 96)}[method]()
    verify_chunks(text, chunker, 32)


def test_deterministic_random_unicode_property():
    rng = random.Random(42)
    alphabet = list("中a🙂。 !?\n\t<>[]") + ["\u0301", "𠮷"]
    for _ in range(60):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 240)))
        budget = rng.randint(4, 40)
        verify_chunks(text, FixedSizeChunker(budget, rng.randrange(budget)), budget)
        verify_chunks(text, SentenceAwareChunker(budget), budget)
        verify_chunks(text, StructureRecursiveChunker(budget), budget)


def test_no_forced_short_split_and_real_overlap_coordinates():
    text = "一二三四五六七八九十甲乙丙丁戊己庚辛壬癸"
    chunker = FixedSizeChunker(8, 2)
    chunks = chunker.chunk({"doc_id": "d", "text": text})
    assert [(c.canonical_start, c.canonical_end) for c in chunks] == [(0, 8), (6, 14), (12, 20)]
    assert all(c.metadata["overlap_previous_tokens"] == 2 for c in chunks[1:])
    assert len(FixedSizeChunker(32, 7).chunk({"doc_id": "d", "text": "short"})) == 1


def test_stable_evidence_and_configuration_specific_chunks():
    doc = {"doc_id": "d", "text": "same\n\nsame\n\nlast"}
    first, second = FixedSizeChunker(6, 0).chunk(doc), FixedSizeChunker(8, 1).chunk(doc)
    assert {s["unit_id"] for c in first for s in c.unit_spans} == {s["unit_id"] for c in second for s in c.unit_spans}
    assert {c.chunk_id for c in first}.isdisjoint(c.chunk_id for c in second)
    assert len({u["unit_id"] for u in evidence_units(doc)}) == 3
    assert first == FixedSizeChunker(6, 0).chunk(doc)
    assert first[0].metadata is not first[-1].metadata


def test_tokenizer_adapter_counts_actual_nonadditive_encoding():
    class MergeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return [1] * len(text.replace("aa", "a"))
        def decode(self, tokens, **kwargs):
            return "a" * len(tokens)
    tokenizer = TokenizerAdapter(MergeTokenizer(), "fixture-merges", "r1")
    chunker = FixedSizeChunker(7, 2, tokenizer=tokenizer)
    verify_chunks("aa" * 90 + "🙂" * 20, chunker, 7)
    assert tokenizer.decode([1]) == "a"
    assert TokenizerAdapter().decode(TokenizerAdapter().encode("中🙂")) == "中🙂"
    assert chunker.chunking_signature != FixedSizeChunker(7, 2).chunking_signature


def test_explicit_unrepresentable_budget():
    class Bytes:
        def encode(self, text):
            return list(text.encode("utf8"))
        def decode(self, tokens):
            return bytes(tokens).decode("utf8")
    tokenizer = TokenizerAdapter(Bytes(), "bytes", "v1")
    assert tokenizer.decode(tokenizer.encode("中")) == "中"
    verify_chunks("中🙂英文abc" * 15, FixedSizeChunker(8, 2, tokenizer=tokenizer), 8)
    with pytest.raises(ValueError, match="Unicode"):
        FixedSizeChunker(1, 0, tokenizer=tokenizer).chunk({"doc_id": "d", "text": "中"})


def test_sentence_overlap_reduction_and_long_sentence():
    chunker = SentenceAwareChunker(12)
    verify_chunks("12345678. 12345678. " + "长" * 40, chunker, 12)
    assert any(event["reason"] == "sentence_overlap_reduced" for event in chunker.last_events)
    assert any(event["reason"] == "oversized_sentence_fallback" for event in chunker.last_events)
    assert SemanticChunker is StructureRecursiveChunker


def test_parent_relationship_and_boundaries():
    doc = {"doc_id": "d", "text": ("# H\n\n" + "text " * 25 + "\n\n") * 4}
    chunker = ParentChildChunker(32, 4, 96)
    children, parents = chunker.chunk_with_parents(doc)
    assert len(parents) > 1
    for child in children:
        parent = chunker.parents[child.parent_id]
        assert parent.doc_id == child.doc_id
        assert parent.canonical_start <= child.canonical_start < child.canonical_end <= parent.canonical_end
        assert parent.n_tokens <= 96


@pytest.mark.parametrize("factory", [lambda: FixedSizeChunker(0, 0), lambda: FixedSizeChunker(8, 8),
    lambda: FixedSizeChunker(8, -1), lambda: FixedSizeChunker(True, 0),
    lambda: FixedSizeChunker(8, 1.5), lambda: SentenceAwareChunker(0),
    lambda: SentenceAwareChunker(8, overlap_sentences=2), lambda: ParentChildChunker(16, 0, 8)])
def test_invalid_parameters(factory):
    with pytest.raises(ValueError):
        factory()


def test_cleaning_preserves_long_documents_code_and_structure():
    raw = "# 标题\r\n\r\n```cpp\r\nif (x < y) {\r\n    T<int> a;\r\n}\r\n```\r\n\r\n" + "内容" * 3000
    cleaned = clean_documents([{"doc_id": "long", "text": raw, "content_type": "markdown"}])[0]
    assert len(cleaned["text"]) > 5000
    assert "\n\n" in cleaned["text"] and "    T<int> a;" in cleaned["text"]
    assert cleaned["raw_checksum"] != cleaned["canonical_checksum"]
    assert clean_documents_legacy([{"doc_id": "long", "text": raw}]) == []
    assert clean_text("<T> x < y", "code") == "<T> x < y"


def test_html_is_explicit_and_retains_pre_content():
    raw = "<h1>Heading</h1><p>A &amp; B</p><pre>if x &lt; 2:\n    go()</pre><script>bad()</script>"
    text = clean_text(raw, "html")
    assert "Heading\n\n" in text and "A & B" in text
    assert "if x < 2:\n    go()" in text and "bad()" not in text
    assert clean_text(raw, "text") == raw


def test_cleaning_rejections_offsets_and_invalid_input():
    log = []
    assert clean_documents([{"doc_id": "a", "text": "x" * 100}], max_chars=10, rejection_log=log) == []
    assert log[0]["reason"] == "above_max_chars"
    assert clean_documents([{"doc_id": "a", "text": ""}]) == []
    with pytest.raises(ValueError):
        clean_documents([], min_chars=10, max_chars=2)
    with pytest.raises(TypeError):
        clean_text(12)
    with pytest.raises(ValueError):
        clean_text("ok", "unknown")
    doc = {"doc_id": "d", "text": "abc\r\ndef"}
    doc["evidence_units"] = evidence_units(doc)
    with pytest.raises(ValueError, match="Clean before"):
        clean_documents([doc])
    doc["evidence_units"][0]["text"] = "bad"
    with pytest.raises(ValueError, match="canonical"):
        evidence_units(doc)


def test_dedup_aliases_and_cross_split_flag():
    aliases = {}
    docs = [{"doc_id": "a", "text": "same", "split": "train"},
            {"doc_id": "b", "text": "same", "split": "train"},
            {"doc_id": "c", "text": "same", "split": "test"}]
    result = dedup_documents(docs, alias_map=aliases)
    assert [doc["doc_id"] for doc in result] == ["a", "c"]
    assert aliases == {"a": "a", "b": "a", "c": "c"}
    assert result[0]["metadata"]["aliases"] == ["b"]
    assert all(doc["metadata"]["cross_split_duplicate"] for doc in result)
    assert "metadata" not in docs[0]


def test_dedup_preserves_query_and_evidence_gold_mapping():
    docs = [{"doc_id": "a", "text": "same\n\nsecond"}, {"doc_id": "b", "text": "same\n\nsecond"}]
    alias_units = evidence_units(docs[1])
    query = {"query_id": "q", "gold_doc_id": "b", "scope": {"doc_id": "b", "doc_ids": ["a", "b"]},
             "evidence_reference_sets": [[unit["unit_id"] for unit in alias_units]]}
    aliases = {}
    deduped = dedup_documents(docs, alias_map=aliases)
    rewritten = remap_query_aliases([query], deduped, aliases)[0]
    assert rewritten["gold_doc_id"] == "a"
    assert rewritten["scope"] == {"doc_id": "a", "doc_ids": ["a"]}
    assert rewritten["evidence_reference_sets"] == [[unit["unit_id"] for unit in evidence_units(docs[0])]]
    assert query["gold_doc_id"] == "b"
    deduped[0]["metadata"]["alias_evidence_units"]["b"][0]["canonical_end"] = 1
    with pytest.raises(ValueError, match="incompatible"):
        remap_query_aliases([query], deduped, aliases)


def test_supplied_evidence_objects_duplicate_ids_and_legacy_short_filter():
    unit = EvidenceUnit("u", "d", "s", 0, "abc", 0, 3)
    assert evidence_units({"doc_id": "d", "text": "abc", "evidence_units": [unit]})[0] == unit.to_dict()
    with pytest.raises(ValueError, match="Duplicate"):
        evidence_units({"doc_id": "d", "text": "abc", "evidence_units": [unit, unit]})
    assert clean_documents_legacy([{"doc_id": "d", "text": "too short"}]) == []
    assert clean_documents_legacy([{"doc_id": "d", "text": "<p>" + "x " * 60 + "</p>"}])[0]["text"] == ("x " * 60).strip()
    assert clean_documents([{"doc_id": "d", "text": "ab"}], min_chars=3) == []


def test_structure_keeps_fitting_code_block_together():
    code = "```py\nx = 1\ny = 2\n```\n"
    text = "# H\n\n" + code + "\n" + "x" * 80
    chunks = StructureRecursiveChunker(32).chunk({"doc_id": "d", "text": text})
    assert any(code in chunk.text for chunk in chunks)
    assert ParentChildChunker(8, 2, 16).chunk_with_parents({"doc_id": "d", "text": ""}) == ([], [])


def test_schema_json_roundtrip_and_unknown_provenance():
    examples = [Document("d", "text"), EvidenceUnit("u", "d", "s", 0, "text", 0, 4),
                Chunk("c", "d", "text", 4, 0), Query("q", "Question?"),
                ContextBundle(), GenerationRecord("q")]
    for record in examples:
        assert type(record).from_dict(json.loads(json.dumps(record.to_dict()))) == record
    assert Document.from_dict({"doc_id": "d", "raw_annotations": [1]}).metadata["raw_annotations"] == [1]
    with pytest.raises(ValueError):
        ContextBundle.from_dict({"wrong_field": 1})
    assert signature({"b": 1, "a": 2}) == signature({"a": 2, "b": 1})
    with pytest.raises(ValueError):
        signature(float("nan"))
    assert FixedSizeChunker(10, 0).chunk(Document("d", "ok"))[0].text == "ok"
