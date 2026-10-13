"""Versioned official Python documentation cases; engineering fixtures only."""

from __future__ import annotations

import re
from .qasper import digest, match_key, normalize_text


REVISION = "v3.12.7"
BASE_URL = f"https://raw.githubusercontent.com/python/cpython/{REVISION}/Doc/tutorial/"
AUTHORSHIP = "agent_authored_not_human_validated"

# Each literal needle is checked against the downloaded version before freezing.
# These are source-grounded engineering examples, not an independent benchmark.
CASE_SPECS = {
    "datastructures": [
        ("What does list.append(x) do?", "Add an item to the end of the list.", "Adds an item to the end of the list."),
        ("What does list.extend(iterable) do?", "Extend the list by appending all the items from the iterable.", "Appends all the items from the iterable to the list."),
        ("Which exception does list.remove(x) raise if x is absent?", "Remove the first item from the list whose value is equal", "ValueError"),
        ("What does list.pop() do when no index is specified?", "If no index is specified", "Removes and returns the last item in the list."),
        ("What does list.clear() do?", "Remove all items from the list.", "Removes all items from the list."),
        ("What index does list.index(x) return?", "Return zero-based index in the list", "The zero-based index of the first item whose value is equal to x."),
        ("What does list.count(x) return?", "Return the number of times", "The number of times x appears in the list."),
        ("Does list.sort() sort the list in place?", "Sort the items of the list in place", "Yes"),
        ("Does list.reverse() reverse the list in place?", "Reverse the elements of the list in place.", "Yes"),
        ("Does list.copy() return a shallow or deep copy?", "Return a shallow copy of the list.", "A shallow copy."),
        ("Which list method removes the top item of a list-based stack?", "To retrieve an item from the top of the stack", "pop() without an explicit index."),
        ("Which collection is recommended for implementing a queue?", "To implement a queue", "collections.deque"),
    ],
    "errors": [
        ("What are syntax errors also known as?", "Syntax errors, also known as parsing errors", "Parsing errors."),
        ("Can a syntactically correct expression still cause an execution error?", "Even if a statement or expression is syntactically correct", "Yes"),
        ("Which exception signals a user interruption using Control-C?", "is signalled by raising", "KeyboardInterrupt"),
        ("What happens to the except clause if no exception occurs?", "If no exception occurs, the", "It is skipped."),
        ("What happens when no matching exception handler is found?", "if no handler is found", "Execution stops with an error message."),
        ("How many handlers of one try statement execute for an exception?", "At most one handler will be executed.", "At most one."),
        ("Does an except clause match instances of derived exception classes?", "A class in an", "Yes, instances of the named class and its derived classes."),
        ("Which clause is triggered when multiple except clauses could match?", "the first matching", "The first matching except clause."),
        ("From which class should user-defined exceptions typically derive?", "Exceptions should typically", "Exception, directly or indirectly."),
        ("Which built-in exception wraps several exception instances together?", "wraps a list of exception instances", "ExceptionGroup"),
    ],
    "inputoutput": [
        ("Which function is intended to produce human-readable string representations?", "The :func:`str` function is meant", "str()"),
        ("What is the default text file encoding if encoding is not specified?", "If *encoding* is not specified", "It is platform dependent."),
        ("Which text encoding does this tutorial recommend by default?", "Because UTF-8 is the modern de-facto standard", "UTF-8"),
        ("Can encoding be specified when opening a file in binary mode?", "You can not specify *encoding*", "No"),
        ("What is an advantage of using with for a file object?", "The advantage is that the file is properly closed", "The file is closed after the suite finishes, even if an exception occurs."),
        ("What does f.read() return after reaching the end of a text file?", "If the end of the file has been reached", "An empty string."),
        ("Which method reads a single line from a file?", "``f.readline()`` reads a single line", "f.readline()"),
        ("How can all lines of a file be read into a list?", "``list(f)`` or ``f.readlines()``.", "list(f) or f.readlines()."),
        ("Which method changes the position of a file object?", "To change the file object's position", "f.seek(offset, whence)"),
        ("What encoding must JSON files use according to this tutorial?", "JSON files must be encoded", "UTF-8"),
    ],
}


def build_technical_document(name: str, raw: str) -> tuple[dict, list[dict]]:
    """Preserve RST including code/markup and exact canonical paragraph offsets."""
    text = normalize_text(raw)
    doc_id = f"python:{REVISION}:{name}"
    units = []
    for index, block in enumerate(re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text)):
        start, end = block.span()
        units.append({"unit_id": f"{doc_id}:p{index}:{digest(block.group())[:12]}",
                      "doc_id": doc_id, "section_id": "body", "paragraph_index": index,
                      "text": block.group(), "canonical_start": start, "canonical_end": end,
                      "kind": "paragraph", "coordinate_system": "canonical_document_char"})
    document = {"doc_id": doc_id, "text": text, "source": "python/cpython", "source_revision": REVISION,
                "language": "en", "title": name.replace("inputoutput", "Input and Output").replace("_", " "),
                "sections": [{"section_id": "body", "title": name}], "raw_checksum": digest(raw),
                "canonical_checksum": digest(text), "split": "engineering", "evidence_units": units,
                "metadata": {"source_url": BASE_URL + name + ".rst", "license": "PSF-2.0",
                             "format": "rst", "authorship": AUTHORSHIP}}
    cases = []
    for index, (question, needle, answer) in enumerate(CASE_SPECS[name]):
        matches = [u for u in units if match_key(needle) in match_key(u["text"])]
        if len(matches) != 1:
            raise ValueError(f"{name} case {index}: evidence needle matched {len(matches)} units: {needle}")
        unit = matches[0]
        cases.append({"query_id": f"{doc_id}:case{index:02d}", "question": question,
                      "split": "engineering", "scope": {"kind": "document", "doc_ids": [doc_id]},
                      "doc_id": doc_id, "answerability": "answerable", "answer_refs": [answer],
                      "evidence_reference_sets": [[unit["unit_id"]]], "annotation_disagreements": [],
                      "metadata": {"authorship": AUTHORSHIP, "source_url": BASE_URL + name + ".rst",
                                   "source_revision": REVISION, "evidence_needle": needle,
                                   "evidence_canonical_span": [unit["canonical_start"], unit["canonical_end"]],
                                   "independent_quality_benchmark": False}})
    return document, cases
