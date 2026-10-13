import os
import pytest

from scripts.audit_storage_v2 import inventory


def test_hard_links_do_not_double_unique_lengths(tmp_path):
    original = tmp_path / "models" / "weights.bin"
    original.parent.mkdir()
    original.write_bytes(b"12345678")
    os.link(original, tmp_path / "alias.bin")
    report = inventory(tmp_path)
    assert report["totals"] == {"files": 2, "logical_bytes": 16,
                                "unique_file_logical_bytes": 8}
    assert original.read_bytes() == b"12345678"


def test_file_link_outside_runtime_is_reported_without_reading(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    outside = tmp_path / "user-data.txt"
    outside.write_text("untouched")
    try:
        (root / "alias").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Host cannot create a test symlink: {exc}")
    report = inventory(root)
    assert report["totals"]["files"] == 0
    assert report["links"][0]["within_runtime"] is False
    assert report["links"][0]["traversed"] is False
    assert outside.read_text() == "untouched"
