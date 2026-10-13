"""Offline reconstruction accounting fixtures, never real-model quality claims."""
import numpy as np
import pytest

from ragscope.experiments.legacy_reconstruction import merge_rrf, validate_vectors, encode


def hit(name):
    return {"chunk_id":name,"doc_id":name,"text":"offline fixture","score":1.}


def test_legacy_hybrid_preserves_twenty_candidates_and_dense_first_ties():
    dense=[hit(f"d{i}") for i in range(25)]
    sparse=[hit(f"s{i}") for i in range(25)]
    results=merge_rrf(dense,sparse,k=40)
    assert [row['chunk_id'] for row in results[:4]]==['d0','s0','d1','s1']
    assert len(results)==40
    assert 'd20' not in {row['chunk_id'] for row in results}
    assert results[0]['score']==pytest.approx(.5/61)


def test_shared_chunk_accumulates_both_ranks_without_duplicate_row():
    results=merge_rrf([hit('a'),hit('b')],[hit('b'),hit('c')])
    assert [row['chunk_id'] for row in results]==['b','a','c']
    assert results[0]['score']==pytest.approx(.5/61+.5/62)


@pytest.mark.parametrize('values',[np.array([[0.,0.]],dtype=np.float32),
    np.array([[float('nan'),1.]],dtype=np.float32),np.array([[1.,0.]],dtype=np.float64)])
def test_invalid_reconstruction_vectors_rejected(values):
    with pytest.raises(ValueError):validate_vectors(values,1,2)


def test_cpu_guard_rejects_model_loading_before_any_import(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CPU-only'):
        encode(tmp_path,tmp_path)
