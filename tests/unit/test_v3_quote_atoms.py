import pytest
from ragscope.v3.quote_atoms import quote_spans


def test_quotes_locate_original_unicode_and_repeated_claims():
    text='Alpha did not win. Alpha did not win.'
    out=quote_spans(text,[{'quote':'Alpha did not win.','kind':'claim'},{'quote':'Alpha did not win.','kind':'claim'}])
    assert out[0]['start']==0 and out[1]['start']==19
    with pytest.raises(ValueError,match='exact ordered'):
        quote_spans(text,[{'quote':'Alpha won.','kind':'claim'}])
    with pytest.raises(ValueError,match='drops original'):
        quote_spans('Only on Delta, Alpha won.',[{'quote':'Alpha won.','kind':'claim'}])
