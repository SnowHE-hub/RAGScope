import socket
import urllib.request

import pytest
import requests

from scripts.verify_ui_candidate_v2 import offline_transport


def test_offline_qa_blocks_both_application_http_transports():
    with offline_transport() as attempts:
        with pytest.raises(RuntimeError, match='HTTP request refused'):
            requests.post('http://127.0.0.1:11436/v1/embeddings', json={'input': 'fixture'})
        with pytest.raises(RuntimeError, match='HTTP request refused'):
            urllib.request.build_opener().open('http://127.0.0.1:11435/api/chat')
    assert len(attempts) == 2


def test_offline_qa_allows_windows_framework_self_pipe():
    with offline_transport() as attempts:
        left, right = socket.socketpair()
        try:
            left.sendall(b'local fixture')
            assert right.recv(32) == b'local fixture'
        finally:
            left.close()
            right.close()
    assert attempts == []
