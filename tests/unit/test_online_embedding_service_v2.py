import json
from pathlib import Path
import pytest
from scripts import serve_online_embedding_v2 as service


def test_request_stop_binds_lifetime_without_contacting_process_or_http(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Stop request must only write its coordinator file')
    monkeypatch.setattr(service.requests, 'post', blocked)
    monkeypatch.setattr(service.requests, 'get', blocked)
    monkeypatch.setattr(service.psutil, 'Process', blocked)
    lifetime={'status':'READY_FOR_ONLINE_UI','server_pid':42,'server_creation_time':123.5,'started_at':'frozen-start'}
    (tmp_path/'SERVICE_LIFETIME.json').write_text(json.dumps(lifetime))
    request=service.request_stop(tmp_path,'All clients exited')
    assert request['server_pid']==42 and request['server_creation_time']==123.5
    assert request['lifetime_started_at']=='frozen-start'
    with pytest.raises(FileExistsError):
        service.request_stop(tmp_path,'duplicate')


def test_unready_service_rejects_stop(tmp_path):
    (tmp_path/'SERVICE_LIFETIME.json').write_text(json.dumps({'status':'RESERVED_BEFORE_LOAD'}))
    with pytest.raises(ValueError,match='not in its ready'):
        service.request_stop(tmp_path,'not yet safe')
    assert not (tmp_path/'STOP_REQUEST.json').exists()


def test_mismatched_release_hash_never_launches(tmp_path,monkeypatch):
    handoff=tmp_path/'release.json';handoff.write_text('{}')
    monkeypatch.setattr(service.subprocess,'Popen',lambda *a,**kw:pytest.fail('Must not create process'))
    with pytest.raises(ValueError,match='handoff SHA'):
        service.run({},tmp_path,handoff,'wrong')
    assert not (tmp_path/'SERVICE_LIFETIME.json').exists()


def test_exact_child_rejects_pid_reuse_and_changed_command():
    class Child:
        def create_time(self):return 123.5
        def cmdline(self):return ['owned','--cpu']
    assert service.exact_child(Child(),123.5,['owned','--cpu'])
    assert not service.exact_child(Child(),120,['owned','--cpu'])
    assert not service.exact_child(Child(),123.5,['other'])


def test_control_rejects_bad_dimension_and_nonfinite(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'data':[{'embedding':vector}]}
    monkeypatch.setattr(service.requests,'post',lambda *a,**kw:Response())
    vector=[1.0]*1023
    with pytest.raises(ValueError,match='dimensions'):
        service.control_request('fixture')
    vector=[1.0]*1024;vector[7]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        service.control_request('fixture')


@pytest.mark.parametrize('handed_off,safe_to_stop,terminates',[(False,False,True),(True,False,False),(True,True,True)])
def test_supervisor_failure_does_not_kill_handed_off_requests(handed_off,safe_to_stop,terminates):
    class Child:
        terminated=False
        def poll(self):return None
        def create_time(self):return 123.5
        def cmdline(self):return ['owned','--cpu']
        def terminate(self):self.terminated=True
        def wait(self,timeout):assert timeout==15
    child=Child()
    service.cleanup_owned_child(child,child,123.5,['owned','--cpu'],handed_off,safe_to_stop)
    assert child.terminated is terminates
