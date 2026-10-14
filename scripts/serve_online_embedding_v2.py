"""Prepare or supervise the existing CPU embedding service for online/UI queries.

--check-only is local metadata only. --run requires a coordinator release receipt.
No corpus encoding, model download, automatic restart, or original-job operation.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
import requests
from filelock import FileLock, Timeout
from ragscope.experiments.runtime import configure
from ragscope.experiments.ledger import now, read_records
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from scripts.materialize_squad_v2 import BINARY_SHA, MODEL_SHA, port_free, process_absent

HOME = Path('D:/RAGScope_Runtime')
MAIN = HOME/'runs/20261014_v2'
DIRECTORY = MAIN/'online_embedding'
BASE_URL = 'http://127.0.0.1:11436'
START_RAM = 3 * 1024**3
LOAD_FLOOR = 1024**3


def lock_observations(paths):
    result = {}
    for path in paths:
        try:
            with FileLock(str(path), timeout=0):
                pass
            result[str(path)] = 'AVAILABLE'
        except Timeout:
            result[str(path)] = 'HELD'
    return result


def port_observation(port):
    try:
        port_free(port)
        return 'BIND_FREE'
    except OSError:
        return 'OCCUPIED'


def shared_locks():
    return [HOME/'runs/gpu.lock', HOME/'runs/generation-client.lock',
            HOME/'runs/20261014_v2_retrieval/writer.lock']


def validate_command(command):
    expected = {'--host':'127.0.0.1','--port':'11436','-ngl':'0','-t':'4','-tb':'4',
                '--pooling':'last','--embd-normalize':'2','-c':'2048','-b':'1100','-ub':'1100',
                '-np':'1','--threads-http':'2'}
    if '--embedding' not in command or '--no-op-offload' not in command:
        raise ValueError('CPU embedding flags missing')
    for key,value in expected.items():
        if command.count(key) != 1 or command[command.index(key)+1] != value:
            raise ValueError(f'Pinned command differs: {key}')


def prepare():
    """Hash fixed assets and observe locks/ports without HTTP or process creation."""
    runtime = configure(cpu_only=True)
    old = json.loads((MAIN/'embedding_service.json').read_text(encoding='utf-8'))
    lifetime_path = HOME/'runs/20261014_v2_retrieval/SQUAD_FINAL/SERVICE_LIFETIME.json'
    lifetime = json.loads(lifetime_path.read_text(encoding='utf-8'))
    command = list(lifetime['command'])
    validate_command(command)
    binary, model = Path(command[0]), Path(command[command.index('-m')+1])
    if file_hash(binary) != BINARY_SHA or file_hash(model) != MODEL_SHA:
        raise ValueError('Pinned CPU executable or GGUF changed')
    identity_path = MAIN/'embedding_identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    if identity['device'] != 'cpu' or identity['dimensions'] != 1024 or identity['weight_sha256'] != MODEL_SHA:
        raise ValueError('Existing semantic identity differs')
    model_dir = model.parent
    tokenizer_manifest = json.loads((model_dir/'tokenizer_manifest.json').read_text(encoding='utf-8'))
    if not verify_artifacts(tokenizer_manifest['artifacts']):
        raise ValueError('Frozen tokenizer resources changed')
    ready_path = HOME/'runs/20261014_v2_retrieval/STRUCTURED_INDEX_READY.json'
    ready = json.loads(ready_path.read_text(encoding='utf-8'))
    if not verify_artifacts([ready['manifest']]):
        raise ValueError('Structured index manifest changed')
    manifest = json.loads(Path(ready['manifest']['path']).read_text(encoding='utf-8'))
    if (manifest['count'] != 6551 or manifest['identity']['embedding'] != identity
            or manifest['signature'] != signature(manifest['identity'])
            or not verify_artifacts(manifest['artifacts'])):
        raise ValueError('Structured index identity or bytes changed')
    cache = HOME/'cache/ragscope/v2_embeddings'/signature(identity)
    if not cache.is_dir():
        raise ValueError('Existing model-specific cache namespace missing')
    cache_samples = []
    for p in sorted(cache.glob('*.json')):
        if p.name.endswith('.usage.json'):
            continue
        item = json.loads(p.read_text(encoding='utf-8'))
        if item['model'] != identity or not verify_artifacts([item['array']]):
            raise ValueError('Observed cache identity/array changed')
        cache_samples.extend([artifact(p), item['array']])
        if len(cache_samples) == 6:
            break
    if len(cache_samples) != 6:
        raise ValueError('Three real cache samples required')
    model_manifest = {'identity':identity,'artifacts':[artifact(binary),artifact(model),
        artifact(model_dir/'Qwen3-Embedding-0.6B-Q8_0.gguf.source.json'),
        artifact(model_dir/'tokenizer_manifest.json'),*tokenizer_manifest['artifacts']],
        'note':'Local binding of previously downloaded fixed assets; not a new upstream model manifest.'}
    files = [artifact(identity_path), artifact(lifetime_path), artifact(ready_path),ready['manifest'],
        *manifest['artifacts'],*cache_samples,artifact(Path(__file__)),
        artifact(ROOT/'scripts/materialize_squad_v2.py'),artifact(ROOT/'scripts/runtime_env.ps1'),
        artifact(ROOT/'ragscope/indexing/local_embedding.py')]
    return {'status':'CHECKED_NO_LAUNCH_NO_HTTP','time':now(),'command':command,'runtime':runtime,
        'model_input_manifest':model_manifest,'model_input_manifest_hash':signature(model_manifest),
        'artifacts':files,'identity':identity,'identity_artifact':artifact(identity_path),
        'cache_namespace':str(cache),'cache_identity_signature':signature(identity),'cache_samples':3,
        'cache_scope':'Three existing arrays/hash manifests checked; complete index bytes checked; no global cache mutation.',
        'index_signature':manifest['signature'],'index_count':6551,'previous_service_absent':process_absent(old),
        'ports':{'11436':port_observation(11436),'11435':port_observation(11435)},
        'locks':lock_observations(shared_locks()),'available_ram':psutil.virtual_memory().available,
        'start_guard_bytes':START_RAM,'load_floor_bytes':LOAD_FLOOR,'cpu_threads':4,'cuda_mask':os.environ['CUDA_VISIBLE_DEVICES'],
        'model_calls':0,'http_calls':0,'service_started':False,'automatic_retry':False,
        'semantic_identity_policy':'Preserve embedding_identity.json bytes; sign its SHA in service receipt only after health and real 1024-dimensional control pass.',
        'stop_policy':'Coordinator request after online/UI clients quiesce; acquire shared locks, single-slot drain, exact PID/create/argv termination; save actual exit code.'}


def exact_child(process, created, command):
    try:
        return abs(process.create_time()-created)<.01 and process.cmdline()==command
    except psutil.NoSuchProcess:
        return False


def cleanup_owned_child(child, process, created, command, handed_off, safe_to_stop):
    """Once clients may use the service, a supervisor error must not kill their requests."""
    if (child is not None and child.poll() is None and process is not None
            and (not handed_off or safe_to_stop) and exact_child(process,created,command)):
        process.terminate()
        process.wait(timeout=15)


def control_request(text):
    response = requests.post(BASE_URL+'/v1/embeddings',json={'input':text},timeout=30)
    response.raise_for_status()
    payload = response.json()
    vector = payload['data'][0]['embedding']
    if len(vector) != 1024 or not all(math.isfinite(v) for v in vector) or not any(vector):
        raise ValueError('CPU control vector dimensions/nonfinite/zero failure')
    return {'dimensions':len(vector),'usage':payload.get('usage'),'response_hash':signature(payload)}


def request_stop(directory, reason):
    lifetime = json.loads((directory/'SERVICE_LIFETIME.json').read_text(encoding='utf-8'))
    if lifetime['status'] not in {'READY_FOR_ONLINE_UI','LOW_RAM_OBSERVED'}:
        raise ValueError('Service is not in its ready lifetime')
    body = {'time':now(),'reason':reason,'clients_quiesced_by_coordinator':True,
            'server_pid':lifetime['server_pid'],'server_creation_time':lifetime['server_creation_time'],
            'lifetime_started_at':lifetime['started_at']}
    path = directory/'STOP_REQUEST.json'
    if path.exists():
        raise FileExistsError('Stop already requested; inspect the existing lifetime')
    atomic_json(path,body)
    return body


def run(prepared, directory, handoff, handoff_sha):
    if file_hash(handoff) != handoff_sha:
        raise ValueError('Coordinator handoff SHA mismatch')
    if not verify_artifacts(prepared['artifacts']+prepared['model_input_manifest']['artifacts']):
        raise ValueError('Prepared source/artifact changed')
    deadline = datetime.fromisoformat(json.loads((MAIN/'EXECUTION_DEADLINE.json').read_text())['deadline_utc'])
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('Hard deadline reached; no new load')
    receipt_path = directory/'SERVICE_LIFETIME.json'
    if receipt_path.exists():
        raise FileExistsError('This one-shot service lifetime already attempted; no restart')
    old = json.loads((MAIN/'embedding_service.json').read_text(encoding='utf-8'))
    if not process_absent(old):
        raise RuntimeError('Previous embedding identity still alive')
    journal = HOME/'runs/20261014_v2_retrieval/embedding_runtime_failures.jsonl'
    if len(read_records(journal)) != 3:
        raise ValueError('Historical exhausted backend journal changed')
    worker = psutil.Process()
    receipt = {'status':'RESERVED_BEFORE_LOAD','started_at':now(),'worker_pid':worker.pid,
        'worker_creation_time':worker.create_time(),'worker_command':worker.cmdline(),
        'handoff':artifact(handoff),'preparation':artifact(directory/'PREPARATION.json'),
        'command':prepared['command'],'identity':prepared['identity_artifact'],
        'failure_journal_before':artifact(journal),'automatic_restart':False,'body_encodings':0}
    child, process, created, service = None,None,None,None
    handed_off, safe_to_stop = False, False
    with FileLock(str(directory/'service.lock'),timeout=0):
        with ExitStack() as launch_locks:
            for path in shared_locks():
                launch_locks.enter_context(FileLock(str(path),timeout=0))
            port_free(11436)
            port_free(11435)  # The coordinator hands over after the previous GPU service exits.
            available = psutil.virtual_memory().available
            if available < START_RAM:
                raise MemoryError('Fresh 3 GiB available RAM load guard not met')
            receipt['available_before'] = available
            with receipt_path.open('x',encoding='utf-8') as stream:
                json.dump(receipt,stream);stream.flush();os.fsync(stream.fileno())
            atomic_json(directory/'embedding_service_before.json',old)
            try:
                with (directory/'embedding_cpu4.log').open('w',encoding='utf-8') as log:
                    child = subprocess.Popen(prepared['command'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
                    process=psutil.Process(child.pid);created=process.create_time()
                    receipt.update(server_pid=child.pid,server_creation_time=created)
                    atomic_json(receipt_path,receipt)
                    end=time.monotonic()+60
                    while time.monotonic()<end:
                        if child.poll() is not None or psutil.virtual_memory().available<LOAD_FLOOR:
                            raise RuntimeError('Child exit or RAM below1GiB during CPU load')
                        try:
                            if requests.get(BASE_URL+'/health',timeout=1).status_code==200:
                                break
                        except requests.RequestException:
                            pass
                        time.sleep(.25)
                    else:
                        raise TimeoutError('CPU health timeout60s')
                    if not exact_child(process,created,prepared['command']):
                        raise ValueError('CPU child identity changed before control')
                    listeners=[c.pid for c in psutil.net_connections(kind='tcp') if c.laddr and c.laddr.port==11436 and c.status==psutil.CONN_LISTEN]
                    if set(listeners)!={child.pid}:
                        raise ValueError('11436 listener does not belong to exact CPU child')
                    receipt['startup_control']=control_request('Online and UI CPU four thread verification')
                    if psutil.virtual_memory().available<LOAD_FLOOR:
                        raise MemoryError('CPU control postcheck RAM below1GiB')
                    # Semantic identity is unchanged; only this live process attestation is new.
                    if not verify_artifacts([prepared['identity_artifact']]):
                        raise ValueError('Embedding semantic identity changed while loading')
                    service={'status':'RUNNING_ONLINE_UI','pid':child.pid,'creation_time':created,'command':prepared['command'],
                        'device':'cpu','cuda_mask':'','binary_sha256':BINARY_SHA,'model_sha256':MODEL_SHA,
                        'performance_identity':{'threads':4,'batch_threads':4},'parent_pid':worker.pid,'parent_alive':True,
                        'supervisor_pid':worker.pid,'lifetime_receipt':str(receipt_path),'time':now(),
                        'embedding_identity':prepared['identity_artifact'],'model_input_manifest_hash':prepared['model_input_manifest_hash']}
                    atomic_json(MAIN/'embedding_service.json',service)
                    receipt.update(status='READY_FOR_ONLINE_UI',ready_at=now(),available_after=psutil.virtual_memory().available)
                    atomic_json(receipt_path,receipt)
                    handed_off = True
                    launch_locks.close()  # Online/GPU owner may now take the shared leases.
                    print(json.dumps({'status':receipt['status'],'worker_pid':worker.pid,'service_pid':child.pid,'available':receipt['available_after']}),flush=True)
                    while not (directory/'STOP_REQUEST.json').exists():
                        if child.poll() is not None:
                            raise RuntimeError('Owned CPU service exited; no automatic restart')
                        available=psutil.virtual_memory().available
                        if available<LOAD_FLOOR and receipt['status']!='LOW_RAM_OBSERVED':
                            receipt.update(status='LOW_RAM_OBSERVED',low_ram_time=now(),low_ram_bytes=available)
                            atomic_json(receipt_path,receipt)
                            print(json.dumps({'status':'LOW_RAM_OBSERVED','available':available,'action':'Notify coordinator; no mid-request termination'}),flush=True)
                        time.sleep(1)
                    stop=json.loads((directory/'STOP_REQUEST.json').read_text(encoding='utf-8'))
                    if (stop['server_pid']!=child.pid or stop['server_creation_time']!=created
                            or stop['lifetime_started_at']!=receipt['started_at'] or not stop['clients_quiesced_by_coordinator']):
                        raise ValueError('Stop request belongs to a different service lifetime')
                    with ExitStack() as stopping:
                        for path in shared_locks():
                            stopping.enter_context(FileLock(str(path),timeout=30))
                        safe_to_stop = True
                        receipt['drain_control']=control_request('Final single slot drain after online and UI clients exited')
                        if not exact_child(process,created,prepared['command']):
                            raise ValueError('CPU child identity changed before planned stop')
                        process.terminate();process.wait(timeout=15)
                        receipt.update(status='COMPLETED_PLANNED_STOP',stop_request=artifact(directory/'STOP_REQUEST.json'))
            except BaseException as exc:
                receipt.update(status='SUPERVISOR_FAILED_SERVICE_RETAINED' if handed_off and not safe_to_stop else 'FAILED_RETAINED_NO_RETRY',
                    error={'type':type(exc).__name__,'message':str(exc)})
                raise
            finally:
                cleanup_owned_child(child,process,created,prepared['command'],handed_off,safe_to_stop)
                receipt.update(ended_at=now(),service_exited=child is None or child.poll() is not None,
                    service_exit_code=None if child is None else child.poll(),
                    failure_journal_unchanged=file_hash(journal)==receipt['failure_journal_before']['sha256'],
                    available_after_exit=psutil.virtual_memory().available,port11436=port_observation(11436))
                if service is not None:
                    service.update(status='EXITED_ONLINE_UI' if receipt['service_exited'] else 'RUNNING_WITHOUT_SUPERVISOR_REQUIRES_COORDINATOR',
                        ended_at=now(),exit_code=receipt['service_exit_code'],parent_alive=False)
                    atomic_json(MAIN/'embedding_service.json',service)
                atomic_json(receipt_path,receipt)
                print(json.dumps({'status':receipt['status'],'service_exited':receipt['service_exited'],'exit_code':receipt['service_exit_code']}),flush=True)
    return 0 if receipt['status']=='COMPLETED_PLANNED_STOP' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--request-stop',action='store_true')
    parser.add_argument('--handoff-receipt',type=Path)
    parser.add_argument('--handoff-sha256')
    parser.add_argument('--reason',default='Online and final UI clients have completed and exited')
    args=parser.parse_args()
    configure(cpu_only=True)
    DIRECTORY.mkdir(exist_ok=True)
    if args.request_stop:
        print(json.dumps(request_stop(DIRECTORY,args.reason)));return 0
    if args.run:
        if not args.handoff_receipt or not args.handoff_sha256:
            raise ValueError('--run requires actual release receipt and its SHA')
        prepared=json.loads((DIRECTORY/'PREPARATION.json').read_text(encoding='utf-8'))
        return run(prepared,DIRECTORY,args.handoff_receipt,args.handoff_sha256)
    path=DIRECTORY/'PREPARATION.json'
    if path.exists():
        raise FileExistsError('Preserve the existing no-launch preparation; do not overwrite it')
    prepared=prepare()
    atomic_json(path,prepared)
    print(json.dumps({'status':prepared['status'],'receipt':artifact(path),'ports':prepared['ports'],'locks':prepared['locks'],'http_calls':0,'model_calls':0}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
