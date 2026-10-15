"""Explicit local-only workbench launch; Ctrl+C or an own stop file releases models."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import atomic_json,canonical_json
from ragscope.v3.hierarchical_runtime import load_hierarchy
from ragscope.v3.online_runtime import OnlineOwnedOllama, UncachedOnlineEmbedding
from ragscope.v3.workbench import Workbench


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--port',type=int,default=11437);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    run=Path(a.campaign).resolve();runtime=Path(configure(cpu_only=True)['home']).resolve()
    if run.parent!=runtime/'runs' or not 1024<=a.port<=65535:raise ValueError('Invalid campaign/port')
    if a.dry_run:print(canonical_json({'host':'127.0.0.1','port':a.port,'download':False,'default':'B0','C1':'MANUAL_DIAGNOSTIC_ONLY'}));return 0
    require_ram(2*1024**3);engine=load_hierarchy(run)
    frozen=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf8'))
    dev_ids={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
    original,_=model_client_factory(frozen['generator_spec'],dev_ids)
    stop=run/('WORKBENCH_STOP_'+str(time.time_ns()));receipt=run/'WORKBENCH_SERVICE.json'
    with FileLock(str(run/'writer.lock'),timeout=0), OnlineOwnedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
        work=Workbench(run,engine,service,UncachedOnlineEmbedding(run,service),original)
        class Handler(BaseHTTPRequestHandler):
            def respond(self,status,payload):
                data=canonical_json(payload).encode('utf8');self.send_response(status)
                self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)))
                self.end_headers();self.wfile.write(data)
            def do_GET(self):
                if self.path!='/api/state':self.respond(404,{'message':'Unknown route'});return
                self.respond(200,{'status':'READY','campaign':run.name,'default':'B0','C1_admitted':False,
                    'human_review_status':'NOT_REVIEWED','stop_file':str(stop),'documents_n':len(engine.documents)})
            def do_POST(self):
                if self.path!='/api/answer':self.respond(404,{'message':'Unknown route'});return
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<=65536:raise ValueError('Bounded JSON body required')
                    payload=json.loads(self.rfile.read(n));self.respond(200,work.answer(payload))
                except (ValueError,TypeError) as exc:self.respond(400,{'action':'INPUT_ERROR','message':str(exc)})
                except Exception as exc:self.respond(503,{'action':'EXECUTION_ERROR','message':str(exc)})
        server=HTTPServer(('127.0.0.1',a.port),Handler);server.timeout=.5
        atomic_json(receipt,{'status':'READY','base_url':f'http://127.0.0.1:{a.port}','backend':str(service.path),'stop_file':str(stop)})
        print(canonical_json({'status':'READY','receipt':str(receipt)}),flush=True)
        try:
            while not stop.exists():server.handle_request()
        except KeyboardInterrupt:pass
        finally:
            server.server_close();atomic_json(receipt,{'status':'EXITED','stop_file':str(stop),'original_training_policy':'USER_MANUAL_RESUME'})
    return 0


if __name__=='__main__':raise SystemExit(main())
