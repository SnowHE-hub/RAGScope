"""Serve the completed local corpus, with reliable ownership and clean exit."""
import argparse
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from finish_experiments import original_client
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import atomic_json,canonical_json,verify_artifacts
from ragscope.v3.full_retrieval import FullCorpusRetriever
from ragscope.v3.full_workbench import FullWorkbench
from ragscope.v3.finish_runtime import ReusableEncoder
from ragscope.v3.online_runtime import OnlineOwnedOllama


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(PROJECT/'configs/v3/default.json'))
    parser.add_argument('--port',type=int,default=11437)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();configure(cpu_only=True)
    config=json.loads(Path(args.config).read_text('utf8'))
    run,source=Path(config['run']),Path(config['source'])
    frozen=json.loads((run/'FINAL_CONFIGURATION.json').read_text('utf8'))
    if not verify_artifacts(frozen['source_artifacts']):raise ValueError('Final inference source changed; rebuild and validate configuration before launching')
    deployment_path=run/('DEPLOYMENT_CONFIGURATION_RESIDENT8K.json' if (run/'DEPLOYMENT_CONFIGURATION_RESIDENT8K.json').exists() else 'DEPLOYMENT_CONFIGURATION.json')
    deployed=json.loads(deployment_path.read_text('utf8'))
    revision_path=run/'APPLICATION_REVISION.json'
    sources=deployed['source_artifacts']
    if revision_path.exists():
        from ragscope.indexing.identity import artifact
        revision=json.loads(revision_path.read_text('utf8'))
        if revision['base_deployment_configuration']!=artifact(deployment_path):raise ValueError('Application revision base differs')
        sources=[revision['implementation'] if Path(ref['path']).resolve()==Path(revision['implementation']['path']).resolve() else ref for ref in sources]
    if not verify_artifacts(sources):raise ValueError('Deployment workflow source changed')
    engine=FullCorpusRetriever(source,run)
    selection=json.loads((run/'GENERATION_SELECTION.json').read_text('utf8'))
    original=original_client(run.parents[1],selection['model'])
    if args.check_only:
        print(canonical_json({'status':'PASS','papers':len(engine.by_doc),'chunks':len(engine.chunks),'model':original.model}));return
    stop=run/('WORKBENCH_STOP_'+str(time.time_ns()));receipt=run/'FULL_WORKBENCH_SERVICE.json'
    with OnlineOwnedOllama(run,run.parents[1]/'tools/ollama-0.34.4/ollama.exe',num_ctx=8192,startup_ram_bytes=512*1024**2) as service:
        work=FullWorkbench(run,engine,service,ReusableEncoder(run,source,service,engine.embedding_identity),original)
        class Handler(BaseHTTPRequestHandler):
            def respond(self,status,payload):
                data=canonical_json(payload).encode('utf8');self.send_response(status)
                self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_GET(self):
                if self.path!='/api/state':self.respond(404,{'message':'Unknown route'});return
                self.respond(200,{'status':'READY','papers':len(engine.by_doc),'chunks':len(engine.chunks),'model':original.model,
                    'documents':engine.documents,'stop_file':str(stop),'index_status':'COMPLETE'})
            def do_POST(self):
                if self.path!='/api/answer':self.respond(404,{'message':'Unknown route'});return
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=65536:raise ValueError('Bounded JSON body required')
                    self.respond(200,work.answer(json.loads(self.rfile.read(length))))
                except (ValueError,TypeError) as exc:self.respond(400,{'action':'INPUT_ERROR','message':str(exc)})
                except Exception as exc:self.respond(503,{'action':'EXECUTION_ERROR','message':str(exc)})
        server=HTTPServer(('127.0.0.1',args.port),Handler);server.timeout=.5
        atomic_json(receipt,{'status':'READY','url':f'http://127.0.0.1:{args.port}','stop_file':str(stop),'owned_backend':str(service.path)})
        print('FULL_WORKBENCH_READY',flush=True)
        try:
            while not stop.exists():server.handle_request()
        except KeyboardInterrupt:pass
        finally:
            server.server_close();atomic_json(receipt,{'status':'EXITED','stop_file':str(stop),'training_policy':'USER_MANUAL_RESUME'})


if __name__=='__main__':main()
