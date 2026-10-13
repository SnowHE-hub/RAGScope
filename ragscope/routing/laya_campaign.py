"""Freeze visible inputs and persist Laya probabilities before path quality labels."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import threading
import time

from ragscope.experiments.ledger import Campaign, RecordStore, read_records
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from .campaign import immutable_json, request_query
from .core import RoutingRequest
from .laya_adapter import LayaRouter, PERMUTATIONS, LAYA_MODEL_REVISION


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def freeze_visible_inputs(home, campaign):
    """Read corpus inputs only; never open generation results or routing labels."""
    home=Path(home)
    corpus=home/"corpora"/campaign
    split_path=corpus/"SPLIT_MANIFEST.json"
    split=read(split_path)
    route_path=home/"runs"/campaign/"ROUTING_QUERY_MANIFEST.json"
    route=read(route_path)
    if (route.get("manifest_hash")!=signature({k:v for k,v in route.items() if k!="manifest_hash"})
            or route.get("source")!=artifact(split_path)):
        raise ValueError("Parent routing query manifest changed")
    paths=[corpus/"documents.jsonl"]+[corpus/f"queries_{s}.jsonl" for s in ("train","dev","test")]
    if any(file_hash(path)!=split["files"][path.name]["sha256"] for path in paths):
        raise ValueError("Frozen corpus input changed")
    lengths={doc["doc_id"]:doc.get("metadata",{}).get("reference_n_tokens") for doc in jsonl(paths[0])}
    populations={}
    for split_name,n in (("train",200),("dev",60),("test",100)):
        ids=route["query_ids"][split_name]
        if len(ids)!=n or len(set(ids))!=n:raise ValueError("Wrong Laya frozen population size")
        queries={q["query_id"]:q for q in jsonl(corpus/f"queries_{split_name}.jsonl")}
        visible=[]
        for qid in ids:
            query=queries[qid]
            if query["split"]!=split_name:raise ValueError("Cross-split routing input")
            source={"query":query,"document_length":{"reference_n_tokens":lengths.get(query.get("doc_id"))}}
            visible.append(request_query(source))
        populations[split_name]={"query_ids":ids,"visible_queries":visible}
    body={"version":"laya-visible-inputs-v1","populations":populations,
          "sources":[artifact(path) for path in paths]+[artifact(split_path),artifact(route_path)],
          "no_quality_labels_read":True,"orders":[list(order) for order in PERMUTATIONS],"min_confidence":.5,
          "expected_queries":360,"expected_decisions":1080,"seed":42}
    body["input_hash"]=signature(body)
    destination=home/"runs"/(campaign+"_laya")/"INPUTS.json"
    immutable_json(destination,body)
    return body


def decisions_to_protocol(raw_directory, protocol, output):
    """Bind already committed default-order choices without accessing route scores."""
    raw_directory=Path(raw_directory)
    frozen=read(raw_directory/"INPUTS.json")
    if frozen["input_hash"]!=signature({k:v for k,v in frozen.items() if k!="input_hash"}):
        raise ValueError("Laya frozen inputs changed")
    summary=read(raw_directory/"SUMMARY.json")
    if summary.get("status")!="COMPLETE" or summary.get("input_hash")!=frozen["input_hash"]:
        raise ValueError("Actual Laya probability population is incomplete")
    rows=read_records(raw_directory/"records.jsonl")
    config=read(raw_directory/"records.config.json")
    if len(rows)!=1080 or any(not row.get("real_model") or row.get("config_hash")!=signature(config) for row in rows):
        raise ValueError("Laya actual record identity/denominator mismatch")
    raw={(row["split"],row["query_id"],row["order_index"]):row for row in rows}
    if len(raw)!=1080:raise ValueError("Duplicate Laya records")
    decisions={}
    for split in ("dev","test"):
        old={q["query_id"]:q for q in frozen["populations"][split]["visible_queries"]}
        pop=protocol["populations"][split]
        if set(pop["query_ids"])!=set(old):raise ValueError("Final protocol population differs from precomputed Laya")
        decisions[split]=[]
        for query in pop["visible_queries"]:
            if old[query["query_id"]]!=query:raise ValueError("Final visible routing input differs")
            row=raw[(split,query["query_id"],0)]
            decisions[split].append({"query_id":query["query_id"],"decision":row["decision"]})
    body={"name":"laya","protocol_hash":protocol["protocol_hash"],"decisions":decisions,
          "router_identity":summary["router_identity"],"precomputed_records":artifact(raw_directory/"records.jsonl")}
    body["decision_hash"]=signature(body)
    immutable_json(Path(output)/"DECISIONS_laya.json",body)
    return body


def run_raw(home,campaign, *, supervision=None):
    """CPU only, bounded and resumable. Resource coordination is external."""
    import psutil
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU-only runtime environment required")
    inputs=freeze_visible_inputs(home,campaign)
    directory=Path(home)/"runs"/(campaign+"_laya")
    if (directory/"SUMMARY.json").exists():
        summary=read(directory/"SUMMARY.json")
        if summary.get("status")=="COMPLETE" and summary.get("input_hash")==inputs["input_hash"]:
            if not verify_artifacts(summary["artifacts"]):raise ValueError("Completed Laya output changed")
            return summary
    minimum=(4.5 if supervision is not None else 5)*1024**3
    if psutil.virtual_memory().available<minimum:raise MemoryError("Laya pre-load RAM requirement is not met")
    if supervision is not None:supervision.checkpoint("LOADING")
    model=Path(home)/"models/laya"/LAYA_MODEL_REVISION
    manifest=read(model/"model_manifest.json")
    if manifest.get("status")!="VERIFIED" or manifest.get("revision")!=LAYA_MODEL_REVISION or not verify_artifacts(manifest["artifacts"]):
        raise ValueError("Laya source checkpoint changed")
    hashes={item["relative_path"]:item["sha256"] for item in manifest["artifacts"]}
    with Campaign(Path(home),campaign+"_laya") as ledger:
        process=psutil.Process(os.getpid())
        atomic_json(directory/"worker.json",{"campaign_id":ledger.campaign_id,"pid":process.pid,"creation_time":process.create_time(),
            "command":process.cmdline(),"cwd":process.cwd(),"device":"cpu","status":"RUNNING","stage":"raw_probabilities"})
        from ragscope.experiments.legacy_reconstruction import _sample_resources
        stop=threading.Event()
        monitor=threading.Thread(target=_sample_resources,args=(directory/"resource_samples.jsonl",stop),daemon=True)
        monitor.start()
        task_id="laya.raw_fixed360_three_orders"
        try:
            import torch
            if torch.version.cuda is not None:raise ValueError("Use the isolated CPU-only Torch environment")
            torch.set_num_threads(4)
            tick=time.perf_counter()
            load_path=directory/"LOAD_STATE.json"
            load=read(load_path) if load_path.exists() else {"attempts":0,"backend_failures":0}
            if load["backend_failures"]>=3:raise RuntimeError("Three Laya loading failures already recorded")
            load.update(status="RUNNING",attempts=load["attempts"]+1,pid=os.getpid(),started_at=time.time())
            atomic_json(load_path,load)
            try:
                router=LayaRouter.load_local(model,expected_sha256=hashes,min_confidence=.5)
            except Exception as error:
                if not isinstance(error,MemoryError):load["backend_failures"]+=1
                load.update(status="PENDING_RESOURCE" if isinstance(error,MemoryError) else "FAILED",
                            error={"type":type(error).__name__,"message":str(error)})
                atomic_json(load_path,load)
                raise
            load.update(status="LOADED",load_seconds=time.perf_counter()-tick)
            atomic_json(load_path,load)
            if psutil.virtual_memory().available<1280*1024**2:raise MemoryError("Laya reserve after model load")
            if supervision is not None:supervision.wait("LOADED_WAITING_PERMISSION","WARMUP")
            if str(router.agent.device)!="cpu":raise ValueError("Laya device differs from declared CPU")
            router.identity.update(torch_version=torch.__version__,cpu_threads=4,
                                   dtype=str(next(router.agent.model.parameters()).dtype),compile=False,backend="eager")
            config={"input_hash":inputs["input_hash"],"router_identity":router.identity,"expected_decisions":1080,
                    "model_manifest":artifact(model/"model_manifest.json"),"warmup_queries":3,
                    "source_sha256":file_hash(Path(__file__))}
            store=RecordStore(directory/"records.jsonl",config,real_only=True,max_records=1080)
            if not ledger.begin(task_id,config,[sys.executable,*sys.argv]):return read(directory/"SUMMARY.json")
            warm=RecordStore(directory/"warmups.jsonl",{**config,"purpose":"excluded_warmup"},real_only=True,max_records=3)
            for query in inputs["populations"]["dev"]["visible_queries"][:3]:
                if query["query_id"] in warm.ids:continue
                if supervision is not None:supervision.checkpoint("WARMUP_RUNNING")
                if psutil.virtual_memory().available<1280*1024**2:raise MemoryError("Laya reserve before warmup")
                decision=router.choose(RoutingRequest.from_query(query))
                warm.add(query["query_id"],{"real_model":True,"model_identity":router.identity,"decision":decision.to_dict()})
                if supervision is not None:supervision.checkpoint("WARMUP_COMMITTED")
                if psutil.virtual_memory().available<1280*1024**2:raise MemoryError("Laya reserve after committed warmup")
            if any(row["decision"]["metadata"].get("error") for row in warm.records):
                raise RuntimeError("Real Laya dev preflight failed; raw failures retained")
            if supervision is not None:
                supervision.wait("PREFLIGHT_PASSED","CONTINUE",warmup_count=len(warm.records),warmup_errors=0)
            failures=0
            for previous in reversed(store.records):
                if not previous["decision"]["metadata"].get("error"):break
                failures+=1
            for split in ("train","dev","test"):
                for query in inputs["populations"][split]["visible_queries"]:
                    for index,order in enumerate(PERMUTATIONS):
                        record_id=f"{split}:{query['query_id']}:{index}"
                        if record_id in store.ids:continue
                        if supervision is not None:supervision.checkpoint("FORMAL")
                        if failures>=3:raise RuntimeError("Three consecutive stored Laya failures require diagnosis")
                        if psutil.virtual_memory().available<1280*1024**2:raise MemoryError("Laya runtime RAM reserve")
                        request=RoutingRequest.from_query(query)
                        decision=router.choose(request,order=order).to_dict()
                        store.add(record_id,{"query_id":query["query_id"],"split":split,"order_index":index,"order":list(order),
                            "visible_input_hash":signature(asdict(request)),"decision":decision,"real_model":True,
                            "model_identity":router.identity,"actual_process_rss":process.memory_info().rss})
                        if supervision is not None:supervision.checkpoint("FORMAL_COMMITTED")
                        failures=failures+1 if decision["metadata"].get("error") else 0
                    if len(store.ids)%30==0:print(json.dumps({"laya_decisions":len(store.ids),"expected":1080}),flush=True)
            summary={"status":"COMPLETE","n_decisions":len(store.ids),"n_queries":360,"input_hash":inputs["input_hash"],
                     "router_identity":router.identity,"elapsed_including_load_s":time.perf_counter()-tick,
                     "errors":sum(bool(row["decision"]["metadata"].get("error")) for row in store.records),
                     "fallbacks":sum(row["decision"]["fallback"] for row in store.records),
                     "quality_labels_read":False,"quality_scoring":"PENDING_REAL_ROUTE_LABELS",
                     "artifacts":[artifact(store.path),artifact(store.path.with_suffix('.config.json')),artifact(warm.path)]}
            atomic_json(directory/"SUMMARY.json",summary)
            ledger.finish(task_id,"PASS",[directory/"SUMMARY.json",store.path,store.path.with_suffix('.config.json'),warm.path],summary)
            return summary
        except MemoryError as error:
            if task_id in ledger.state["tasks"]:
                task=ledger.state["tasks"][task_id];task.update(status="PENDING",attempts=max(0,task["attempts"]-1))
            ledger.event("laya_resource_pause",reason=str(error))
            raise
        finally:
            stop.set();monitor.join(timeout=2)
            worker=read(directory/"worker.json");worker["status"]="EXITING";atomic_json(directory/"worker.json",worker)
            if supervision is not None:
                supervision.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home",default="D:/RAGScope_Runtime")
    parser.add_argument("--campaign",default="20261014_v2")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--supervised-worker",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.supervised_worker and not args.run:raise ValueError("Supervised worker requires actual run mode")
    if args.run:
        from .laya_supervision import ChildControl, conditional_run
        if args.supervised_worker:
            control=ChildControl(args.supervised_worker,args.home,args.campaign)
            result=run_raw(args.home,args.campaign,supervision=control)
        else:result=conditional_run(args.home,args.campaign)
    else:result=freeze_visible_inputs(args.home,args.campaign)
    print(json.dumps({k:v for k,v in result.items() if k not in {'populations','sources','artifacts'}},ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
