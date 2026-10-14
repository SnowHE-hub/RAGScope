"""Single-GPU BF16 LoRA, bounded steps, explicit lease and complete recovery.

Importing this module and --mode guard use no torch/model imports or GPU memory.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

from ragscope.experiments.ledger import RecordStore, append_jsonl
from ragscope.experiments.runtime import runtime_home, require_ram, require_disk, validate_gpu_lease
from ragscope.indexing.identity import atomic_json, file_hash, signature, verify_artifacts
from ragscope.training.checkpoint import CheckpointStore


def validate_config(config):
    exact = {"r": 8, "alpha": 16, "dropout": .05, "batch_size": 1,
             "gradient_accumulation": 8, "learning_rate": 1e-4, "warmup_ratio": .03,
             "dtype": "bfloat16", "gradient_checkpointing": True}
    for key, value in exact.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen LoRA setting differs: {key}")
    if config.get("max_seq_len") not in {512, 1024} or not 1 <= config.get("max_steps", 0) <= 200:
        raise ValueError("LoRA token or step bound exceeded")
    if config.get("max_epochs") != 1 or config.get("preflight_steps") != 10 or config.get("max_checkpoints") != 3:
        raise ValueError("LoRA epoch/preflight/retention bound changed")
    if len(config.get("paths", [])) > 2:
        raise ValueError("At most two actual training paths are permitted")
    return config


def verify_lease(path):
    lease = validate_gpu_lease(path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise ValueError("CPU-only process cannot begin training")
    return lease


def resource_guard(config, path_config, report_path):
    """Record measured resource rejection without pretending a model OOM ran."""
    validate_config(config)
    report = {"model": path_config["model"], "revision": path_config["revision"],
              "max_seq_len": config["max_seq_len"], "config_hash": signature(config),
              "status": "PENDING", "gpu_allocated": False, "weights_loaded": False,
              "torch_imported": False, "time_unix": time.time()}
    try:
        report["disk"] = require_disk(path_config.get("estimated_checkpoint_bytes", 256*1024**2))
        report["ram"] = require_ram(path_config.get("minimum_host_bytes", 1024**3))
        report["status"] = "PASS_RESOURCE_GUARD_ONLY"
    except RuntimeError as exc:
        report.update(status="BLOCKED_RESOURCE", error=str(exc), failure_type="preload_resource_guard")
    atomic_json(report_path, report)
    return report


def discover_attention_targets(model):
    """Inspect actual module classes/ancestry; no guessed projection-name list."""
    modules = dict(model.named_modules())
    targets = []
    for name, module in modules.items():
        lowered = name.lower()
        if any(part in lowered for part in ("vision", "visual", "image", "mlp", "lm_head", "embed")):
            continue
        if type(module).__name__ != "Linear" or not hasattr(module, "weight") or getattr(module.weight, "ndim", None) != 2:
            continue
        ancestry = [".".join(name.split(".")[:i]) for i in range(1, len(name.split(".")))]
        classes = [type(modules[parent]).__name__.lower() for parent in ancestry if parent in modules]
        if any("attention" in cls or "gateddeltanet" in cls for cls in classes):
            targets.append(name)
    if not targets:
        raise ValueError("No verified language attention Linear modules; do not guess target_modules")
    return sorted(targets)


def verify_training_snapshot(model_directory, path_config):
    """Reject incomplete receipts before importing any GPU-capable library."""
    model_directory = Path(model_directory)
    receipt = json.loads((model_directory / "model_manifest.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "VERIFIED":
        raise ValueError("Training snapshot receipt is not complete and VERIFIED")
    if receipt.get("model") != path_config["model"] or receipt.get("revision") != path_config["revision"]:
        raise ValueError("Pinned model weight receipt differs")
    if any(not Path(item["path"]).resolve().is_relative_to(model_directory.resolve()) for item in receipt.get("artifacts", [])):
        raise ValueError("Model receipt points outside the pinned local snapshot")
    if not receipt.get("artifacts") or not verify_artifacts(receipt["artifacts"]):
        raise ValueError("Pinned local model artifacts failed checksum verification")
    names = {str(Path(item['path']).resolve().relative_to(model_directory.resolve())).replace('\\','/') for item in receipt['artifacts']}
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json'}
    index_path = model_directory / 'model.safetensors.index.json'
    if index_path.exists():
        required.add('model.safetensors.index.json')
        mapping = json.loads(index_path.read_text(encoding='utf-8')).get('weight_map', {})
        if not mapping:
            raise ValueError('Empty weight shard index')
        required.update(mapping.values())
    else:
        required.add('model.safetensors')
    if not required <= names:
        raise ValueError('Receipt omits tokenizer/config or required weight shards')
    local_config = model_directory / "config.json"
    if file_hash(local_config) != path_config["config_sha256"]:
        raise ValueError("Training model config differs from pinned official revision")
    configuration = json.loads(local_config.read_text(encoding="utf-8"))
    architecture = configuration["architectures"][0]
    if architecture != "Qwen3_5ForConditionalGeneration":
        raise ValueError("Unverified model architecture for the frozen two-path training plan")
    return architecture


def load_training_model(model_directory, path_config, config):
    architecture = verify_training_snapshot(model_directory, path_config)
    import torch
    import transformers
    import numpy as np
    from peft import LoraConfig, get_peft_model
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    if int(transformers.__version__.split(".")[0]) < 5:
        raise RuntimeError("Qwen3.5 training requires a verified Transformers v5 environment")
    model_directory = Path(model_directory)
    loader = getattr(transformers, architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_directory, local_files_only=True, trust_remote_code=False)
    model = loader.from_pretrained(model_directory, dtype=torch.bfloat16, device_map={"": "cuda:0"},
                                   attn_implementation="eager", local_files_only=True, trust_remote_code=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = discover_attention_targets(model)
    model = get_peft_model(model, LoraConfig(r=config["r"], lora_alpha=config["alpha"],
        lora_dropout=config["dropout"], target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    if hasattr(model.config,'text_config'):
        model.config.text_config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name or any(term in name.lower() for term in ("vision", "visual")) for name in trainable):
        raise ValueError("Unexpected trainable non-language-adapter parameters")
    initial_hash=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            initial_hash.update(name.encode('utf-8'))
            initial_hash.update(str(tuple(parameter.shape)).encode('ascii'))
            initial_hash.update(parameter.detach().float().cpu().numpy().tobytes())
    return model, tokenizer, {"targets": targets, "trainable_names": trainable,
                              'initialization_seed':config['seed'],'initialization_policy':'before_model_and_adapter_v1',
                              'initial_trainable_state_sha256':initial_hash.hexdigest(),
                              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                              "torch": torch.__version__, "transformers": transformers.__version__,
                              "peft": importlib.metadata.version("peft")}


def _rng_state(torch):
    import numpy as np
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def _restore_rng(torch, state):
    import numpy as np
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))


def _state_equal(torch, left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left.detach().cpu(),right.detach().cpu())
    if isinstance(left,dict) and isinstance(right,dict):
        return set(left)==set(right) and all(_state_equal(torch,left[key],right[key]) for key in left)
    if isinstance(left,(tuple,list)) and isinstance(right,(tuple,list)):
        return len(left)==len(right) and all(_state_equal(torch,a,b) for a,b in zip(left,right))
    return left==right


def train_loop(model, tokenizer, examples, config, run_dir, *, resume=False, stop_after=None, dev_examples=None,
               empty_cache_after_microbatch=False):
    """Save at optimizer-step boundaries; pending microbatches replay on restart."""
    import torch
    import psutil
    from peft import set_peft_model_state_dict, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_linear_schedule_with_warmup
    if not examples:
        raise ValueError("No complete train-only examples fit the frozen token budget")
    if len(examples) > 1000 or any(len(row["input_ids"]) > config["max_seq_len"] for row in examples):
        raise ValueError("Prepared training examples exceed frozen bounds")
    if any(not any(label != -100 for label in row["labels"]) for row in examples):
        raise ValueError("Prepared training example has no assistant labels")
    validate_config(config)
    runtime_policy={'allocator_cache':'empty_unused_after_each_microbatch_and_dev_example' if empty_cache_after_microbatch else 'pytorch_default_cache'}
    runtime_policy_hash=signature(runtime_policy)
    run_dir = Path(run_dir)
    immutable = {"config": config, "query_ids": [row["query_id"] for row in examples],
                 "data_hash": signature(examples), "dev_hash": signature(dev_examples or [])}
    store = CheckpointStore(run_dir / "checkpoints", immutable)
    metrics_store = RecordStore(run_dir / "steps.jsonl", immutable, max_records=200)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    order = list(range(len(examples)))
    random.Random(config["seed"]).shuffle(order)
    accumulation = config["gradient_accumulation"]
    max_steps = min(config["max_steps"], math.ceil(len(order)/accumulation))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(max_steps*config["warmup_ratio"]), max_steps)
    progress = {"step": 0, "epoch": 0, "data_position": 0, "data_order": order,
                "best_dev_loss": None, "seed": config["seed"], "max_steps": max_steps}
    if resume:
        checkpoint = store.latest()
        if checkpoint is None:
            raise ValueError("No complete checkpoint to resume")
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        adapter_state=load_file(str(checkpoint / "adapter/adapter_model.safetensors"))
        expected_adapter=get_peft_model_state_dict(model)
        if set(adapter_state)!=set(expected_adapter) or any(adapter_state[key].shape!=expected_adapter[key].shape for key in adapter_state):
            raise ValueError('Checkpoint adapter keys/shapes differ from instantiated LoRA modules')
        set_peft_model_state_dict(model,adapter_state)
        restored_adapter=get_peft_model_state_dict(model)
        if any(not torch.equal(value.detach().cpu(),adapter_state[key]) for key,value in restored_adapter.items()):
            raise ValueError('Restored adapter tensors do not exactly match checkpoint')
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        progress = json.loads((checkpoint / "progress.json").read_text(encoding="utf-8"))
        _restore_rng(torch, state["rng"])
        verified={"adapter_exact":True,
            "optimizer_exact":_state_equal(torch,optimizer.state_dict(),state['optimizer']),
            "scheduler_exact":_state_equal(torch,scheduler.state_dict(),state['scheduler']),
            "rng_exact":_state_equal(torch,_rng_state(torch),state['rng']),
            "data_order_exact":progress['data_order']==order,
            "data_progress_consistent":progress['data_position']==min(progress['step']*accumulation,len(order))}
        if not all(verified.values()):
            raise ValueError('Training state failed exact restoration verification: '+str(verified))
        atomic_json(run_dir/f"resume_from_step_{progress['step']:06d}.json",{
            'status':'RESTORED_VERIFIED','checkpoint':str(checkpoint),'manifest_sha256':file_hash(checkpoint/'manifest.json'),
            'step':progress['step'],'data_position':progress['data_position'],'verification':verified,
            'runtime_policy':runtime_policy,'runtime_policy_hash':runtime_policy_hash,
            'checkpoint_artifacts':json.loads((checkpoint/'manifest.json').read_text(encoding='utf-8'))['artifacts']})
        if progress.get("last_metric"):
            metrics_store.add(str(progress["step"]), progress["last_metric"])
    elif store.latest() is not None:
        raise ValueError("Existing training run requires explicit resume")
    stop_requested = [False]
    handlers = {}
    def request_stop(signum, frame):
        stop_requested[0] = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        handlers[signum] = signal.signal(signum, request_stop)
    def write_checkpoint(directory, state):
        model.save_pretrained(directory / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(directory / "tokenizer")
        torch.save({key: state[key] for key in ("optimizer", "scheduler", "rng")}, directory / "training_state.pt")
        atomic_json(directory / "progress.json", progress)
    def save(best=False):
        state = {**progress, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": _rng_state(torch)}
        return store.save(progress["step"], state, write_checkpoint, best=best)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    process=psutil.Process(os.getpid())
    peak_sampled_rss=0
    min_sampled_host_available=None
    try:
        while progress["step"] < max_steps and progress["data_position"] < len(order):
            start, losses = time.perf_counter(), []
            positions = progress["data_order"][progress["data_position"]:progress["data_position"]+accumulation]
            for index in positions:
                row = examples[index]
                batch = {key: torch.tensor([row[key]], device="cuda:0") for key in ("input_ids", "attention_mask", "labels")}
                loss = model(**batch).loss
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite loss")
                (loss/len(positions)).backward()
                losses.append(float(loss.detach().cpu()))
                peak_sampled_rss=max(peak_sampled_rss,process.memory_info().rss)
                available=psutil.virtual_memory().available
                min_sampled_host_available=available if min_sampled_host_available is None else min(min_sampled_host_available,available)
                if empty_cache_after_microbatch:
                    del loss,batch
                    torch.cuda.empty_cache()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
                raise FloatingPointError("Missing or non-finite adapter gradients")
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress["step"] += 1
            progress["data_position"] += len(positions)
            metric = {"step": progress["step"], "loss": sum(losses)/len(losses), "gradient_norm": grad_norm,
                      "data_position": progress["data_position"], "seconds": time.perf_counter()-start,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                      "peak_sampled_process_rss_bytes":peak_sampled_rss,
                      "minimum_sampled_host_available_bytes":min_sampled_host_available,
                      "runtime_policy":runtime_policy,"runtime_policy_hash":runtime_policy_hash,
                      "host_sampling":"after each microbatch; can miss transient peaks", "dtype": "bfloat16"}
            best = False
            if dev_examples and (progress["step"] % 25 == 0 or progress["step"] == max_steps):
                model.eval()
                with torch.no_grad():
                    values=[]
                    for row in dev_examples:
                        value=float(model(**{key:torch.tensor([row[key]],device='cuda:0') for key in
                            ('input_ids','attention_mask','labels')}).loss.cpu())
                        if not math.isfinite(value):raise FloatingPointError('Non-finite dev loss')
                        values.append(value)
                        if empty_cache_after_microbatch:torch.cuda.empty_cache()
                model.train()
                metric["dev_loss"] = sum(values)/len(values)
                best = progress["best_dev_loss"] is None or metric["dev_loss"] < progress["best_dev_loss"]
                if best:
                    progress["best_dev_loss"] = metric["dev_loss"]
            progress["epoch"] = 1 if progress["data_position"] >= len(order) else 0
            progress["last_metric"] = metric
            save(best=best)
            metrics_store.add(str(progress["step"]), metric)
            print(json.dumps({'event':'optimizer_step',**metric}),flush=True)
            if stop_requested[0] or (stop_after is not None and progress["step"] >= stop_after):
                break
            if psutil.virtual_memory().available<512*1024**2:
                progress['stop_reason']='low_host_ram_after_complete_optimizer_checkpoint'
                break
        progress["epoch"] = 1 if progress["data_position"] >= len(order) else 0
        return {**progress, "status": "STOPPED_RECOVERABLE" if progress["step"] < max_steps else "TRAIN_LOOP_COMPLETE",
                "latest_checkpoint": str(store.latest()), "actual_optimizer_steps": progress["step"],
                "preflight_10_steps_passed": progress["step"] >= config["preflight_steps"],
                "best_checkpoint_available": (store.root / "best.json").exists()}
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2/lora.json")
    parser.add_argument("--path-index", type=int, default=0, choices=(0, 1))
    parser.add_argument("--max-seq-len", type=int, choices=(512, 1024))
    parser.add_argument("--mode", choices=("guard", "preflight", "train"), default="guard")
    parser.add_argument("--model-dir")
    parser.add_argument("--examples")
    parser.add_argument("--dev-examples")
    parser.add_argument("--lease")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument('--empty-cache-after-microbatch',action='store_true',
        help='Free only unused CUDA allocator cache between examples; optimization and RNG state unchanged')
    parser.add_argument("--gpu-lock", default=str(runtime_home() / "runs/gpu.lock"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.max_seq_len:
        config["max_seq_len"] = args.max_seq_len
    config["selected_path_index"] = args.path_index
    validate_config(config)
    path_config = config["paths"][args.path_index]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    guard = resource_guard(config, path_config, run_dir / "resource_guard.json")
    print(json.dumps(guard, ensure_ascii=False), flush=True)
    if args.mode == "guard":
        return 0 if guard["status"] == "PASS_RESOURCE_GUARD_ONLY" else 2
    if guard["status"] != "PASS_RESOURCE_GUARD_ONLY":
        return 2
    if not args.lease or not args.examples or not args.model_dir:
        parser.error("Actual training requires --lease --examples --model-dir")
    if not args.dev_examples:
        parser.error("Both preflight and training require the same frozen --dev-examples")
    lease = verify_lease(args.lease)
    if args.mode == "train":
        preflight = run_dir / "training_result.json"
        if not args.resume or not preflight.exists() or not json.loads(preflight.read_text(encoding="utf-8")).get("preflight_10_steps_passed"):
            raise ValueError("Formal training must resume a completed ten-step preflight")
    from filelock import FileLock
    with FileLock(args.gpu_lock, timeout=0):
        model, tokenizer, observed = load_training_model(args.model_dir, path_config, config)
        atomic_json(run_dir / "model_observed.json", {**observed, "lease_device": lease["gpu_device"], "path": path_config})
        append_jsonl(run_dir/'initialization_receipts.jsonl',{
            'time_unix':time.time(),'initialization_seed':observed['initialization_seed'],
            'policy':observed['initialization_policy'],'initial_trainable_state_sha256':observed['initial_trainable_state_sha256'],
            'before_checkpoint_restore':True,'resume_requested':args.resume,
            'code_artifact':{'path':str(Path(__file__).resolve()),'sha256':file_hash(Path(__file__))}})
        examples = [json.loads(line) for line in Path(args.examples).read_text(encoding="utf-8").splitlines() if line]
        if any(row.get("split") != "train" for row in examples):
            raise ValueError("Non-train example in training input")
        dev = [json.loads(line) for line in Path(args.dev_examples).read_text(encoding="utf-8").splitlines() if line] if args.dev_examples else []
        if len(dev) > 60:
            raise ValueError("Dev checkpoint-selection pool exceeds 60")
        if dev and ({row["doc_id"] for row in dev} & {row["doc_id"] for row in examples} or any(row.get("split") != "dev" for row in dev)):
            raise ValueError("Dev leakage or invalid selection split")
        result = train_loop(model, tokenizer, examples, config, run_dir, resume=args.resume,
                            stop_after=args.stop_after or (config["preflight_steps"] if args.mode == "preflight" else None), dev_examples=dev,
                            empty_cache_after_microbatch=args.empty_cache_after_microbatch)
        atomic_json(run_dir / "training_result.json", result)
        print(json.dumps(result), flush=True)
    return 0 if result["preflight_10_steps_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
