"""RAGScope v2 local Streamlit workbench. Importing never loads a model.

Run: python -m streamlit run ui/chat_app.py --server.address 127.0.0.1
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def read_ui_config(path=None):
    path = Path(path or os.environ.get("RAGSCOPE_UI_CONFIG", _ROOT / "configs/v2/ui.json"))
    if not path.exists():
        return {"models": [], "indexes": [], "result_files": [], "status": "未配置就绪资源"}
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    for name in ("models", "indexes", "result_files", "routers"):
        if not isinstance(config.get(name, []), list):
            raise ValueError(f"UI config {name} must be a list")
    return config


def ready_options(items):
    return [item for item in items if item.get("ready") is True]


def validate_index_entry(entry):
    from ragscope.indexing.identity import signature, verify_artifacts
    path = Path(entry["manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest["signature"] != signature(manifest["identity"]):
        raise ValueError("索引身份签名不一致")
    if not manifest.get("artifacts") or not verify_artifacts(manifest["artifacts"]):
        raise ValueError("索引产物缺失或校验和不一致")
    return manifest


def load_index(entry):
    from ragscope.indexing.array_index import ArrayIndex
    manifest = validate_index_entry(entry)
    return ArrayIndex.load(Path(entry["manifest_path"]).parent, manifest["identity"])


def context_options(entry, index, strategy, budget):
    options = {"max_tokens": budget, "expansion": strategy,
               "chunk_lookup": {chunk["chunk_id"]: chunk for chunk in index.chunks}}
    if strategy == "parent":
        from ragscope.indexing.identity import signature, verify_artifacts
        manifest = json.loads(Path(entry["chunk_manifest_path"]).read_text(encoding="utf-8-sig"))
        if manifest["signature"] != signature(manifest["identity"]) or not verify_artifacts(manifest["artifacts"]):
            raise ValueError("父块来源 manifest 验证失败")
        parent_artifacts = [item for item in manifest["artifacts"] if Path(item["path"]).name == "parents.jsonl"]
        if len(parent_artifacts) != 1:
            raise ValueError("父块来源不明确")
        parents = [json.loads(line) for line in Path(parent_artifacts[0]["path"]).read_text(encoding="utf-8").splitlines() if line]
        options["parent_lookup"] = {parent["chunk_id"]: parent for parent in parents}
    return options


def load_result_file(path):
    """Read only immutable, hash-verified RecordStore files; never repair in UI."""
    from ragscope.experiments.ledger import read_records
    from ragscope.indexing.identity import signature
    path = Path(path)
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("记录超过浏览器单文件 128 MiB 预算，请使用已导出的汇总文件")
    config_path = path.with_suffix(".config.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    expected = signature(config)
    rows = read_records(path, repair_tail=False)
    ids = set()
    for row in rows:
        if not row.get("record_hash") or row.get("config_hash") != expected:
            raise ValueError("记录缺少校验和或配置身份不一致")
        if row.get("record_id") in ids:
            raise ValueError("记录 ID 重复")
        ids.add(row.get("record_id"))
    return config, rows


def probe_model(entry):
    """Metadata-only readiness check: no generate/pull/start requests."""
    from ragscope.generation.llm_client import _local_url, _NoRedirect
    if entry.get("ready") is not True:
        return False, entry.get("reason", "未完成真实模型预检")
    identity = entry.get("identity", {})
    if not all(identity.get(key) for key in ("revision", "weight_sha256", "backend_version")):
        return False, "模型身份不完整"
    base = _local_url(entry["base_url"])
    route = "/api/tags" if entry["provider"] == "ollama" else "/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(base+route, timeout=2) as response:
            payload = json.load(response)
        models = payload.get("models", []) if entry["provider"] == "ollama" else payload.get("data", [])
        if entry['provider'] == 'ollama':
            from ui.model_identity import tag_matches
            return (True, '服务 tag 与冻结模型 manifest 一致') if tag_matches(entry, models) else (False, '同名模型的 digest 已改变或模型未就绪')
        names = {model.get("name", model.get("id")) for model in models}
        return (True, "本地模型已就绪") if entry["model"] in names else (False, "所选模型未在该服务中就绪")
    except Exception as exc:
        return False, f"本地服务不可用：{type(exc).__name__}"


def make_client(entry):
    from ragscope.generation.llm_client import LocalLLMClient
    identity = entry["identity"]
    options = {key: identity[key] for key in ("revision", "weight_sha256", "backend_version", "quantization", "chat_template_hash", "backend_runtime_config") if key in identity}
    options.update({key: entry[key] for key in ("num_ctx", "num_gpu", "service_device_verified", "think", "timeout") if key in entry})
    client = LocalLLMClient(entry["model"], provider=entry["provider"], base_url=entry["base_url"], **options)
    if entry["provider"] == "llama.cpp":
        def count_prompt(messages):
            rendered = client._request("/apply-template", {"messages": messages,
                "chat_template_kwargs": {"enable_thinking": client.think}})
            if not isinstance(rendered.get("prompt"), str):
                raise ValueError("后端未返回真实 chat template")
            return len(client._request("/tokenize", {"content": rendered["prompt"], "add_special": True})["tokens"])
        client.prompt_token_counter = count_prompt
    elif entry.get('counter_kind') in {'qwen35','lfm25'}:
        from ragscope.generation.token_budget import Qwen35PromptCounter, LFMPromptCounter
        cls = Qwen35PromptCounter if entry['counter_kind']=='qwen35' else LFMPromptCounter
        counter = cls(entry['tokenizer_path'],entry['renderer_source'])
        if counter.template_hash != identity.get('chat_template_hash'):
            raise ValueError('已核验 tokenizer/模板身份发生变化')
        client.prompt_token_counter = counter
    elif entry.get("tokenizer_path"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(entry["tokenizer_path"], local_files_only=True, trust_remote_code=False)
        client.prompt_token_counter = lambda messages: len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=client.think, return_dict=False))
    else:
        raise ValueError("Ollama 需要已落盘的真实 tokenizer 路径以校验完整输入预算")
    return client


class ScopedRetriever:
    def __init__(self, index, entry, scope):
        self.index, self.entry, self.scope = index, entry, scope
        self.identity = {"index": index.identity, "method": entry.get("method", "bm25"), "scope": scope}
        self.last_timings = {}

    def retrieve(self, question, k=10):
        method = self.entry.get("method", "bm25").lower()
        start, query_vector, embedding_identity = time.perf_counter(), None, None
        if method != "bm25":
            from ragscope.indexing.local_embedding import HFJsonTokenizer, LocalEmbeddingService
            config = self.entry["embedding"]
            if not config.get("ready") or config.get("service_device_verified") != "cpu":
                raise ValueError("查询 embedding 必须是已核验的 CPU 本地服务")
            tokenizer = HFJsonTokenizer(config["tokenizer_path"], config["identity"]["model"], config["identity"]["revision"])
            service = LocalEmbeddingService(config["base_url"], config["identity"], tokenizer,
                                            Path(config["cache_dir"]), config.get("max_input_tokens", 1100))
            query_vector, embedding_identity = service.embed_query(question), service.identity
        self.last_timings["query_embedding_ms"] = (time.perf_counter()-start)*1000
        ranking_start = time.perf_counter()
        matches = self.index.search(question, method="hybrid" if method == "hybrid+rerank" else method,
            query_vector=query_vector, embedding_identity=embedding_identity, scope=self.scope,
            k=50 if method == "hybrid+rerank" else k, candidate_depth=50)
        self.last_timings["ranking_ms"] = (time.perf_counter()-ranking_start)*1000
        if method == "hybrid+rerank":
            from ragscope.retrieval.verified_reranker import load_verified_reranker
            from ragscope.experiments.runtime import require_ram
            config = self.entry["reranker"]
            if not config.get("ready"):
                raise ValueError("重排模型未就绪")
            # CPU weight/framework estimate plus a separate host reserve. A UI
            # click must not blindly load a second model into exhausted RAM.
            require_ram(1536*1024**2, reserve_bytes=1280*1024**2)
            load_start = time.perf_counter()
            from ragscope.experiments.runtime import runtime_home
            reranker = load_verified_reranker(runtime_home(), device='cpu')
            if reranker.identity != config['identity']:
                raise ValueError('重排模型与已登记的实际版本/运行配置不一致')
            self.last_timings["reranker_load_ms"] = (time.perf_counter()-load_start)*1000
            require_ram(0, reserve_bytes=1280*1024**2)
            rerank_start = time.perf_counter()
            matches = reranker.rerank(question, matches, top_k=k)
            self.last_timings["rerank_ms"] = (time.perf_counter()-rerank_start)*1000
        return matches


def result_view(st, record):
    result = record.get("result", record)
    if record.get('routing'):
        routing = record['routing']
        st.caption(f"实际路由：{routing['policy_id']} → {routing['decision']['route']}")
        with st.expander('路由依据与耗时'):
            st.json(routing)
    if result.get("error"):
        st.error(f"执行错误：{result['error']}")
    elif result.get("abstain"):
        st.warning("模型拒答：现有上下文不足以支持答案。")
        st.write(result.get("answer", ""))
    elif "answer" in result:
        st.markdown(result["answer"])
    if (result.get("real_model") is False and result.get("experiment_type") != "retrieval"
            and any(key in result for key in ("answer", "raw_output"))):
        st.warning("此记录使用 Mock，仅用于软件功能测试，不代表真实问答质量。")
    if result.get("invalid_citations"):
        st.error("不可见引用：" + ", ".join(result["invalid_citations"]))
    bundle = result.get("context_bundle", record.get("context_bundle", {}))
    citations = set(result.get("citations", []))
    st.caption("下面展示已打包的上下文片段；原始候选、预算与未纳入片段见下方记录。")
    for position, piece in enumerate(bundle.get("pieces", []), 1):
        label = f"{position}. {piece['chunk_id']}" + (" · 被引用" if piece["chunk_id"] in citations else "")
        with st.expander(label):
            st.caption(f"论文：{piece.get('doc_id', '')} · 父块：{piece.get('parent_id') or '无'}")
            st.text(piece.get("text", ""))
            st.json(piece.get("unit_spans", []))
    with st.expander("检索结果、预算与去重记录"):
        st.json({"retrieved_chunks": result.get("retrieved_chunks", result.get("raw_retrieval", [])),
                 "budget": bundle.get("budget", {}), "deduplication": bundle.get("deduplication", []),
                 "truncations": bundle.get("truncations", []), "context_hash": bundle.get("context_hash")})
    timing = result.get("timings", {})
    columns = st.columns(3)
    for column, key, label in zip(columns, ("retrieval_ms", "generation_ms", "total_ms"), ("检索毫秒", "生成毫秒", "总毫秒")):
        column.metric(label, f"{timing[key]:.1f}" if isinstance(timing.get(key), (int, float)) else "N/A")
    with st.expander("模型身份、资源、评分与原始输出"):
        st.json({key: result.get(key) for key in ("model_identity", "backend_metadata", "resources", "scores", "timings", "prompt_hash", "cache_hit")})
        st.code(result.get("raw_output", ""), language=None)
    st.caption("非流式请求没有首 token 实测值；引用可见率不等于答案支持性。")


def main():
    import streamlit as st
    st.set_page_config(page_title="RAGScope v2", page_icon="🔎", layout="wide")
    try:
        config = read_ui_config()
    except Exception as exc:
        st.error(f"界面配置无法读取：{exc}")
        return
    st.title("RAGScope v2 · 技术资料问答与实验台")
    st.caption("英文公开文本实验 · 中文操作说明 · 本地运行")
    tabs = st.tabs(["资料问答", "实验记录", "资源状态"])
    indexes, models = ready_options(config.get("indexes", [])), ready_options(config.get("models", []))
    with st.sidebar:
        st.subheader("运行状态")
        st.write(config.get("status", "使用已冻结、已验证资源"))
        st.caption(f"就绪检索配置 {len(indexes)} · 就绪生成模型 {len(models)}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            st.info("本进程 CPU-only；生成服务的设备状态需单独核验。")
        st.caption("页面不会自动下载或启动模型。")
    with tabs[0]:
        selected_index = None
        if indexes:
            ids = [item["id"] for item in indexes]
            default = ids.index(config["default_retriever"]) if config.get("default_retriever") in ids else 0
            chosen = st.selectbox("检索配置", ids, index=default)
            selected_index = indexes[ids.index(chosen)]
        else:
            st.info("没有已验证的就绪索引；仍可查看实验记录与资源状态。")
        choices = ["仅检索证据"] + [item["id"] for item in models]
        default = choices.index(config["default_model"]) if config.get("default_model") in choices else 0
        model_id = st.selectbox("生成模型", choices, index=default)
        if not models:
            st.info("真实生成模型尚未就绪；不会自动切换到 Mock。")
        scope = st.text_input("论文范围（doc_id；留空为全库检索）", value=config.get("default_scope", ""))
        routers = [item for item in ready_options(config.get('routers', []))
                   if selected_index and item.get('index_id') == selected_index['id']]
        selected_router = None
        if routers:
            from ui.routing_options import deployable_default
            route_ids = ['固定当前检索配置'] + [item['id'] for item in routers]
            default_policy = next((item for item in routers if item['id'] == config.get('default_router')), None)
            current_model = next((item for item in models if item['id'] == model_id), None)
            can_default = bool(default_policy and deployable_default(default_policy,
                model_identity=(current_model or {}).get('identity'),
                index_manifest=validate_index_entry(selected_index),
                runtime_config={'execution_profile': 'cold_application_loads'}))
            route_default = route_ids.index(default_policy['id']) if can_default else 0
            route_id = st.selectbox('路由策略', route_ids, index=route_default)
            selected_router = next((item for item in routers if item['id'] == route_id), None)
            if selected_router:
                st.caption('实验路由使用已冻结的上下文策略和最多 10 个片段，选择过程与加载时间均计入实际耗时。')
        strategies = ["none", "neighbor"] + (["parent"] if selected_index and selected_index.get("parent_ready") else [])
        strategy_default = (selected_router or selected_index or {}).get('expansion', 'none')
        strategy = st.selectbox("上下文策略", strategies,
            index=strategies.index(strategy_default) if strategy_default in strategies else 0,
            disabled=selected_router is not None,
            format_func=lambda value: {"none": "直接使用片段", "neighbor": "补相邻片段", "parent": "扩展父块"}[value])
        k = st.slider("最多返回片段数", 1, 10, 10, disabled=selected_router is not None)
        if selected_router:
            k = 10
        question = st.text_input("问题", placeholder="输入关于已选论文的问题")
        if st.button("运行", type="primary", disabled=selected_index is None):
            if not question.strip():
                st.warning("请输入问题。")
            else:
                try:
                    from ragscope.generation.context import pack_context
                    started = time.perf_counter()
                    index = load_index(selected_index)
                    routing, active_index = None, selected_index
                    if selected_router:
                        from ui.routing_options import choose_route
                        active_index, routing = choose_route(selected_router, question, scope or None, selected_index, index.identity)
                    retriever = ScopedRetriever(index, active_index, scope or None)
                    packing = context_options(selected_index, index, strategy, config.get("reference_budget", 2048))
                    if scope and scope not in index.by_doc:
                        raise ValueError("该论文不在所选索引中")
                    if model_id == "仅检索证据":
                        chunks = retriever.retrieve(question, k)
                        bundle = pack_context(chunks, scope=scope or None, **packing)
                        record = {"question": question, "retrieved_chunks": chunks, "context_bundle": asdict(bundle),
                                  "timings": {"retrieval_ms": (time.perf_counter()-started)*1000,
                                              "total_ms": (time.perf_counter()-started)*1000, **retriever.last_timings}}
                    else:
                        from ui.live_generation import answer_live
                        entry = next(model for model in models if model["id"] == model_id)
                        ready, reason = probe_model(entry)
                        if not ready:
                            raise ValueError(reason)
                        client = make_client(entry)
                        record = answer_live(client, retriever, packing, question, k, scope or None, config['execution'], model_entry=entry)
                    if routing:
                        record['routing'] = routing
                        record.setdefault('timings', {}).update(routing['timings'])
                    # This application timing includes source verification, index
                    # and router/model-client loading, unlike pipeline-only time.
                    record.setdefault('timings', {})['application_total_ms'] = (time.perf_counter()-started)*1000
                    record['timings']['pipeline_total_ms'] = record['timings'].get('total_ms')
                    record['timings']['total_ms'] = record['timings']['application_total_ms']
                    if model_id != '仅检索证据':
                        from ui.live_generation import save_live_record
                        record['saved_record_path'] = save_live_record(record, config['execution'], supplement=True)
                    st.session_state["live_result"] = record
                except Exception as exc:
                    st.error(f"本次执行未完成：{exc}")
        if "live_result" in st.session_state:
            result_view(st, st.session_state["live_result"])
    with tabs[1]:
        files = config.get("result_files", [])
        if not files:
            st.info("尚无已登记实验记录。这里不会将示例数据当作真实实验。")
        else:
            selected = st.selectbox("运行记录文件", files)
            if st.button("加载并校验记录"):
                try:
                    _, rows = load_result_file(selected)
                    st.session_state["view_rows"] = rows
                    st.session_state["view_path"] = selected
                except Exception as exc:
                    st.error(f"记录验证失败：{exc}")
            if st.session_state.get("view_path") == selected:
                rows = st.session_state.get("view_rows", [])
                st.caption(f"已验证 {len(rows)} 条记录 · 此处是历史回放")
                if rows:
                    row_index = st.number_input("记录序号", min_value=1, max_value=len(rows), value=1)
                    row = rows[int(row_index)-1]
                    st.write(row.get("question", row.get("query_id", row.get("record_id", ""))))
                    result_view(st, row)
                    with st.expander("完整原始记录"):
                        st.json(row)
    with tabs[2]:
        for label in ("indexes", "models", "routers"):
            st.subheader({'indexes': '索引配置', 'models': '模型配置', 'routers': '实验路由'}[label])
            entries = config.get(label, [])
            if not entries:
                st.write("未登记")
            for entry in entries:
                st.write(f"{entry['id']}：{'已登记就绪' if entry.get('ready') else '未安装 / 未评测 / 未就绪'}")
                st.caption(entry.get("reason", ""))
                with st.expander(f"{entry['id']} · 版本与设备"):
                    st.json(entry)
        st.caption("已登记就绪仍会在运行时检查文件校验和、模型服务和输入预算；CPU offload 只按实际记录展示。")


if __name__ == "__main__":
    main()
