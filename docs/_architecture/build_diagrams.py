"""Generate the documentation SVG assets; no model, server, or network access.

The six SVGs are intended to be pasted verbatim into architecture.md. They are
not fetched by the HTML renderer: the Markdown remains the published source.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
COLORS = {
    "active": ("#edf7f2", "#257455"),
    "data": ("#eff2fc", "#534ab7"),
    "experiment": ("#fff7e8", "#a86a19"),
    "history": ("#f0f0ee", "#74746b"),
    "note": ("#fff", "#a6a297"),
}


class Diagram:
    def __init__(self, name, title, description, width=1200, height=740):
        self.name, self.width, self.height = name, width, height
        prefix = f"ragscope-{name}"
        self.marker = prefix + "-arrow"
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-labelledby="{prefix}-title {prefix}-desc">',
            f'<title id="{prefix}-title">{escape(title)}</title>',
            f'<desc id="{prefix}-desc">{escape(description)}</desc>',
            '<defs><marker id="' + self.marker + '" markerWidth="9" markerHeight="9" '
            'refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#77776f"/></marker></defs>',
            '<rect width="100%" height="100%" rx="16" fill="#fdfcf9"/>',
        ]
        self.text(30, 38, title, size=24, weight=700)

    def text(self, x, y, text, size=16, color="#252522", weight=400, anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            'font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    def box(self, x, y, w, h, title, lines=(), kind="active", title_size=19, line_size=15):
        fill, stroke = COLORS[kind]
        dash = ' stroke-dasharray="7 5"' if kind in {"experiment", "history"} else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>'
        )
        self.text(x + 15, y + 29, title, title_size, stroke, 700)
        for i, line in enumerate(lines):
            self.text(x + 15, y + 55 + i * 23, line, line_size)

    def arrow(self, points, label="", label_at=None, dashed=False):
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="#77776f" stroke-width="1.8" '
            f'marker-end="url(#{self.marker})"{dash}/>'
        )
        if label and label_at:
            self.text(*label_at, label, size=14, color="#5e5e56")

    def legend(self, y):
        for x, kind, text in [(30, "active", "当前默认链路"), (300, "data", "共享数据 / 索引"),
                              (600, "experiment", "独立实验 / 非默认"), (900, "history", "历史实现 / 非默认")]:
            self.box(x, y, 250, 40, text, kind=kind, title_size=15)

    def save(self, directory):
        text = "\n".join([*self.parts, "</svg>"]) + "\n"
        ET.fromstring(text)
        path = directory / (self.name + ".svg")
        path.write_text(text, encoding="utf8")
        return path


def global_architecture():
    d = Diagram("01-global", "RAGScope：当前系统与独立研究链路", "实线为默认问答；虚线为独立或历史研究，不表示默认启用。", height=850)
    d.box(30, 80, 340, 130, "离线数据与切块", ["QASPER 长论文与历史数据", "规范文本 / section / 原文坐标", "R512 参考块与身份清单"], "data")
    d.box(430, 80, 340, 130, "完整共享索引", ["1585 篇 / 19673 个参考块", "BM25 + 1024 维 Dense", "文档级索引 / SHA 与行顺序校验"], "data")
    d.box(830, 80, 340, 130, "实验与评测资产", ["冻结划分 / 配对基线 / 失败保留", "检索、生成与真实在线记录", "独立于手工工作台交互"], "experiment")
    d.arrow([(370, 145), (430, 145)])
    d.arrow([(770, 145), (830, 145)], dashed=True)
    d.box(30, 300, 300, 130, "本地 Streamlit 工作台", ["ui/full_workbench.py", "问题 / 论文范围 / 任务", "答案 / 引用原文 / 实验历史"])
    d.box(420, 300, 350, 130, "本地 API 与 FullWorkbench", ["127.0.0.1:11437", "deployment_context + 检索", "证据预算 / schema / 解析"])
    d.box(860, 300, 310, 130, "私有本地模型后端", ["127.0.0.1:11435", "Qwen Embedding 0.6B", "Qwen 2B 量化生成 / 8K 窗口"])
    d.arrow([(330, 352), (420, 352)], "请求", (350, 338))
    d.arrow([(420, 390), (330, 390)], "答案与引用", (338, 418))
    d.arrow([(770, 352), (860, 352)], "编码 / 生成", (775, 338))
    d.arrow([(860, 390), (770, 390)])
    d.arrow([(595, 210), (595, 300)], "加载", (610, 259))
    d.box(30, 535, 350, 135, "BF16 LoRA 研究", ["独立训练、checkpoint、adapter", "真实实验与负结果均保留", "不在当前默认问答路径"], "experiment")
    d.box(425, 535, 350, 135, "路由研究", ["规则 / 线性 / MLP / LSTM / Laya", "学习选择检索动作的独立研究", "当前默认不加载学习型路由"], "experiment")
    d.box(820, 535, 350, 135, "历史 V1 / V2 实现", ["BGE / Chroma / Cross-Encoder", "旧入口、报告、索引仍保留", "不能当作当前默认配置"], "history")
    d.text(30, 728, "默认生成：2B；默认检索：Hybrid。Cross-Encoder、LoRA、学习型路由均未在默认链路启用。", 17)
    d.legend(770)
    return d


def offline_pipeline():
    d = Diagram("02-offline", "离线：从论文原文到可复核索引", "同一参考块的文本、坐标、ID、向量行号与配置哈希共同构成索引身份。", height=850)
    d.box(30, 85, 330, 130, "原始语料与题库", ["QASPER full_text / 章节段落", "历史 SQuAD / arXiv 另行保留", "源数据、划分、版本清单"], "data")
    d.box(430, 85, 330, 130, "规范文本与证据映射", ["doc_id / section_id", "canonical_start / canonical_end", "证据片段可回到原文坐标"], "data")
    d.box(830, 85, 330, 130, "结构与参考切块", ["R512 参考块 / chunk_id", "邻块与 parent/section 关系", "不冒称相似度驱动语义切块"], "data")
    d.arrow([(360, 150), (430, 150)])
    d.arrow([(760, 150), (830, 150)])
    d.box(30, 315, 330, 135, "BM25 稀疏索引", ["正文 / 元数据 chunk 索引", "文档级 BM25", "压缩矩阵、词表与清单"], "data")
    d.box(430, 315, 330, 135, "真实 Embedding 编码", ["Qwen Embedding 0.6B / 1024 维", "批量编码 / 受控卸载 / 断点", "复用 13045；新增 6628 向量"], "active")
    d.box(830, 315, 330, 135, "完整 Dense 矩阵", ["vectors.npy：19673 × 1024", "float32 / L2 归一化 / mmap", "文档级 Dense 另有资产"], "data")
    d.arrow([(990, 215), (990, 260), (190, 260), (190, 315)])
    d.arrow([(990, 260), (595, 260), (595, 315)])
    d.arrow([(760, 380), (830, 380)])
    d.box(230, 550, 740, 145, "索引对齐、覆盖与身份校验", ["文档集合 / chunk 行 ID SHA / 文本与模型身份", "向量维度、有限值、单位范数 / sparse 与 dense 行顺序", "FULL_DENSE_INDEX.json → FullCorpusRetriever 启动时验证"], "active")
    d.arrow([(195, 450), (195, 620), (230, 620)])
    d.arrow([(995, 450), (995, 620), (970, 620)])
    d.text(30, 736, "所有大资产位于 D:/RAGScope_Runtime；本说明生成不重新编码、不重建索引、不启动 GPU。", 17)
    d.legend(770)
    return d


def online_chain():
    d = Diagram("03-online", "在线：一次真实问题的函数调用链", "当前完整工作台经本地 API 使用 FullWorkbench；并非旧 Workbench 或 RAGPipeline.answer 全链路。", height=1100)
    rows = [
        ("① 工作台输入", ["ui/full_workbench.py", "question / doc_ids / task / method"], "② API 接收", ["scripts/v3/serve_full_workbench.py", "POST /api/answer；串行执行"]),
        ("③ 请求与范围核验", ["FullWorkbench.answer", "空问题 / 无效论文 / 歧义对象处理"], "④ 查询真实编码", ["ReusableEncoder.query(cached=True)", "同身份查询缓存；Qwen 1024 维"]),
        ("⑤ 任务与证据选择", ["deployment_context", "普通问答 / 总结 / 比较 / 多篇"], "⑥ 完整索引检索", ["FullCorpusRetriever.search", "范围过滤 / BM25 + Dense / RRF"]),
        ("⑦ 原文证据打包", ["pack_context / answer_messages", "参考预算 2048 或最多 6144 token"], "⑧ 本地真实生成", ["DeploymentClient.chat → 私有 Ollama", "2B / 后端窗口 8192 / 输出 512 或 768"]),
        ("⑨ 解析与引用检查", ["RAGPipeline.parse_output", "JSON 字段 / 可见 chunk ID / token 核验"], "⑩ 返回与可追溯记录", ["answer / citations / abstain / 原文", "独立 APPLICATION_*.jsonl 保留失败"]),
    ]
    y = 90
    for i, (left, ll, right, rl) in enumerate(rows):
        d.box(55, y, 485, 120, left, ll)
        d.box(665, y, 485, 120, right, rl)
        d.arrow([(540, y + 60), (665, y + 60)])
        if i < len(rows) - 1:
            d.arrow([(1150, y + 60), (1180, y + 60), (1180, y + 145), (30, y + 145), (30, y + 230), (55, y + 230)])
        y += 180
    d.text(55, 1026, "模型异常或 JSON/引用校验失败：记录 EXECUTION_ERROR；不会用 Mock 冒充真实答案。", 17)
    return d


def multilevel_retrieval():
    d = Diagram("04-retrieval", "多级论文检索、任务证据与生成预算", "全库与指定论文使用不同范围策略；文档得分在全库中作为块级融合先验，不是强制只搜前十篇。", height=1010)
    d.box(415, 75, 370, 105, "问题 + 显式论文范围", ["doc_ids 为空：全库；否则：所选论文", "最多 10 篇；任务可自动或显式选择"])
    d.box(45, 245, 495, 145, "全库检索", ["文档 BM25 / Dense → 文档先验", "全部 chunk 候选：正文 + 元数据 + Dense", "按权重 RRF 融合，并加入文档先验分数"], "active")
    d.box(660, 245, 495, 145, "单篇 / 多篇指定范围", ["只取指定 doc_id 对应的 chunk 行", "跳过全库文档定位；BM25 / Dense / Hybrid", "多篇可按论文交替组织候选"], "active")
    d.arrow([(490, 180), (490, 210), (295, 210), (295, 245)])
    d.arrow([(710, 180), (710, 210), (905, 210), (905, 245)])
    d.box(45, 475, 495, 145, "普通事实 / 解释", ["所选论文：保留 B0 Hybrid + 邻块", "全库：开发集选定的全库检索参数", "最多 2048 参考 token；输出 512"], "active")
    d.box(660, 475, 495, 145, "总结 / 比较 / 多篇", ["总结：章节线索与原文顺序补充证据", "比较 / 多篇：分组交替，避免单篇占满", "最多 6144 参考 token；输出 768"], "active")
    d.arrow([(295, 390), (295, 432), (295, 475)])
    d.arrow([(905, 390), (905, 432), (905, 475)])
    d.arrow([(540, 315), (600, 315), (600, 433), (905, 433), (905, 475)])
    d.arrow([(660, 315), (625, 315), (625, 450), (295, 450), (295, 475)])
    d.box(245, 715, 710, 125, "去重、原文坐标与实际模型预算检查", ["pack_context → 带 chunk_id / doc_id / canonical 坐标的 ContextBundle", "统一后端窗口 8192；完整 prompt + 输出预留必须适配", "可见引用 ID 验证 ≠ 引用支持性证明；证据覆盖 ≠ 答案正确率"], "active")
    d.arrow([(295, 620), (295, 665), (430, 665), (430, 715)])
    d.arrow([(905, 620), (905, 665), (770, 665), (770, 715)])
    d.box(45, 890, 1110, 72, "历史 Cross-Encoder / 学习型检索路由：当前未进入默认链路", ["历史候选筛选与重排可改变证据覆盖，不能把实验存在写成在线已启用。"], "history", title_size=18)
    return d


def evolution():
    d = Diagram("05-evolution", "V1 → V2 → V3：工程演进与能力边界", "时间线区分工程扩展和测得质量；不同数据与指标不能跨版本直接比较。", height=800)
    d.box(35, 100, 340, 300, "V1：检索研究", ["SQuAD / arXiv 历史实验", "短文本 chunking 对照", "BM25 / Dense / Hybrid", "BGE / Chroma / Cross-Encoder", "检索指标：Recall / MRR / nDCG", "高召回不等于生成高 F1", "旧实现与报告继续保留"], "history")
    d.box(430, 100, 340, 300, "V2：长论文与生成研究", ["长文本、结构切块与证据坐标", "真实本地生成模型对照", "检索、生成、路由、LoRA 实验", "模型身份与配置哈希", "冻结题目 / 失败保留 / 断点", "更多模块未自动带来更高质量", "负结果构成已有证据"], "history")
    d.box(825, 100, 340, 300, "V3：全库与任务工作台", ["1585 篇 / 19673 块完整 Dense", "范围检索、全库文档先验", "总结比较的章节与分组证据", "统一 8K 驻留窗口，修复重载", "真实问答 / 全库 / 在线测试", "保留 2B；停用无充分收益组件", "完整本地 UI / API / 结果追溯"], "active")
    d.arrow([(375, 245), (430, 245)])
    d.arrow([(770, 245), (825, 245)])
    d.box(115, 500, 970, 130, "可证明的最终范围", ["同批题、同评测条件：普通问答 F1 0.3904 → 0.4065；任务 F1 0.4274 → 0.4582", "全库 Hybrid 完整证据覆盖 11.18% → 17.65%，但文档命中 40.5% → 38.0%", "不可答正确拒答仍为 3/30；语义相似度、引用合法性不能替代人工事实判定"], "note")
    d.legend(700)
    return d


def evaluation():
    d = Diagram("06-evaluation", "实验与评价：从问题、对照到可以支持的结论", "只使用真实模型日志作为真实实验；手工应用日志与冻结评测独立，失败样本不剔除。", height=1000)
    d.box(30, 85, 350, 125, "数据与划分身份", ["训练 / 开发 / 冻结评估分离", "问题、参考答案与证据坐标", "推理输入不混入参考答案"], "data")
    d.box(425, 85, 350, 125, "开发选型", ["模型 / Prompt / 检索参数对照", "有限预算与不利结果保留", "选型后固定正式配置"], "experiment")
    d.box(820, 85, 350, 125, "固定实验条件", ["同题配对 B0 与最终配置", "模型、索引、配置 SHA 与请求", "真实后端响应 / 错误 / 耗时"], "experiment")
    d.arrow([(380, 145), (425, 145)])
    d.arrow([(775, 145), (820, 145)])
    d.box(30, 315, 350, 155, "检索评价", ["全库：200 题 × 3 方法 × 2 配置", "共 1200 条；scope ID 移除", "文档命中 / 完整证据覆盖", "固定预算：检索好 ≠ 生成一定好"], "experiment")
    d.box(425, 315, 350, 155, "生成评价", ["H_QA 200 + H_tasks 20", "B0 / 最终，共 440 条评分", "EM / F1 / ROUGE-L / 语义余弦", "拒答、数字与极性规则诊断"], "experiment")
    d.box(820, 315, 350, 155, "真实在线评价", ["每组 30 条，无答案/查询缓存", "编码 + 检索 + 打包 + 真实生成", "热请求 P50 / P95；保留预热", "不将缓存命中当作模型速度"], "experiment")
    d.arrow([(995, 210), (995, 260), (195, 260), (195, 315)], dashed=True)
    d.arrow([(995, 260), (600, 260), (600, 315)], dashed=True)
    d.arrow([(995, 260), (995, 315)], dashed=True)
    d.box(230, 575, 740, 145, "汇总、配对变化与证据边界", ["正式逐题记录 → 分组统计 → 配对改善/退步 → 中文报告", "保留旧版、失败与准确分母；语义余弦不是事实正确率", "引用 ID 可见不是引用支持；自动标记不是人工确认的事实错误"], "experiment")
    d.arrow([(195, 470), (195, 650), (230, 650)], dashed=True)
    d.arrow([(600, 470), (600, 575)], dashed=True)
    d.arrow([(995, 470), (995, 650), (970, 650)], dashed=True)
    d.box(30, 810, 535, 110, "工程功能验证", ["82 项测试；真实 API 与 Streamlit 组件验收", "不作为模型质量、人工事实审核的替代"], "active")
    d.box(635, 810, 535, 110, "独立应用交互", ["APPLICATION_*.jsonl；可选内存答案缓存", "不回写正式成绩；浏览器视觉验收未完成"], "active")
    return d


def generate(directory=ROOT / "diagrams"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return [fn().save(directory) for fn in (
        global_architecture, offline_pipeline, online_chain,
        multilevel_retrieval, evolution, evaluation,
    )]


if __name__ == "__main__":
    for p in generate():
        print(p)
