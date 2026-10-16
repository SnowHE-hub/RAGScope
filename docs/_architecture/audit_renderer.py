"""CPU-only documentation conversion fidelity, size, and SVG boundary audit."""
from __future__ import annotations

from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

from build_architecture import MarkdownRenderer, wrap_document, check_document, inline, table_cells, self_test

HERE = Path(__file__).resolve().parent


def stress_source():
    source = ["# RAGScope 中文压力与保真核验", ""]
    for i in range(19):
        source.extend([f"## {i}. 中文章节与路径核验", ""])
        if i == 0:
            source.extend([
                r"[代码] 文件 `ragscope/v3/full_workbench.py:50` 与 [本地路径](../../ragscope/v3/full_workbench.py#L50)。",
                "",
                r"Windows 路径 `D:\RAGScope_Runtime\runs\v3_finish_20261015_1415\vectors.npy`；含空格路径 `C:\Users\A B\实验\a.json`。",
                "",
                "| 类型 | 内容 | 数值 |", "| --- | --- | ---: |",
                r"| 简单管道 | `a|b` | 0.40651362118691137 |",
                r"| 转义管道 | left\|right | 10.13 → 1.99 |",
                r"| 多反引号 | ``a`b|c`` | 19673 × 1024 |",
                r"| 代码路径 | `ragscope/v3/deployment.py:25` | 6144 / 8192 |",
                "", "```python",
                r"path = r'D:\RAGScope_Runtime\runs\v3_finish_20261015_1415\vectors.npy'",
                'payload = {"中文": "<证据>&引号\\\\路径", "condition": 1 < 2 > 0}',
                r"pattern = r'\b(a|b)\b\s+(\d+)'", "```", "",
            ])
        for j in range(14):
            source.extend([f"第{i}章第{j}段。" + "这是离线文档压力样本，包含中文说明、数值 0.3904 → 0.4065、代码 `FullCorpusRetriever.search`，不得在转换中丢失。" * 4, ""])
        if i < 6:
            source.extend([sorted((HERE / "diagrams").glob("*.svg"))[i].read_text(encoding="utf8"), ""])
    return "\n".join(source)


def svg_boundaries():
    # Pillow and the Chinese font are already installed on this machine. This
    # supplements the completed actual sharp render, not a browser screenshot.
    from PIL import ImageFont
    normal = Path("C:/Windows/Fonts/msyh.ttc")
    bold = Path("C:/Windows/Fonts/msyhbd.ttc")
    result = []
    for p in sorted((HERE / "diagrams").glob("*.svg")):
        root = ET.fromstring(p.read_text(encoding="utf8"))
        _, _, width, height = map(float, root.attrib["viewBox"].split())
        count = 0; outside = []; rects = []
        for e in root.iter():
            tag = e.tag.rsplit("}", 1)[-1]
            if tag == "rect" and all(k in e.attrib for k in ("x", "y", "width", "height")):
                rects.append(tuple(float(e.attrib[k]) for k in ("x", "y", "width", "height")))
            if tag != "text": continue
            count += 1; text = "".join(e.itertext()); size = int(float(e.attrib.get("font-size", 16)))
            font = ImageFont.truetype(str(bold if int(e.attrib.get("font-weight", 400)) >= 600 else normal), size)
            x, y = float(e.attrib.get("x", 0)), float(e.attrib.get("y", 0))
            left, top, right, bottom = font.getbbox(text, anchor="ls")
            bounds = (x + left, y + top, x + right, y + bottom)
            if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > width or bounds[3] > height:
                outside.append({"text": text, "bounds": bounds})
        if outside: raise AssertionError(f"SVG canvas clips text in {p.name}: {outside}")
        result.append({"file": p.name, "sha256": sha256(p.read_bytes()).hexdigest(),
                       "viewBox": root.attrib["viewBox"], "text_nodes": count,
                       "font": str(normal), "text_outside_viewBox": outside,
                       "actual_render": str(HERE / "previews" / (p.stem + ".png"))})
    return result


def main():
    small = self_test()
    # Explicit checks catch markup accidentally swallowing evidence tags/paths.
    linked = inline("[代码] 路径 `ragscope/v3/full_workbench.py:50`，参见 [源码](../../ragscope/v3/full_workbench.py#L50)。")
    assert linked.startswith("[代码] 路径 <code>")
    assert linked.count("<a ") == 1 and '>源码</a>' in linked
    assert table_cells(r"| x | ``a`b|c`` | a\|b |") == ["x", "``a`b|c``", r"a\|b"]
    source = stress_source(); began = time.perf_counter()
    record = MarkdownRenderer().render(source); document = wrap_document(source, record)
    receipt = check_document(source, document); elapsed = time.perf_counter() - began
    assert len(source.encode("utf8")) > 150_000 and len(document.encode("utf8")) > 150_000
    assert 'D:\\RAGScope_Runtime\\runs\\v3_finish_20261015_1415\\vectors.npy' in unescape(document)
    assert 'ragscope/v3/full_workbench.py:50' in unescape(document)
    receipt.update({"source_bytes": len(source.encode("utf8")), "html_bytes": len(document.encode("utf8")),
                    "render_and_full_consistency_seconds": elapsed,
                    "large_document_paragraphs": 266,
                    "performance_scope": "CPU-only synthetic document; not a RAG/model benchmark",
                    "extended_fidelity_cases": ["inline path:line", "Windows backslashes and spaces", "code < > & and quotes", "escaped/code/multiple-backtick table pipes", "Chinese headings", "evidence tag followed by actual link"],
                    "converter_self_test": small,
                    "svg_boundary_checks": svg_boundaries(),
                    "browser_started": False, "gpu_started": False})
    out = HERE / "RENDERER_QA.json"
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
