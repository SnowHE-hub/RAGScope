"""Read the complete provided HTML reference; record design, not finance facts."""
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class Reference(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = Counter()
        self.headings = []
        self.body_text = []
        self.style = []
        self.svg = []
        self.current_svg = None
        self.current_heading = None
        self.in_body = self.in_style = False

    def handle_starttag(self, tag, attrs):
        self.tags[tag] += 1
        if tag == "body": self.in_body = True
        if tag == "style": self.in_style = True
        if re.fullmatch(r"h[1-6]", tag): self.current_heading = []
        if tag == "svg":
            self.current_svg = {"attrs": dict(attrs), "elements": Counter(), "labels": []}
            self.svg.append(self.current_svg)
        if self.current_svg is not None: self.current_svg["elements"][tag] += 1

    def handle_endtag(self, tag):
        if tag == "body": self.in_body = False
        if tag == "style": self.in_style = False
        if re.fullmatch(r"h[1-6]", tag) and self.current_heading is not None:
            self.headings.append("".join(self.current_heading)); self.current_heading = None
        if tag == "svg": self.current_svg = None

    def handle_data(self, data):
        if self.in_body: self.body_text.append(data)
        if self.in_style: self.style.append(data)
        if self.current_heading is not None: self.current_heading.append(data)
        if self.current_svg is not None and data.strip(): self.current_svg["labels"].append(data.strip())


def main():
    p = Path("D:/MCS_26S2课程资料/quantmind/docs/architecture.html")
    raw = p.read_bytes(); text = raw.decode("utf8")
    parser = Reference(); parser.feed(text); parser.close()
    result = {
        "path": str(p), "sha256": sha256(raw).hexdigest(), "bytes": len(raw),
        "parsed_entire_file": True, "text_characters": len("".join(parser.body_text)),
        "tags": dict(parser.tags), "heading_count": len(parser.headings),
        "style": "".join(parser.style),
        "svg_structures": [{"attrs": row["attrs"], "elements": row["elements"]} for row in parser.svg],
        "reference_facts_not_copied": "只记录样式与结构，不复制金融模块、实验数值或业务正文。",
        "adopted_design": ["暖白背景 / 紫色强调", "本机中文无衬线字体与等宽代码", "章节横线、引用底色、表格斑马纹", "完整目录 / SVG 内嵌 / 响应式 / 打印样式"],
        "adapted_for_larger_handbook": ["桌面粘性完整目录", "手机目录滚动与宽图横向滚动", "不从参考拷贝金融模块与数字", "Markdown 单正文生成、锚点与表格代码一致性校验"],
    }
    out = Path(__file__).resolve().parent / "REFERENCE_STYLE_AUDIT.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
