"""Build the offline handbook from its single Markdown source (stdlib only).

Run from the repository root with an existing Python, for example::

    D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py
    D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py --check-only
    D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py --self-test
    D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py --diagrams

Supported authoring syntax: ATX headings, paragraphs, fenced code, Markdown
tables (including escaped pipes), ordered/unordered nested lists, blockquotes,
horizontal rules, links, images, inline code/emphasis, and raw inline SVG.
No remote resources are downloaded or used. Raw SVG must already be embedded
in the Markdown. SVG XML is checked and copied byte-for-byte to the HTML.
Other raw HTML is deliberately escaped so document structure stays auditable.

This is a documentation utility, never a runtime/experiment entry point.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
from html import escape, unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_SOURCE = REPO / "docs" / "architecture.md"
DEFAULT_OUTPUT = REPO / "docs" / "architecture.html"
CSS = r"""
:root{--bg:#fdfcf9;--paper:#fff;--ink:#242421;--muted:#66665f;--line:#dedbd2;
--accent:#534ab7;--card:#f5f3ed;--code:#f1efe8;--toc-width:290px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:28px;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,
"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;font-size:16px;line-height:1.88}
.layout{max-width:1510px;margin:auto;display:grid;grid-template-columns:var(--toc-width) minmax(0,1fr);gap:42px;padding:28px 32px 90px}
.sidebar{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;border:1px solid var(--line);
border-radius:10px;background:var(--card);padding:15px 17px;font-size:13.4px;line-height:1.65;scrollbar-width:thin}
.sidebar h2{border:0;padding:0;margin:0 0 9px;font-size:17px}.sidebar ol{list-style:none;margin:0;padding:0}
.sidebar li{margin:4px 0}.sidebar li[data-level="3"]{padding-left:12px}.sidebar li[data-level="4"]{padding-left:24px}
.sidebar li[data-level="5"],.sidebar li[data-level="6"]{padding-left:32px}.sidebar a{color:var(--muted);border:0}
.sidebar a:hover{color:var(--accent)}main{min-width:0;max-width:1100px}
h1{font-size:33px;line-height:1.4;letter-spacing:-.02em;font-weight:650;margin:15px 0 22px}
h2{font-size:25px;font-weight:650;margin:62px 0 18px;padding-top:22px;border-top:1px solid var(--line);line-height:1.5}
h3{font-size:20px;line-height:1.5;margin:36px 0 12px}h4{font-size:17px;margin:28px 0 10px}h5,h6{font-size:16px;margin:23px 0 10px}
h1,h2,h3,h4,h5,h6{overflow-wrap:anywhere;scroll-margin-top:22px}
p{margin:14px 0}strong{font-weight:650}a{color:var(--accent);text-decoration:none;border-bottom:1px solid #c5bfe7;overflow-wrap:anywhere}
a:hover{border-bottom-color:var(--accent)}.heading-anchor{font-size:.65em;margin-left:.45em;opacity:0;border:0}
h1:hover .heading-anchor,h2:hover .heading-anchor,h3:hover .heading-anchor,h4:hover .heading-anchor{opacity:1}
ul,ol{padding-left:26px;margin:14px 0}li{margin:5px 0}li>p{margin:6px 0}li ul,li ol{margin:5px 0}
blockquote{background:var(--card);border-left:4px solid var(--accent);margin:22px 0;padding:10px 19px;border-radius:0 8px 8px 0;color:#4c4c45}
blockquote p{margin:6px 0}hr{border:0;border-top:1px solid var(--line);margin:40px 0}
code{font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace;font-size:.86em;
background:var(--code);padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}
pre{background:var(--code);border:1px solid var(--line);border-radius:9px;padding:16px 19px;overflow:auto;line-height:1.65;margin:21px 0;tab-size:4}
pre code{padding:0;background:none;font-size:13.5px;white-space:pre;overflow-wrap:normal;border-radius:0}
.table-wrap{overflow:auto;margin:23px 0;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14.4px;line-height:1.75;min-width:520px;margin:0}
th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);border-right:1px solid var(--line);overflow-wrap:anywhere}
tr:last-child td{border-bottom:0}th:last-child,td:last-child{border-right:0}th{background:var(--card);font-weight:650}
tbody tr:nth-child(even){background:#faf9f5}td code,th code{font-size:.88em}
.diagram{margin:28px 0;padding:4px;overflow:auto;background:var(--paper);border:1px solid var(--line);border-radius:10px}
.diagram svg{display:block;width:100%;min-width:630px;height:auto;background:var(--paper)}
.build-note{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:14px;margin-top:50px;word-break:break-all}
img{max-width:100%;height:auto}::selection{background:#e3dff8}
@media(max-width:1100px){.layout{gap:24px;padding:20px 23px 75px;grid-template-columns:240px minmax(0,1fr)}.sidebar{font-size:12.5px}}
@media(max-width:820px){.layout{display:block;padding:14px 18px 65px}.sidebar{position:static;max-height:340px;margin:10px 0 28px;font-size:13.5px}
body{font-size:15.5px}h1{font-size:27px}h2{font-size:22px;margin-top:45px}h3{font-size:18.5px}.diagram svg{min-width:1080px}
pre{padding:13px}table{font-size:13.5px}.heading-anchor{opacity:.5}}
@media print{@page{size:A4;margin:17mm 15mm}html{scroll-behavior:auto}body{background:white;font-size:9.5pt;line-height:1.65}
.layout{display:block;padding:0;max-width:none}main{max-width:none}.sidebar{position:static;max-height:none;overflow:visible;background:white;
font-size:8.5pt;line-height:1.4;break-after:page;border:0;padding:0}.sidebar ol{columns:2;column-gap:25px}.sidebar li{break-inside:avoid;margin:2px 0}
h1{font-size:23pt}h2{font-size:17pt;margin-top:25pt}h3{font-size:13pt}h4{font-size:11pt}h1,h2,h3,h4,h5,h6{break-after:avoid}
pre{white-space:pre-wrap;overflow:visible;break-inside:avoid;padding:8pt}pre code{white-space:pre-wrap;overflow-wrap:anywhere;font-size:7.6pt}
.table-wrap{overflow:visible;border-radius:0;break-inside:auto}table{min-width:0;font-size:8pt}thead{display:table-header-group}
tr{break-inside:avoid}th,td{padding:5pt}.diagram{overflow:visible;break-inside:avoid;border:0;padding:0}.diagram svg{min-width:0;width:100%}
blockquote{break-inside:avoid}a{color:inherit;border:0}.heading-anchor{display:none}.build-note{font-size:7pt}}
"""


def digest(text):
    return sha256(text.encode("utf8")).hexdigest()


def plain_inline(text):
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[*`~]", "", text).replace("\\|", "|")


def slug(text):
    text = unicodedata.normalize("NFKC", plain_inline(text)).strip().lower()
    text = re.sub(r"[^\w\u3400-\u9fff\- ]+", "", text, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "-", text).strip("-") or "section"


def inline(text):
    """Inline conversion with escaped literals; paths in code are untouched."""
    result, i = [], 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in r"\`*_{}[]()#+-.!|>~":
            result.append(escape(text[i + 1])); i += 2; continue
        if text[i] == "`":
            ticks = re.match(r"`+", text[i:])[0]
            end = text.find(ticks, i + len(ticks))
            if end >= 0:
                result.append("<code>" + escape(text[i + len(ticks):end]) + "</code>")
                i = end + len(ticks); continue
        image = text.startswith("![", i)
        if image or text[i] == "[":
            start = i + (2 if image else 1)
            # A preceding evidence tag such as [代码] must not become part of
            # an unrelated Markdown link later in the same paragraph.
            close, depth = start, 1
            while close < len(text):
                if text[close] == "\\" and close + 1 < len(text): close += 2; continue
                if text[close] == "[": depth += 1
                elif text[close] == "]":
                    depth -= 1
                    if depth == 0: break
                close += 1
            if depth == 0 and text[close:close+2] == "](":
                end, depth = close + 2, 1
                while end < len(text):
                    if text[end] == "(": depth += 1
                    elif text[end] == ")":
                        depth -= 1
                        if depth == 0: break
                    end += 1
                if depth == 0:
                    label, href = text[start:close], text[close + 2:end].strip()
                    if href.startswith("<") and href.endswith(">"):
                        href = href[1:-1]
                    if image:
                        result.append('<img alt="' + escape(plain_inline(label), quote=True) + '" src="' + escape(href, quote=True) + '" loading="lazy"/>')
                    else:
                        result.append('<a href="' + escape(href, quote=True) + '">' + inline(label) + '</a>')
                    i = end + 1; continue
        matched = False
        for mark, tag in [("**", "strong"), ("__", "strong"), ("~~", "del"), ("*", "em")]:
            if text.startswith(mark, i):
                end = text.find(mark, i + len(mark))
                if end > i + len(mark):
                    result.append(f"<{tag}>" + inline(text[i + len(mark):end]) + f"</{tag}>")
                    i = end + len(mark); matched = True; break
        if not matched:
            result.append(escape(text[i])); i += 1
    return "".join(result)


def table_cells(line):
    text = line.strip()
    if text.startswith("|"): text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"): text = text[:-1]
    cells, value, code_delimiter, i = [], [], 0, 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            value.extend(text[i:i+2]); i += 2; continue
        if text[i] == "`":
            run = len(re.match(r"`+", text[i:])[0])
            if not code_delimiter: code_delimiter = run
            elif code_delimiter == run: code_delimiter = 0
            value.extend(text[i:i+run]); i += run; continue
        if text[i] == "|" and not code_delimiter:
            cells.append("".join(value).strip()); value = []
        else: value.append(text[i])
        i += 1
    cells.append("".join(value).strip())
    return cells


def is_table_separator(line):
    return bool(line.strip()) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in table_cells(line))


@dataclass
class Rendered:
    html: str = ""
    headings: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    code_blocks: list = field(default_factory=list)
    svgs: list = field(default_factory=list)


class MarkdownRenderer:
    def __init__(self):
        self.record = Rendered()
        self.used_ids = {}

    def heading(self, level, title):
        base = slug(title); count = self.used_ids.get(base, 0) + 1; self.used_ids[base] = count
        ident = base if count == 1 else f"{base}-{count}"
        self.record.headings.append((level, ident, plain_inline(title)))
        return f'<h{level} id="{ident}">{inline(title)}<a class="heading-anchor" aria-label="章节链接" href="#{ident}">#</a></h{level}>'

    def render(self, markdown):
        self.record.html = self.blocks(markdown.splitlines())
        return self.record

    def blocks(self, lines):
        out, i = [], 0
        while i < len(lines):
            line = lines[i]
            if not line.strip(): i += 1; continue
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                out.append(self.heading(len(heading[1]), heading[2])); i += 1; continue
            fence = re.match(r"^\s*(`{3,}|~{3,})([^\s]*)\s*$", line)
            if fence:
                opening, lang = fence[1], fence[2]; i += 1; content = []
                while i < len(lines) and not re.fullmatch(r"\s*" + re.escape(opening[0]) + "{" + str(len(opening)) + r",}\s*", lines[i]):
                    content.append(lines[i]); i += 1
                if i >= len(lines): raise ValueError("Unclosed code fence")
                code = "\n".join(content) + "\n"
                self.record.code_blocks.append(code)
                cls = f' class="language-{escape(lang, quote=True)}"' if lang else ""
                out.append("<pre><code" + cls + ">" + escape(code) + "</code></pre>"); i += 1; continue
            if line.lstrip().startswith("<svg"):
                svg_lines = [line]; i += 1
                while "</svg>" not in svg_lines[-1] and i < len(lines):
                    svg_lines.append(lines[i]); i += 1
                svg = "\n".join(svg_lines)
                if "</svg>" not in svg: raise ValueError("Unclosed SVG")
                validate_svg(svg)
                self.record.svgs.append(svg)
                out.append('<figure class="diagram">' + svg + "</figure>"); continue
            if line.strip().startswith("<!--"):
                while "-->" not in lines[i]:
                    i += 1
                    if i >= len(lines): raise ValueError("Unclosed Markdown comment")
                i += 1; continue
            if re.fullmatch(r"\s{0,3}([-*_])(?:\s*\1){2,}\s*", line):
                out.append("<hr/>"); i += 1; continue
            if i + 1 < len(lines) and "|" in line and is_table_separator(lines[i + 1]):
                heads = table_cells(line); sep = table_cells(lines[i + 1]); rows = [heads]; i += 2
                if len(heads) != len(sep): raise ValueError("Table header/separator widths differ")
                while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                    cells = table_cells(lines[i])
                    if len(cells) != len(heads): raise ValueError(f"Table row width differs at: {lines[i][:100]}")
                    rows.append(cells); i += 1
                self.record.tables.append(rows)
                out.append('<div class="table-wrap"><table><thead><tr>' + "".join("<th>" + inline(c) + "</th>" for c in heads) + "</tr></thead><tbody>")
                out.extend("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in row) + "</tr>" for row in rows[1:])
                out.append("</tbody></table></div>"); continue
            if line.lstrip().startswith(">"):
                quoted = []
                while i < len(lines) and (lines[i].lstrip().startswith(">") or not lines[i].strip()):
                    quoted.append(re.sub(r"^\s*> ?", "", lines[i])); i += 1
                out.append("<blockquote>" + self.blocks(quoted) + "</blockquote>"); continue
            item = re.match(r"^(\s*)([-+*]|\d+[.)])\s+(.+)$", line)
            if item:
                base = len(item[1]); ordered = item[2][0].isdigit(); tag = "ol" if ordered else "ul"
                start = int(re.match(r"\d+", item[2])[0]) if ordered else 1
                out.append(f'<ol start="{start}">' if ordered and start != 1 else f"<{tag}>")
                while i < len(lines):
                    match = re.match(r"^(\s*)([-+*]|\d+[.)])\s+(.+)$", lines[i])
                    if not match or len(match[1]) != base or match[2][0].isdigit() != ordered: break
                    content = [match[3]]; continuation_indent = len(match[1]) + len(match[2]) + 1; i += 1
                    while i < len(lines):
                        nxt = lines[i]
                        if not nxt.strip():
                            if i + 1 < len(lines) and len(lines[i + 1]) - len(lines[i + 1].lstrip()) > base:
                                content.append(""); i += 1; continue
                            break
                        lead = len(nxt) - len(nxt.lstrip())
                        if lead <= base: break
                        content.append(nxt[min(lead, continuation_indent):]); i += 1
                    item_html = self.blocks(content)
                    if item_html.startswith("<p>") and item_html.endswith("</p>") and item_html.count("<p>") == 1:
                        item_html = item_html[3:-4]
                    out.append("<li>" + item_html + "</li>")
                    if i < len(lines) and not lines[i].strip(): i += 1
                out.append(f"</{tag}>"); continue
            paragraph = [line.strip()]; i += 1
            while i < len(lines) and lines[i].strip():
                nxt = lines[i]
                if re.match(r"^(#{1,6})\s+|^\s*(`{3,}|~{3,})|^\s*<svg|^\s*>|^\s*<!--|^\s*([-+*]|\d+[.)])\s+", nxt): break
                if i + 1 < len(lines) and "|" in nxt and is_table_separator(lines[i + 1]): break
                if re.fullmatch(r"\s{0,3}([-*_])(?:\s*\1){2,}\s*", nxt): break
                paragraph.append(nxt.strip()); i += 1
            out.append("<p>" + inline("\n".join(paragraph)) + "</p>")
        return "\n".join(out)


def validate_svg(svg):
    root = ET.fromstring(svg)
    if root.tag.rsplit("}", 1)[-1] != "svg" or "viewBox" not in root.attrib:
        raise ValueError("SVG requires svg root and viewBox")
    if not any(e.tag.rsplit("}", 1)[-1] == "title" for e in root.iter()):
        raise ValueError("SVG requires an accessible title")
    for e in root.iter():
        if e.tag.rsplit("}", 1)[-1] in {"script", "foreignObject", "image"}:
            raise ValueError("SVG must be self-contained vector content")
        if any(k.lower().startswith("on") for k in e.attrib): raise ValueError("SVG event handler not allowed")
        for k, value in e.attrib.items():
            if k.rsplit("}", 1)[-1] in {"href", "src"} and not value.startswith("#"):
                raise ValueError("External SVG asset not allowed")


def wrap_document(source, record):
    toc = "\n".join(f'<li data-level="{level}"><a href="#{ident}">{escape(title)}</a></li>'
                    for level, ident, title in record.headings if level >= 2)
    title = next((t for level, _, t in record.headings if level == 1), "RAGScope 项目架构说明")
    return f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="ragscope-markdown-sha256" content="{digest(source)}"/>
<meta name="ragscope-body-sha256" content="{digest(record.html)}"/>
<title>{escape(title)}</title><style>{CSS}</style></head><body>
<div class="layout"><aside class="sidebar"><nav aria-label="完整章节目录"><h2>阅读目录</h2><ol>{toc}</ol></nav></aside>
<main id="document-content">{record.html}</main></div>
<footer class="build-note" style="max-width:1100px;margin:0 auto 35px;padding:0 25px">离线单文件阅读版 · 正文唯一来源：docs/architecture.md · Markdown SHA-256：{digest(source)}</footer>
</body></html>
'''


class Inspection(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.hrefs, self.resources = [], [], []
        self.tables, self.codes, self.heading_rows = [], [], []
        self.in_main = self.in_code = self.in_cell = self.in_heading = False
        self.in_svg = 0
        self.heading_text, self.heading_info, self.skip_anchor = [], None, False
        self.buffer, self.current_table, self.current_row = [], None, None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "id" in d: self.ids.append(d["id"])
        if "href" in d: self.hrefs.append(d["href"])
        if tag in {"script", "img", "iframe", "link"}: self.resources.append((tag, d))
        if tag == "main": self.in_main = True
        if tag == "svg": self.in_svg += 1
        if not self.in_main or self.in_svg: return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.in_heading = True; self.heading_info = (int(tag[1]), d.get("id")); self.heading_text = []
        if tag == "a" and d.get("class") == "heading-anchor": self.skip_anchor = True
        if tag == "pre": self.in_code = True; self.buffer = []
        if tag == "table": self.current_table = []
        if tag == "tr": self.current_row = []
        if tag in {"td", "th"}: self.in_cell = True; self.buffer = []

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "svg": self.in_svg -= 1; return
        if not self.in_main or self.in_svg: return
        if tag == "a": self.skip_anchor = False
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.in_heading:
            self.heading_rows.append((*self.heading_info, "".join(self.heading_text)))
            self.in_heading = False
        if tag == "pre": self.codes.append("".join(self.buffer)); self.in_code = False
        if tag in {"td", "th"} and self.in_cell:
            self.current_row.append("".join(self.buffer)); self.in_cell = False
        if tag == "tr" and self.current_table is not None: self.current_table.append(self.current_row)
        if tag == "table" and self.current_table is not None:
            self.tables.append(self.current_table); self.current_table = None
        if tag == "main": self.in_main = False

    def handle_data(self, data):
        if self.in_svg: return
        if self.in_heading and not self.skip_anchor: self.heading_text.append(data)
        if self.in_code or self.in_cell: self.buffer.append(data)


def check_document(source, document, enforce_scope=True):
    record = MarkdownRenderer().render(source)
    expected = wrap_document(source, record)
    if document != expected: raise AssertionError("HTML differs from deterministic Markdown build; regenerate it")
    parser = Inspection(); parser.feed(document); parser.close()
    if parser.heading_rows != record.headings: raise AssertionError("Markdown/HTML heading mismatch")
    if parser.codes != record.code_blocks: raise AssertionError("Code/path bytes were altered")
    expected_tables = [[[unescape(re.sub(r"<[^>]+>", "", inline(c))) for c in row] for row in table] for table in record.tables]
    if parser.tables != expected_tables: raise AssertionError("Markdown/HTML table cells differ")
    if len(parser.ids) != len(set(parser.ids)): raise AssertionError("Duplicate HTML/SVG ID")
    missing = sorted({h[1:] for h in parser.hrefs if h.startswith("#")} - set(parser.ids))
    if missing: raise AssertionError("Broken internal anchors: " + repr(missing))
    for tag, attrs in parser.resources:
        if tag in {"script", "iframe", "link"}: raise AssertionError("Unexpected external/active dependency")
        if tag == "img": raise AssertionError("Embed diagram SVG inline; single-file HTML cannot depend on images")
    if re.search(r"@import|url\(\s*['\"]?https?://", CSS, re.I): raise AssertionError("CSS remote dependency")
    if "\ufffd" in source or "\ufffd" in document: raise AssertionError("Unicode replacement character")
    if enforce_scope:
        h2 = [title for level, _, title in record.headings if level == 2]
        for number in range(19):
            if not any(re.match(rf"^{number}[.、．\s]", title) for title in h2):
                raise AssertionError(f"Required main section {number} is missing")
        if len(record.svgs) < 6: raise AssertionError("At least six inline SVG diagrams required")
        if re.search(r"\bTODO\b|待补充占位|此处填入|生成器报错", source, re.I):
            raise AssertionError("Unresolved author placeholder")
    return {"status": "PASS", "markdown_sha256": digest(source), "html_sha256": digest(document),
            "rendered_body_sha256": digest(record.html), "source_characters": len(source),
            "headings": len(record.headings), "main_sections": sum(l == 2 for l, _, _ in record.headings),
            "tables": len(record.tables), "code_blocks": len(record.code_blocks), "inline_svg": len(record.svgs),
            "toc_targets": sum(l >= 2 for l, _, _ in record.headings), "broken_anchors": [],
            "code_blocks_exact": True, "table_cells_exact": True, "svg_xml_valid": True,
            "html_deterministic_from_markdown": True, "offline_external_dependencies": 0}


def self_test():
    source = '''# 中文手册

## 0. 阅读说明

正文 **粗体**、*斜体*、`D:\\RAGScope_Runtime` 与 [章节](#1-重复标题)。

## 1. 重复标题

| 指标 | 值 | 代码 |
| --- | ---: | --- |
| F1 | 0.4065 | `a|b` |
| escaped | 10\\|20 | `x < y` |

```powershell
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe --version
$value = "中 & <文>"
```

> 引用。
>
> - 列表引用

1. 第一项
   - 子项 `foo()`
2. 第二项

## 1. 重复标题

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40"><title>示例图</title><text x="5" y="20">中文</text></svg>
'''
    record = MarkdownRenderer().render(source)
    result = check_document(source, wrap_document(source, record), enforce_scope=False)
    assert record.headings[-1][1] == "1-重复标题-2"
    assert result["tables"] == 1 and result["code_blocks"] == 1 and result["inline_svg"] == 1
    assert "<ul>" in record.html and "<ol>" in record.html
    try: check_document(source, wrap_document(source, record).replace("0.4065", "0.9999"), False)
    except AssertionError: pass
    else: raise AssertionError("Changed table value was not detected")
    return {"self_test": "PASS", "coverage": ["Chinese duplicate anchors", "fenced code exact", "escaped/code pipe tables", "nested lists", "blockquote", "inline SVG", "tamper detection"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diagrams", action="store_true", help="Generate only six source SVG assets, not the final handbook")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2)); return
    if args.diagrams:
        from build_diagrams import generate
        print(json.dumps({"svg_assets": [str(p) for p in generate()]}, ensure_ascii=False, indent=2)); return
    source = args.source.read_bytes().decode("utf8")
    record = MarkdownRenderer().render(source)
    document = wrap_document(source, record)
    if args.check_only:
        actual = args.output.read_text(encoding="utf8")
    else:
        # Never silently destroy a pre-existing HTML edition.
        if args.output.exists() and args.output.read_text(encoding="utf8") != document:
            prior = args.output.read_bytes()
            backup = HERE / "backups" / (args.output.stem + "." + sha256(prior).hexdigest()[:12] + ".html")
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists(): backup.write_bytes(prior)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf8", newline="\n") as handle: handle.write(document)
        actual = args.output.read_text(encoding="utf8")
    result = check_document(source, actual)
    result["source"] = str(args.source.resolve()); result["output"] = str(args.output.resolve())
    receipt = HERE / "CONSISTENCY_CHECK.json"
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
