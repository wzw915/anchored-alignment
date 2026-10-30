"""Markdown -> academic-paper HTML（确定性转换，正文逐字保留）。

遵循 doc-typeset 的 academic-paper 规范：
  * h2 = 一、二、三…（正文主体章节）；h3 = （一）（二）…（章节内小节）
  * 表格一律三线表 .three-line-table（无竖线）
  * 图片 -> <figure>/<figcaption>
  * 每块显式声明 text-align（html4docx 不做继承）
  * 禁止 grid / dl / span-badge / 装饰短横线
产物为单个完整 HTML：封面 section + 目录 section + 正文 section。

用法: python src/md_to_html.py --md <报告.md> --out <输出.html> --img-src <图片目录> --img-dst <HTML同级images目录>
"""
import os
import re
import html
import shutil
import base64
import mimetypes
import argparse


def data_uri(path: str) -> str:
    """把图片编码为 data URI。

    实测：转换为 docx 时，相对路径与 file:// 绝对路径都会被落成占位文本
    `[图片: src]`（不嵌入图片），只有 data URI 能真正嵌入 word/media/。
    """
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ---------- 行内元素 ----------
def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r'<code>\1</code>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", t)
    return t


def split_row(line: str):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|?[\s:\-|]+\|?", line.strip())) and "-" in line


CN_NUM = "一二三四五六七八九十"


def cn(n: int) -> str:
    if n <= 10:
        return CN_NUM[n - 1]
    if n < 20:
        return "十" + CN_NUM[n - 11]
    return CN_NUM[n // 10 - 1] + "十" + (CN_NUM[n % 10 - 1] if n % 10 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-src", required=True)
    ap.add_argument("--img-dst", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--meta", default=None,
                    help="封面元信息行, 分号分隔; 缺省用通用行")
    ap.add_argument("--date", default=None, help="完成日期")
    a = ap.parse_args()

    md = open(a.md, encoding="utf-8").read().replace("\r\n", "\n")
    lines = md.split("\n")

    # ---------- 预扫描：抽出标题/副标题 ----------
    title = a.title
    subtitle = a.subtitle
    body_start = 0
    for i, ln in enumerate(lines[:12]):
        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
        elif ln.startswith("## ——") and not subtitle:
            subtitle = ln[3:].strip()

    os.makedirs(a.img_dst, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(a.out))

    # ---------- 正文块解析 ----------
    blocks = []          # (kind, payload)
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue
        if s == "---":
            i += 1
            continue
        if s.startswith("```"):
            j = i + 1
            buf = []
            while j < n and not lines[j].strip().startswith("```"):
                buf.append(lines[j]); j += 1
            blocks.append(("pre", "\n".join(buf)))
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            txt = m.group(2).strip()
            if lvl == 1 and txt == title:
                i += 1; continue
            if lvl == 2 and txt.startswith("——"):
                i += 1; continue
            blocks.append(("h", (lvl, txt)))
            i += 1
            continue
        if s.startswith("|") and i + 1 < n and is_sep(lines[i + 1]):
            head = split_row(s)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i])); i += 1
            blocks.append(("table", (head, rows)))
            continue
        if s.startswith("!["):
            m2 = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", s)
            if m2:
                blocks.append(("img", (m2.group(1), m2.group(2))))
                i += 1
                continue
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            blocks.append(("quote", " ".join(x for x in buf if x)))
            continue
        if re.match(r"^[-*]\s+", s):
            buf = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                buf.append(re.sub(r"^[-*]\s+", "", lines[i].strip())); i += 1
            # 续行并入
            while i < n and lines[i].startswith(("  ", "\t")) and lines[i].strip():
                buf[-1] += " " + lines[i].strip(); i += 1
            blocks.append(("ul", buf))
            continue
        if re.match(r"^\d+\.\s+", s):
            buf = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                buf.append(re.sub(r"^\d+\.\s+", "", lines[i].strip())); i += 1
            while i < n and lines[i].startswith(("  ", "\t")) and lines[i].strip():
                buf[-1] += " " + lines[i].strip(); i += 1
            blocks.append(("ol", buf))
            continue
        blocks.append(("p", s))
        i += 1

    # ---------- 组装 ----------
    OUT = []
    toc = []
    body_html = []
    h2_i = 0
    h3_i = 0
    in_appendix = False
    abstract_open = False
    refs_open = False

    def close_open():
        nonlocal abstract_open, refs_open
        if abstract_open:
            body_html.append("</div>")
            abstract_open = False
        if refs_open:
            body_html.append("</div>")
            refs_open = False

    for kind, payload in blocks:
        if kind == "h":
            lvl, txt = payload
            if lvl == 2:
                if txt in ("摘要", "Abstract"):
                    close_open()
                    body_html.append('<div class="abstract" id="sec-abstract" aria-label="摘要">')
                    body_html.append('<p class="abstract-heading">摘要</p>')
                    abstract_open = True
                    toc.append(("1", "摘要", "sec-abstract"))
                    continue
                if txt in ("参考文献", "附录 C：主要参考"):
                    close_open()
                    body_html.append('<div class="references" aria-label="参考文献">')
                    body_html.append('<h2 id="sec-refs">参考文献</h2>')
                    refs_open = True
                    toc.append(("1", "参考文献", "sec-refs"))
                    continue
                close_open()
                in_appendix = txt.startswith("附录")
                if in_appendix:
                    sec_id = "sec-" + re.sub(r"[^\w]", "", txt)[:24]
                    body_html.append(f'<h2 id="{sec_id}">{inline(txt)}</h2>')
                    toc.append(("1", txt, sec_id))
                else:
                    h2_i += 1
                    h3_i = 0
                    num = txt
                    sec_id = f"section-{h2_i}"
                    body_html.append(f'<h2 id="{sec_id}">{inline(num)}</h2>')
                    toc.append(("1", num, sec_id))
            elif lvl == 3:
                h3_i += 1
                # 1.1 -> （一）
                txt2 = re.sub(r"^\d+\.\d+\s*", "", txt)
                label = f"（{cn(h3_i)}）{txt2}"
                anc = f"sec{h2_i}-{h3_i}"
                body_html.append(f'<h3 id="{anc}">{inline(label)}</h3>')
                toc.append(("2", label, anc))
            else:
                body_html.append(f"<h4>{inline(payload[1])}</h4>")
        elif kind == "p":
            body_html.append(f"<p>{inline(payload)}</p>")
        elif kind == "quote":
            body_html.append(
                '<p style="border-left:3px solid var(--color-info);'
                'background:var(--color-info-bg);padding:var(--spacing-pad-pre);text-indent:0">'
                f"&nbsp;&nbsp;{inline(payload)}</p>")
        elif kind == "ul":
            body_html.append('<ul style="text-indent:0">' +
                             "".join(f"<li>{inline(x)}</li>" for x in payload) + "</ul>")
        elif kind == "ol":
            if refs_open:
                body_html.append('<ol class="reference-list">' +
                                 "".join(f'<li class="reference-item">{inline(x)}</li>' for x in payload) +
                                 "</ol>")
            else:
                body_html.append('<ol style="text-indent:0">' +
                                 "".join(f"<li>{inline(x)}</li>" for x in payload) + "</ol>")
        elif kind == "pre":
            body_html.append("<pre>" + html.escape(payload) + "</pre>")
        elif kind == "table":
            head, rows = payload
            t = ['<table class="three-line-table"><thead><tr>']
            t += [f"<th>{inline(c)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            body_html.append("".join(t))
        elif kind == "img":
            alt, path = payload
            base = os.path.basename(path.replace("\\", "/"))
            src = os.path.join(a.img_src, base)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(a.img_dst, base))   # 契约：图片落 images/
                body_html.append(
                    f'<figure><img src="{data_uri(src)}" alt="{html.escape(alt)}" '
                    f'style="max-width:100%"></figure>')
                print(f"[md2html] 内联图片 {base}")
            else:
                print(f"[md2html] !! 缺图 {base}")

    close_open()

    toc_html = "\n".join(
        (f'<li><a href="#{anc}">{html.escape(txt)}</a></li>' if lvl == "1"
         else f'<li style="margin-left:0"><ol class="toc-list">'
              f'<li><a href="#{anc}">{html.escape(txt)}</a></li></ol></li>')
        for lvl, txt, anc in toc)

    css = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_academic_paper.css"), encoding="utf-8").read()

    meta_lines = [x.strip() for x in a.meta.split(";") if x.strip()] if a.meta else []
    meta_html = "\n".join(
        f'    <p class="cover-meta-line">{html.escape(x)}</p>' for x in meta_lines)
    date_txt = a.date or __import__("datetime").date.today().isoformat()

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="docx-page-size" content="A4">
<title>{html.escape(title)}</title>
<style>
{css}
</style>
</head>
<body>
<section role="cover">
  <div class="cover-body">
    <p class="cover-eyebrow">人工智能 · 课程大作业报告</p>
    <p class="cover-title">{html.escape(title)}</p>
    <p class="cover-subtitle">{html.escape(subtitle)}</p>
  </div>
  <div class="cover-meta">
    <p class="cover-meta-line">课　　程：人工智能</p>
{meta_html}
    <p class="cover-meta-line">完成日期：{date_txt}</p>
  </div>
</section>

<section role="cover">
<nav class="doc-toc" aria-label="文档目录">
  <p class="toc-title">目　录</p>
  <ol class="toc-list">
{toc_html}
  </ol>
</nav>
</section>

<section role="body" data-page-restart="1">
<main class="paper-body">
{chr(10).join(body_html)}
</main>
</section>
</body>
</html>
"""

    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[md2html] HTML -> {a.out}")
    print(f"[md2html] 章节 {h2_i} 个（h2）/ 小节 {sum(1 for l,_,_ in toc if l=='2')} 个（h3）/ 目录条目 {len(toc)} 条")
    print(f"[md2html] 图片 -> {a.img_dst}")


if __name__ == "__main__":
    main()
