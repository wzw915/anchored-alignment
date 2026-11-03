# -*- coding: utf-8 -*-
"""把 markdown 论文构建为投稿样式的 docx：
- 无封面、无目录，正文直接从标题开始
- 正文 12 pt Times New Roman，双倍行距，两端对齐，段后 6 pt
- 一级/二级标题加粗，表格三线表（无竖线），图题居中 10 pt 斜体
- 参考文献悬挂缩进
用法: python src/build_paper_docx.py <input.md> <output.docx>
"""
import os, re, sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "Times New Roman"


def set_run_font(run, size=12, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(attr), FONT)


def base_style(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.line_spacing = 2.0
    pf.space_before = Pt(0)
    pf.space_after = Pt(6)


SUB_RE = re.compile(r'(_[A-Za-z0-9]+)')


URL_RE = re.compile(r'(https?://\S+)')


def add_math_runs(p, text, size, bold=False, italic=False):
    """把 X_j / δ_max / τ_eval 这类下标记号渲染成 Word 真下标。
    URL 内部的下划线是文件路径的一部分（如 meta_Books.jsonl），不做下标处理。"""
    for u in URL_RE.split(text):
        if not u:
            continue
        if u.startswith(("http://", "https://")):
            r = p.add_run(u)
            set_run_font(r, size, bold, italic)
            continue
        for seg in SUB_RE.split(u):
            if not seg:
                continue
            if seg.startswith("_") and len(seg) > 1:
                r = p.add_run(seg[1:])
                set_run_font(r, size, bold, italic)
                r.font.subscript = True
            else:
                r = p.add_run(seg)
                set_run_font(r, size, bold, italic)


def add_para(doc, text, size=12, bold=False, italic=False, align="just",
             space_after=6, indent=None, hanging=None, first_indent=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 2.0
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if indent is not None:
        pf.left_indent = Cm(indent)
    if hanging is not None:
        pf.first_line_indent = Cm(-hanging)
    if first_indent is not None:
        pf.first_line_indent = Cm(first_indent)
    pf.alignment = {"just": WD_ALIGN_PARAGRAPH.JUSTIFY,
                    "center": WD_ALIGN_PARAGRAPH.CENTER,
                    "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    # 支持 **bold** 片段
    for i, chunk in enumerate(re.split(r'(\*\*.+?\*\*)', text)):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            add_math_runs(p, chunk[2:-2], size, True, italic)
        else:
            add_math_runs(p, chunk, size, bold, italic)
    return p


def set_row_cant_split(row):
    """禁止表格行内跨页断开（Elsevier 排版要求）。"""
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    trPr.append(el)


def set_cell_borders(cell, top=False, bottom=False):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, on in (("top", top), ("bottom", bottom)):
        el = OxmlElement(f"w:{edge}")
        if on:
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "8")
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)
    for edge in ("left", "right", "insideV", "insideH"):
        el = OxmlElement(f"w:{edge}"); el.set(qn("w:val"), "nil"); borders.append(el)
    tcPr.append(borders)


def add_table(doc, rows):
    # 表 2 特例：Main protocol / Centered 二级表头（合并单元格）
    grp = None
    if rows and sum("Main protocol" in c for c in rows[0]) == 2:
        grp = ["", "Main protocol", "Main protocol", "Centered", "Centered"]
        rows = [rows[0][0:1] + ["RMSE", "Kendall τ", "RMSE", "Kendall τ"]] + rows[1:]
    ncol = len(rows[0])
    n_extra = 1 if grp else 0
    tbl = doc.add_table(rows=len(rows) + n_extra, cols=ncol)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    for ri in range(len(rows) + n_extra):
        set_row_cant_split(tbl.rows[ri])
    for i, row in enumerate(rows):
        di = i + n_extra                      # 数据行在表中的实际行号
        for j, cell_text in enumerate(row):
            cell = tbl.cell(di, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.alignment = (WD_ALIGN_PARAGRAPH.LEFT if j == 0
                                            else WD_ALIGN_PARAGRAPH.CENTER)
            bold = (di == 0) or ("**" in cell_text)
            clean = cell_text.replace("**", "")
            add_math_runs(p, clean, 10, bold)
            set_cell_borders(cell, top=(di <= n_extra),
                             bottom=(di == n_extra or di == len(rows) + n_extra - 1))
    if grp:
        # 分组表头行：写文本并横向合并（只写每组首格，避免合并后文本重复）
        for j, txt in enumerate(grp):
            cell = tbl.cell(0, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_math_runs(p, txt if j in (1, 3) else "", 10, True)
            set_cell_borders(cell, top=True, bottom=False)
        tbl.cell(0, 1).merge(tbl.cell(0, 2))
        tbl.cell(0, 3).merge(tbl.cell(0, 4))
    # 表后补一行间距
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(2)
    sp.paragraph_format.line_spacing = 1.0
    return tbl


def flush_table(doc, buf):
    if not buf:
        return
    rows = []
    for ln in buf:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
            continue
        rows.append(cells)
    if rows:
        add_table(doc, rows)
    buf.clear()


def main(md_path, out_path):
    text = open(md_path, encoding="utf-8").read()
    lines = text.split("\n")
    doc = Document()
    base_style(doc)
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2.54)
    sec.left_margin = sec.right_margin = Cm(2.54)

    tbl_buf = []
    in_refs = False
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|"):
            tbl_buf.append(line); continue
        flush_table(doc, tbl_buf)
        if not line.strip():
            continue
        if line.startswith("# "):
            add_para(doc, line[2:], size=16, bold=True, align="center", space_after=12)
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            in_refs = (title.lower() == "references")
            # 参考文献与附录各起新页, 避免连续排版时附录看起来"插在参考文献中间"
            if title.lower() == "references" or title.startswith("Appendix"):
                doc.add_page_break()
            add_para(doc, title, size=13, bold=True, align="left", space_after=8)
            continue
        if line.startswith("### "):
            add_para(doc, line[4:], size=12, bold=True, align="left", space_after=6)
            continue
        if line.startswith("!["):
            m = re.match(r'!\[(.+?)\]\((.+?)\)', line)
            if m:
                caption, rel = m.group(1), m.group(2)
                path = os.path.join(PROJ, rel)
                if os.path.exists(path):
                    doc.add_picture(path, width=Cm(15.2))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    doc.paragraphs[-1].paragraph_format.space_before = Pt(6)
                add_para(doc, caption, size=10, italic=True, align="center", space_after=10)
            continue
        if line.startswith("- "):
            add_para(doc, "\u2022\u2002" + line[2:], size=12, align="left",
                     space_after=2, indent=0.9, hanging=0.5)
            continue
        if in_refs and re.match(r'^\[\d+\]', line):
            add_para(doc, line, size=11, align="just", space_after=4,
                     indent=0.75, hanging=0.75)
            continue
        if line.startswith("**Table"):
            add_para(doc, line, size=10.5, bold=False, align="left", space_after=4)
            continue
        add_para(doc, line)
    flush_table(doc, tbl_buf)

    doc.save(out_path)
    print("saved:", out_path)
    print("paragraphs:", len(doc.paragraphs), "| tables:", len(doc.tables))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
