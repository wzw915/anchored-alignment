# -*- coding: utf-8 -*-
"""md_to_elsarticle.py — 把 deliverable/英文小论文_v2.md 转成 elsarticle LaTeX 投稿稿。

用法：
    python src/md_to_elsarticle.py
输出：
    deliverable/paper_elsarticle.tex

设计要点：
- 引用 [n] → \cite{refn}，文末 [n] 条目 → thebibliography（编号即文件内顺序 = 首现顺序）。
- 四个编号公式硬编码映射到 equation 环境；正文 Eq. (n) → Eq.~\eqref{}。
- 表格 → booktabs 三线表，表题取自表格上方 **Table N.** 行；图片 → figure 环境，
  优先使用同名 PDF 矢量图。
- 行内符号统一转数学模式：δ τ Δ ∞ ≤ R² j₀ 等；源稿零箭头零破折号。
"""
import os
import re

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJ, "deliverable", "英文小论文_v2.md")
OUT = os.path.join(PROJ, "deliverable", "paper_elsarticle.tex")
FIGDIR = os.path.join(PROJ, "figures")

EQS = {
    1: (r"S_{ik} = P(\text{second choice} = k \mid \text{first choice} = i)",
        "eq:diversion"),
    2: (r"z_j = W\, c_j", "eq:map"),
    3: (r"e_j = c_j + \Delta_j, \quad \text{subject to } "
        r"\lVert \Delta_j \rVert \leq \delta_{\max},", "eq:anchor"),
    4: (r"\mathcal{L} \;=\; - \sum_i \sum_{k \neq i} S_{ik} \, "
        r"\log\, \mathrm{softmax}\!\left( e_i^{\top} e_k \,/\, \tau_{\mathrm{train}} \right)_{k},",
        "eq:loss"),
}

PREAMBLE = r"""\documentclass[review,12pt]{elsarticle}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{url}

\journal{Decision Support Systems}

\begin{document}

\begin{frontmatter}
"""

TITLE = ("Anchored Alignment: Escaping the Cold-Start Trap "
         "in Multimodal Demand Estimation")
KEYWORDS = ["cold start", "demand estimation", "multimodal embeddings",
            "representation drift", "substitution patterns", "anchored alignment"]

PHRASES = [
    ("J − 1", r"$J-1$"),
    ("||Δ_j|| ≤ δ_max", r"$\lVert \Delta_j \rVert \leq \delta_{\max}$"),
    ("e_j = c_j + Δ_j with", r"$e_j = c_j + \Delta_j$ with"),
    ("τ_eval", r"$\tau_{\mathrm{eval}}$"),
    ("τ_train", r"$\tau_{\mathrm{train}}$"),
    ("p = ", r"$p$ = "),
    ("p < ", r"$p$ < "),
    ("p > ", r"$p$ > "),
    ("δ_max", r"$\delta_{\max}$"),
    ("S_ik", r"$S_{ik}$"),
    ("e_iᵀ e_k", r"$e_i^{\top} e_k$"),
    ("Δ_j", r"$\Delta_j$"),
    ("e_j", r"$e_j$"),
    ("c_j", r"$c_j$"),
    ("z_j", r"$z_j$"),
    ("e_i", r"$e_i$"),
    ("e_k", r"$e_k$"),
    ("j₀", r"$j_0$"),
    ("R²", r"$R^2$"),
    ("δ", r"$\delta$"),
    ("τ", r"$\tau$"),
    ("Δ", r"$\Delta$"),
    ("∞", r"$\infty$"),
    ("≤", r"$\leq$"),
    ("∈", r"$\in$"),
    ("−", r"$-$"),
]


def inline(s: str) -> str:
    # 哨兵暂存机制: PHRASES 生成的数学片段立即存入 store 并以占位符替换,
    # 后续规则(长短语/短符号/正则)无法再对已生成的数学内容二次加工,
    # 避免 e_j -> $e_j$ 被再次替换成 $$e_j$ 这类不配对输出.
    store = []

    def stash(rep: str) -> str:
        store.append(rep)
        return "\x00%d\x00" % (len(store) - 1)

    for a, b in PHRASES:
        s = s.replace(a, stash(b))
    s = re.sub(r"Eq\. \((\d)\)",
               lambda m: stash(f"Eq.~\\eqref{{{EQS[int(m.group(1))][1]}}}"), s)
    s = re.sub(r"\[(\d+(?:,\s*\d+)*)\]",
               lambda m: stash("\\cite{" + ",".join(
                   "ref" + x.strip() for x in m.group(1).split(",")) + "}"), s)
    # 正文图引用 -> LaTeX 交叉引用 (图注前缀在 figure 块中已剥离, 不会误命中)
    s = re.sub(r"Figure (\w+)",
               lambda m: stash(f"Figure~\\ref{{fig:{m.group(1)}}}"), s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = s.replace("%", r"\%")
    s = re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], s)
    return s


def fig_path(png: str) -> str:
    base = os.path.splitext(os.path.basename(png))[0]
    for ext in (".pdf", ".png"):
        if os.path.exists(os.path.join(FIGDIR, base + ext)):
            return "figures/" + base + ext
    return "figures/" + png


def cell(c: str) -> str:
    return inline(c).replace("|", "$|")


def make_table(buf, caption_md):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in buf]
    header, body = rows[0], rows[2:]
    ncol = len(header)
    spec = []
    for j in range(ncol):
        longest = max([len(r[j]) for r in body] + [len(header[j])] if body
                      else [len(header[j])])
        spec.append("p{4.8cm}" if longest > 34 else ("l" if j == 0 else "r"))
    tex = ["\\begin{table}[htbp]", "\\centering", "\\small",
           "\\caption{" + inline(caption_md) + "}",
           "\\begin{tabular}{" + "".join(spec) + "}"]
    # 表 2 特例：Main protocol / Centered 二级表头（分组 multicolumn）
    if "Ingredient analysis" in caption_md:
        tex += ["\\toprule",
                "\\multicolumn{1}{l}{} & "
                "\\multicolumn{2}{c}{Main protocol} & "
                "\\multicolumn{2}{c}{Centered} \\\\",
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
                " & ".join(cell(h) for h in
                           ["Bound δ_max", "RMSE", "Kendall τ",
                            "RMSE", "Kendall τ"]) + r" \\",
                "\\midrule"]
    else:
        tex += ["\\toprule",
                " & ".join(cell(h) for h in header) + r" \\",
                "\\midrule"]
    tex += [" & ".join(cell(c) for c in r) + r" \\" for r in body]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return tex


def main():
    lines = open(SRC, encoding="utf-8").read().splitlines()
    out, highlights, abstract, refs = [], [], [], []
    pending_cap = None
    stage = "head"                       # head → highlights → abstract → body

    _i = 0
    while _i < len(lines):
        ln = lines[_i]
        _i += 1
        t = ln.strip()

        # ---------------- 前置区 ----------------
        if stage == "head":
            if t.startswith("# "):
                continue                 # 标题用常量
            if t == "**Highlights**":
                stage = "highlights"
                continue
            if t.startswith("**Keywords:**"):
                continue
            if t == "## Abstract":
                stage = "abstract"
                continue
            continue
        if stage == "highlights":
            if t.startswith("- "):
                highlights.append("\\item " + inline(t[2:]))
                continue
            if not t or t.startswith("**Keywords:**"):
                continue
            if t == "## Abstract":
                stage = "abstract"
                continue
            stage = "body"               # 落到正文处理
        if stage == "abstract":
            if t == "## 1. Introduction":
                stage = "body"
            else:
                if t:
                    abstract.append(inline(t))
                continue

        # ---------------- 正文 ----------------
        if t == "## References":
            continue
        if not t:
            continue
        if t.startswith("## "):
            title = re.sub(r"^\d+\.\s+", "", t[3:])
            out.append("\\section{" + inline(title) + "}")
            out.append("")
            continue
        if t.startswith("### "):
            title = re.sub(r"^\d+\.\d+\s+", "", t[4:])
            out.append("\\subsection{" + inline(title) + "}")
            out.append("")
            continue
        if t.startswith("**Table"):
            pending_cap = re.sub(r"^\*\*Table\s+[A-Z]?\d+\.\*\*\s*", "", t)
            continue
        if t.startswith("|"):
            buf = [t]
            while _i < len(lines) and lines[_i].strip().startswith("|"):
                buf.append(lines[_i].strip())
                _i += 1
            out.extend(make_table(buf, pending_cap or ""))
            pending_cap = None
            continue
        m = re.match(r"^!\[(.*)\]\((.*)\)$", t)
        if m:
            cap, path = m.group(1), fig_path(m.group(2))
            fig_no = re.search(r"Figure (\w+)", cap).group(1)
            # \caption 自动产生 "Figure N:" 前缀, 剥离 md 里的 "Figure N." 避免重复
            cap_body = re.sub(r"^Figure\s+\w+\.\s*", "", cap)
            out.extend(["\\begin{figure}[htbp]", "\\centering",
                        f"\\includegraphics[width=\\linewidth]{{{path}}}",
                        f"\\caption{{{inline(cap_body)}}}",
                        f"\\label{{fig:{fig_no}}}",
                        "\\end{figure}", ""])
            continue
        m = re.match(r"^(.*?),?\s+\((\d)\)$", t)
        if m and m.group(1) and any(ch in m.group(1) for ch in "_=Σ+"):
            num = int(m.group(2))
            if num in EQS:
                body, lab = EQS[num]
                out.extend(["\\begin{equation}", body,
                            f"\\label{{{lab}}}", "\\end{equation}", ""])
                continue
        m = re.match(r"^\[(\d+)\]\s+(.*)$", t)
        if m:
            txt = re.sub(r"(https?://\S+)", r"\\url{\1}", m.group(2))
            refs.append((int(m.group(1)), inline(txt)))
            continue
        out.append(inline(t))
        out.append("")

    # ---------------- 组装 ----------------
    tex = [PREAMBLE,
           "\\title{" + TITLE + "}",
           "\\author[aff1]{Author Name\\corref{cor1}}",
           "\\ead{author@example.edu}",
           "\\cortext[cor1]{Corresponding author.}",
           "\\affiliation[aff1]{organization={Department, University},"
           " city={City}, country={Country}}",
           ""]
    if highlights:
        tex += ["\\begin{highlights}"] + highlights + ["\\end{highlights}", ""]
    tex += ["\\begin{abstract}"] + abstract + ["\\end{abstract}", ""]
    tex += ["\\begin{keyword}", " \\sep ".join(KEYWORDS),
            "\\end{keyword}", "\\end{frontmatter}", ""]
    tex += out
    tex.append("\\begin{thebibliography}{37}")
    for n, txt in sorted(refs):
        tex.append(f"\\bibitem{{ref{n}}} {txt}")
    tex += ["\\end{thebibliography}", "", "\\end{document}", ""]

    open(OUT, "w", encoding="utf-8").write("\n".join(tex))
    print(f"written {OUT}  ({len(tex)} lines, refs={len(refs)}, "
          f"highlights={len(highlights)}, tables="
          f"{sum(1 for x in out if x == '\\\\begin{table}[htbp]')}, "
          f"figs={sum(1 for x in out if x.startswith('\\\\includegraphics'))})")


if __name__ == "__main__":
    main()
