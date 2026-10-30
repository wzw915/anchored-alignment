# -*- coding: utf-8 -*-
"""Figure 1 (paper): overview of the anchored-alignment pipeline.

期刊版总体框架图 (对应 DSS/SESSIM 的 "Overview of the framework")：
左→右流程，输入层 → 三个建模组件 → 输出 → 下游决策。
Arial 8-9pt, Okabe-Ito 配色, 矢量 PDF + 300dpi PNG.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from fig_style import OI, apply_rc

apply_rc()
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PROJ, "figures")

NAVY, VERM, GREEN, GRAY = OI["navy"], OI["vermillion"], OI["green"], OI["gray"]
LIGHT, WARM, COOL = "#F2F6FA", "#FDF1E8", "#EAF4EF"
INK = "#333333"

fig = plt.figure(figsize=(6.9, 3.2))
ax = fig.add_axes([0, 0, 1, 1])          # 坐标轴铺满画布, 单位距离可控
ax.set_xlim(0, 17)
ax.set_ylim(0, 7.6)
ax.axis("off")
IU = 6.9 / 17.0                           # 英寸/单位 (横向)


def box(x, y, w, h, title, lines, edge, face=LIGHT, tfs=9.0, fs=7.6,
        dashed=False, tcolor=None, gap=0.50):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=face, edgecolor=edge, linewidth=1.2 if dashed else 1.3,
        linestyle=(0, (3, 2)) if dashed else "-", zorder=2))
    cy = y + h - 0.42
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=tfs,
            fontweight="bold", color=tcolor or edge, zorder=3,
            linespacing=1.15)
    for ln in lines:
        if ln == "":
            cy -= 0.30
            continue
        cy -= gap
        ax.text(x + w / 2, cy, ln, ha="center", va="center", fontsize=fs,
                color=INK, zorder=3)


def arrow(p0, p1, color=GRAY, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, color=color,
        lw=1.2, linestyle=ls, zorder=1,
        connectionstyle=f"arc3,rad={rad}"))


# ---- 列标题 ----
for xc, txt in [(1.55, "Input data"), (7.20, "Model building"),
                (12.80, "Output"), (15.70, "Decision support")]:
    ax.text(xc, 7.32, txt, ha="center", va="center", fontsize=8.0,
            fontweight="bold", color="#4D4D4D")

# ---- 输入层 ----
box(0.15, 4.55, 2.80, 2.25, "Stated-choice\nsurvey",
    ["10 book items", "9,265 pairwise", "choices"],
    NAVY, face=LIGHT, fs=7.4, tfs=8.5)
box(0.15, 1.45, 2.80, 2.65, "Amazon\nbehavioural proxies",
    ["books co-purchase", "(2.49 M buyers)", "games co-review",
     "(2.75 M reviewers)"], NAVY, face=LIGHT, fs=7.2, tfs=8.5)

# ---- 建模容器 ----
ax.add_patch(FancyBboxPatch(
    (3.35, 1.15), 7.70, 5.90, boxstyle="round,pad=0.05,rounding_size=0.10",
    facecolor="white", edgecolor="#BFBFBF", linewidth=1.0,
    linestyle=(0, (4, 3)), zorder=0))

# ---- 三个组件 ----
box(3.55, 5.35, 7.30, 1.55, "(1)  Empirical decomposition",
    ["$S_{ik} = P(\\mathrm{2nd} = k \\mid \\mathrm{1st} = i)$",
     "row-normalized co-occurrence for proxies"],
    VERM, face=WARM, fs=7.5)
box(3.55, 3.35, 7.30, 1.55, "(2)  Multimodal representation",
    ["384-d pretrained encoder, frozen anchor $c_j$",
     "trainable residual $e_j = c_j + \\Delta_j$"],
    VERM, face=WARM, fs=7.5)
box(3.55, 1.35, 7.30, 1.55, "(3)  Norm-bounded optimization",
    ["product-level InfoNCE ($\\tau_{\\mathrm{train}}$)",
     "project $\\Delta_j$ onto $\\|\\Delta_j\\| \\leq \\delta_{\\max}$"],
    GREEN, face=COOL, fs=7.5)

# ---- 输出与下游 ----
box(11.45, 1.30, 2.70, 2.40, "Anchored\ndiversion matrix",
    ["bounded residual rows", "LOO evaluation",
     "cold start: $\\Delta_j = 0$", "$\\Rightarrow$ frozen similarity"],
    VERM, face=WARM, fs=7.2, tfs=8.5, gap=0.42)
box(11.45, 0.15, 2.70, 0.90, "Day-0 product\nno training, no data", [],
    GRAY, face="#F7F7F7", dashed=True, tfs=7.6)

box(14.50, 2.55, 2.35, 1.50, "Bertrand-Nash\npricing", [],
    NAVY, face=LIGHT, fs=7.4, tfs=8.5)
box(14.50, 0.85, 2.35, 1.50, "Merger\nscreening", [],
    NAVY, face=LIGHT, fs=7.4, tfs=8.5)

# ---- 箭头 ----
arrow((2.97, 5.75), (3.51, 6.15))                     # survey -> (1)
arrow((2.97, 3.55), (3.51, 5.55), rad=0.15, ls="--")  # amazon -> (1)
arrow((5.20, 5.33), (5.20, 4.92))                     # (1) -> (2)
arrow((5.20, 3.33), (5.20, 2.92))                     # (2) -> (3)
arrow((10.87, 2.30), (11.42, 2.30))                   # (3) -> output
arrow((14.17, 3.05), (14.47, 3.25))                   # output -> pricing
arrow((14.17, 1.90), (14.47, 1.70))                   # output -> merger
arrow((12.80, 1.07), (12.80, 1.28), ls="--")          # day-0 -> output
# 间隙内的过渡标签 (放在两箱之间的空白带上)
ax.text(7.90, 5.13, "supervision $S$", fontsize=7.0, color=GRAY,
        ha="center", va="center", style="italic")
ax.text(7.90, 3.13, "residual $e_j$", fontsize=7.0, color=GRAY,
        ha="center", va="center", style="italic")

# ---- 底部记号说明 ----
ax.text(0.15, 0.10,
        "$c_j$ frozen anchor;  $\\Delta_j$ bounded residual;  "
        "$e_j = c_j + \\Delta_j$;  $S$ row-normalized diversion matrix",
        fontsize=7.0, color="#666666", ha="left", va="bottom")

fig_out = os.path.join(FIG, "fig01_framework")
fig.savefig(fig_out + ".pdf", facecolor="white")
fig.savefig(fig_out + ".png", dpi=300, facecolor="white")
plt.close(fig)
print("saved fig01_framework.pdf/.png")
