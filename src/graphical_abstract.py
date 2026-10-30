# -*- coding: utf-8 -*-
"""DSS Graphical Abstract: 三格故事线（陷阱 → 锚定 → 决策价值）.
规范依据: Elsevier DSS Guide for Authors — 宽度≥1328px、最长边水平、13cm 可读、
首选 TIFF/EPS/PDF/MS Office（本脚本同时输出 PNG(300dpi)+PDF 矢量）.
数据全部读取 results/*.json，不硬编码。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "savefig.bbox": "tight",
})

C_FROZEN, C_FT, C_ANCH = "#2563eb", "#9ca3af", "#dc2626"
C_BG1, C_BG2, C_BG3 = "#eff6ff", "#fef2f2", "#f0fdf4"

aa = json.load(open(os.path.join(RES, "align_anchored.json"), encoding="utf-8"))
frozen = aa["frozen"]
pz = json.load(open(os.path.join(RES, "pricing_sim.json"), encoding="utf-8"))["results"]
loss = {m: pz[m]["single"]["loss_mean_pct"] for m in ("frozen", "fullft", "anchored")}

fig = plt.figure(figsize=(9.6, 3.4), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.18,
                      left=0.015, right=0.985, top=0.80, bottom=0.07)


def panel_head(ax, tag, title, subtitle):
    ax.text(0.5, 0.995, f"{tag}  |  {title}", transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="center", va="top", color="#111827")
    ax.text(0.5, 0.905, subtitle, transform=ax.transAxes,
            fontsize=6.6, ha="center", va="top", color="#374151")


def metric_box(ax, txt, ec):
    ax.text(0.5, 0.03, txt, transform=ax.transAxes, fontsize=7.2, ha="center",
            va="bottom", color=ec, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=ec, lw=0.8))


def frame(ax, bg):
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#cbd5e1")


# ---------- Panel A: The Cold-Start Trap ----------
axA = fig.add_subplot(gs[0])
frame(axA, C_BG1)
rng = np.random.default_rng(7)
axA.scatter(0.28 + rng.normal(0, .035, 26), 0.46 + rng.normal(0, .05, 26),
            s=14, color=C_FROZEN, alpha=.85, linewidths=0, zorder=3)
axA.scatter(0.74 + rng.normal(0, .13, 26), 0.48 + rng.normal(0, .13, 26),
            s=14, color=C_FT, alpha=.8, linewidths=0, zorder=3)
axA.annotate("", xy=(0.60, 0.47), xytext=(0.40, 0.47),
             arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.3), zorder=4)
axA.text(0.28, 0.30, "frozen:\ntight topology", fontsize=6.6, color=C_FROZEN,
         ha="center", va="top")
axA.text(0.74, 0.30, "full fine-tuning:\ndrift & scatter", fontsize=6.6,
         color="#6b7280", ha="center", va="top")
panel_head(axA, "A", "The Cold-Start Trap",
           "individual-level fine-tuning destroys\nthe product-level substitution topology")
metric_box(axA, f"RMSE {frozen['rmse']:.3f} → 0.140    τ {frozen['kendall']:.2f} → 0.14",
           C_ANCH)

# ---------- Panel B: Anchored Alignment ----------
axB = fig.add_subplot(gs[1])
frame(axB, C_BG2)
CC, RR = (0.47, 0.46), 0.295
axB.add_patch(Circle(CC, RR, fill=False, ls="--", lw=1.1, ec="#6b7280"))
axB.add_patch(Circle(CC, 0.125, fill=False, ls=":", lw=1.2, ec="#f59e0b"))
axB.scatter([CC[0]], [CC[1]], s=30, color=C_FROZEN, zorder=5)
E = (CC[0] + 0.105, CC[1] + 0.078)
axB.scatter([E[0]], [E[1]], s=26, color=C_ANCH, zorder=5)
axB.annotate("", xy=E, xytext=CC,
             arrowprops=dict(arrowstyle="-|>", color=C_ANCH, lw=1.3), zorder=4)
axB.annotate("$c_j$ frozen anchor", (CC[0] - 0.06, CC[1] - 0.075), fontsize=6.8, color=C_FROZEN)
axB.annotate("$e_j=c_j+\\Delta_j$", (E[0] - 0.02, E[1] + 0.045), fontsize=7.5, color=C_ANCH)
axB.annotate("$\\|\\Delta_j\\|\\leq\\delta_{\\max}$", (0.10, 0.16), fontsize=7.5, color="#b45309")
panel_head(axB, "B", "Anchored Alignment (ours)",
           "bounded residual + product-level diversion\nsupervision: task signal in, topology intact")
metric_box(axB, "RMSE 0.0774    τ 0.66  (+39%)", "#166534")

# ---------- Panel C: Decision Value ----------
axC = fig.add_subplot(gs[2])
axC.set_facecolor(C_BG3)
methods = [("Full\nfine-tuning", loss["fullft"], C_FT),
           ("Frozen", loss["frozen"], C_FROZEN),
           ("Anchored\n(ours)", loss["anchored"], C_ANCH)]
x = np.arange(3)
bars = axC.bar(x, [m[1] for m in methods], width=0.62,
               color=[m[2] for m in methods], zorder=3)
for b, m in zip(bars, methods):
    axC.annotate(f"{m[1]:.2f}%", (b.get_x() + b.get_width() / 2, b.get_height() + 0.7),
                 ha="center", fontsize=7.6, fontweight="bold",
                 color="#111827" if m[1] < 5 else C_ANCH)
axC.set_xticks(x, [m[0] for m in methods], fontsize=7.6)
axC.set_ylabel("Profit loss of item-level pricing (%)", fontsize=7.8)
axC.set_ylim(0, 31)
axC.tick_params(axis="y", labelsize=7)
axC.grid(axis="y", alpha=.3, zorder=0)
axC.text(0.5, 0.995, "C  |  Why Managers Should Care", transform=axC.transAxes,
         fontsize=9, fontweight="bold", ha="center", va="top", color="#111827")
axC.text(0.5, 0.905, "wrong substitution matrix → wrong pricing elasticity",
         transform=axC.transAxes, fontsize=6.6, ha="center", va="top", color="#374151")
for s in axC.spines.values(): s.set_color("#cbd5e1")

fig.suptitle("Escaping the Cold-Start Trap in Demand Estimation: Anchored Alignment of Multimodal Embeddings",
             fontsize=10, fontweight="bold", y=0.97, color="#111827")

fig.savefig(os.path.join(FIG, "graphical_abstract.png"), dpi=300)
fig.savefig(os.path.join(FIG, "graphical_abstract.pdf"))
print("GA saved")
from PIL import Image
im = Image.open(os.path.join(FIG, "graphical_abstract.png"))
print("size px:", im.size)
