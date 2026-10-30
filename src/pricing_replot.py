# -*- coding: utf-8 -*-
"""重绘 fig11_profit_loss: Y 轴改对数刻度, 解决小柱不可见问题.

背景: 0.02%-0.21% 的柱与 25.45% 主结果跨 3 个数量级, 线性轴下小柱
高度接近 0 像素. 对数刻度 (0.01 -> 40) 下每根柱均清晰可辨, 且保留
25.45% 的量级优势 (跨约 1.2 个 decade). 经济学期刊处理跨数量级
利润损失的常规做法.

只读 results/pricing_sim.json, 不重跑仿真.
"""
import os, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
from fig_style import OI, apply_rc, save


def main():
    with open(os.path.join(RES, "pricing_sim.json"), encoding="utf-8") as f:
        out = json.load(f)
    rows = out["results"]

    apply_rc()
    disp = {"frozen": "Frozen embeddings",
            "fullft": "Full fine-tuning\n(individual InfoNCE)",
            "anchored": "Anchored alignment\n(ours, delta = 0.5)"}
    labels = [disp[m] for m in rows]
    l_plat = [rows[m]["platform"]["loss_pct"] for m in rows]
    l_single = [rows[m]["single"]["loss_mean_pct"] for m in rows]

    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    b1 = ax.bar(x - w / 2, l_single, w, label="Single-seller pricing",
                color=OI["wine"], zorder=3)
    b2 = ax.bar(x + w / 2, l_plat, w, label="Platform-uniform pricing",
                color=OI["lblue"], zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(0.01, 45)               # 0.02% 距轴底仍有 2 倍高度, 可见
    # 主刻度只留 decade: 0.01 / 0.1 / 1 / 10, 次刻度线弱化作网格
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(FixedLocator(
        [0.02, 0.03, 0.05, 0.2, 0.3, 0.5, 2, 3, 5, 20, 30]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", alpha=0.35, zorder=0)
    ax.grid(axis="y", which="minor", alpha=0.12, zorder=0)

    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            emph = v > 20     # 25.45% 主结果柱：加粗放大，一眼可见
            ax.annotate(f"{v:.2f}%",
                        (b.get_x() + b.get_width() / 2, v * 1.25),
                        ha="center", fontsize=12 if emph else 10.5,
                        fontweight="bold" if emph else "normal",
                        color=OI["wine"] if emph else "#333333",
                        zorder=4)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Profit loss vs. true-optimal pricing (%)  [log scale]")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, FIG, "fig11_profit_loss")
    plt.close(fig)
    print("replotted:", os.path.join(FIG, "fig11_profit_loss.png"))


if __name__ == "__main__":
    main()
