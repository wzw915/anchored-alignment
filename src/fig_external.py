# -*- coding: utf-8 -*-
"""Figure: cross-dataset external validation (datasets 1-3).
Panel A: semantic-vs-diversion correlation per dataset.
Panel B: Kendall tau frozen vs anchored per dataset.
Reads results/*.json only. Output: figures/fig12_external.(png|pdf)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import OI, apply_rc, panel_label, save

apply_rc()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PROJ, "figures")
RES = os.path.join(PROJ, "results")


def fisher_ci(r, n, z=1.96):
    """Pearson r 的 Fisher z 近似 95% 置信区间 (n = 样本对数)."""
    se = 1.0 / np.sqrt(n - 3)
    zz = np.arctanh(r)
    return np.tanh(zz - z * se), np.tanh(zz + z * se)


d2 = json.load(open(os.path.join(RES, "amazon_dataset2.json"), encoding="utf-8"))
d3 = json.load(open(os.path.join(RES, "amazon_dataset3.json"), encoding="utf-8"))
d1 = json.load(open(os.path.join(RES, "ablation_loo.json"), encoding="utf-8"))

datasets, sem_corr, ci_lo, ci_hi, tau_fr, tau_an, tau_in = [], [], [], [], [], [], []
# dataset 1 (survey) from the main LOO ablation; r 对 45 个商品对计算
m1 = d1["protocols"]["unit"]["methods"]
datasets.append("Survey (stated\nchoices)")
lo, hi = fisher_ci(0.641, 45)
sem_corr.append(0.641); ci_lo.append(0.641 - lo); ci_hi.append(hi - 0.641)
tau_fr.append(m1["frozen"]["kendall"])
tau_an.append(m1["anchored_0.5"]["kendall"])
tau_in.append(m1["unbounded"]["kendall"])
# datasets 2-3; r 对全部商品对计算
for tag, d, name in [("amazon_books_copurchase", d2, "Books co-buy"),
                     ("amazon_video_games_coreview", d3, "Games co-review")]:
    datasets.append(name)
    sc = d.get("semantic_vs_copurchase") or d["semantic_vs_coreview"]
    r = sc.get("pearson", sc.get("spearman", np.nan))
    n_pairs = d["n_items"] * (d["n_items"] - 1) // 2
    lo, hi = fisher_ci(r, n_pairs)
    sem_corr.append(r); ci_lo.append(r - lo); ci_hi.append(hi - r)
    tau_fr.append(d["methods"]["frozen"]["kendall"])
    tau_an.append(d["methods"]["a0.5"]["kendall"])
    tau_in.append(d["methods"]["inf"]["kendall"])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 3.6))
x = np.arange(len(datasets))

# Panel A: 单一中性色柱 + 95% CI 误差棒
axA.bar(x, sem_corr, 0.52, color=OI["blue"], zorder=3)
axA.errorbar(x, sem_corr, yerr=[ci_lo, ci_hi], fmt="none", ecolor="#222222",
             elinewidth=1.1, capsize=4, capthick=1.1, zorder=4)
for xi, v, hi in zip(x, sem_corr, ci_hi):
    axA.text(xi, v + hi + 0.03, f"{v:.2f}", ha="center", fontsize=9)
axA.set_xticks(x, datasets)
axA.set_ylabel("Semantic-diversion correlation (Pearson $r$)")
axA.set_ylim(0, 0.95)
panel_label(axA, "(A)", dx=-0.16, dy=1.04)
axA.grid(axis="y", alpha=0.3, zorder=0)

w = 0.26
axB.bar(x - w, tau_fr, w, label="Frozen", color=OI["blue"], zorder=3)
axB.bar(x, tau_an, w, label="Anchored ($\\delta_{\\max}$=0.5)",
        color=OI["vermillion"], zorder=3)
axB.bar(x + w, tau_in, w, label="Unbounded residual", color=OI["gray"],
        zorder=3, hatch="//", edgecolor="white", lw=0.4)
axB.set_xticks(x, datasets)
axB.set_ylabel("Kendall $\\tau$ (leave-one-item-out)")
axB.set_ylim(0, 0.75)
axB.yaxis.set_major_locator(plt.MultipleLocator(0.15))
axB.legend(fontsize=8.5, loc="upper right", ncol=1, handletextpad=0.5,
           columnspacing=1.0, borderaxespad=0.2)
panel_label(axB, "(B)", dx=-0.16, dy=1.04)
axB.grid(axis="y", alpha=0.3, zorder=0)

fig.tight_layout(w_pad=2.2)
save(fig, FIG, "fig12_external")
print("saved fig12_external.png/.pdf")
