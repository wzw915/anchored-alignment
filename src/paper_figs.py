# -*- coding: utf-8 -*-
"""论文初稿配图: 图S1 锚定几何示意 (模仿 RecSys 图2) + 图S2 delta 鲁棒性曲线 (模仿 RecSys 图3)."""
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

# ---------- 图S1: 锚定几何 ----------
fig, ax = plt.subplots(figsize=(5.6, 4.4))
C_ANCH, C_RES = OI["blue"], OI["vermillion"]
th = np.linspace(-0.4, 2.2, 200)
R = 1.0
# 超球面 (部分弧)
ax.plot(R*np.cos(th), R*np.sin(th), color="#8a8a8a", lw=1.0, ls="--")
# 三个锚点 c_j 在球面上
angles = [0.35, 1.15, 1.85]
rng = np.random.default_rng(7)
for k, a in enumerate(angles):
    c = np.array([R*np.cos(a), R*np.sin(a)])
    ax.scatter(*c, s=40, color=C_ANCH, zorder=5)
    ax.annotate(f"$c_{k+1}$", c + np.array([0.05, 0.05]), fontsize=11, color=C_ANCH)
    # 有界残差 delta: 随机方向, 模长 < delta_max, 终点仍在球内(近似)
    d_dir = a + rng.uniform(-0.45, 0.45)
    dlen = rng.uniform(0.10, 0.22)
    d = np.array([dlen*np.cos(d_dir), dlen*np.sin(d_dir)])
    e = c + d
    ax.annotate("", xy=e, xytext=c,
                arrowprops=dict(arrowstyle="->", color=C_RES, lw=1.5))
    ax.scatter(*e, s=30, color=C_RES, zorder=5)
    ax.annotate(f"$e_{{{k+1}}}$", e + np.array([0.04, -0.09]), fontsize=11, color=C_RES)
    # 残差球
    circ = plt.Circle(c, 0.22, fill=True, color=C_RES, alpha=0.07, zorder=1)
    ax.add_patch(circ)
# delta_max 界标注
c0 = np.array([R*np.cos(0.35), R*np.sin(0.35)])
ax.annotate(r"$\|\Delta_j\|_2 \leq \delta_{\max}$", (c0[0]-0.02, c0[1]-0.42),
            fontsize=12, color=C_RES, ha="center")
# 完全微调的"逃逸"箭头
a = 1.15
c = np.array([R*np.cos(a), R*np.sin(a)])
far = c * 1.55 + np.array([0.25, -0.15])
ax.annotate("", xy=far, xytext=c,
            arrowprops=dict(arrowstyle="->", color="#9a9a9a", lw=1.3, ls=":"))
ax.annotate("Full fine-tuning:\ndrifts off the manifold", far + np.array([0.03, 0.0]),
            fontsize=9, color="#5a5a5a")
ax.annotate("Frozen embedding $c_j$ (pretrained prior)", (0.62, -0.35), fontsize=9.5, color=C_ANCH)
ax.annotate("Anchored representation $e_j = c_j + \\Delta_j$", (0.30, 0.72), fontsize=9.5, color=C_RES)
ax.set_xlim(-0.55, 2.15); ax.set_ylim(-0.75, 1.65)
ax.set_aspect("equal"); ax.axis("off")
fig.savefig(os.path.join(FIG, "figS1_anchor_geometry.png"), dpi=300)
fig.savefig(os.path.join(FIG, "figS1_anchor_geometry.pdf"))
plt.close(fig)

# ---------- 图S2: delta 鲁棒性（双协议 6 线版，数据源 ablation_loo.json）----------
abl = json.load(open(os.path.join(PROJ, "results", "ablation_loo.json"), encoding="utf-8"))
deltas = [0.1, 0.3, 0.5, 0.7]

def curve(proto, meth, key):
    m = abl["protocols"][proto]["methods"]
    if meth == "anchored":
        return [m[f"anchored_{d:.1f}"][key] for d in deltas]
    return [m[meth][key]] * len(deltas)

# Okabe-Ito 色盲安全 + 线型/marker 双重编码 (黑白打印可读):
# 主协议=暖色系, 居中协议=冷色系; anchored=实线圆/方, frozen=虚线, unbounded=点线
STYLES = [
    ("unit", "anchored",  dict(color=OI["vermillion"], ls="-",  marker="o", lw=1.7, ms=4.5,
                               label="Anchored, main protocol")),
    ("centered", "anchored", dict(color=OI["orange"], ls="-", marker="s", lw=1.7, ms=4.5,
                                  label="Anchored, centered protocol")),
    ("unit", "frozen",    dict(color=OI["blue"],       ls="--", marker="^", lw=1.3, ms=4.5,
                               label="Frozen, main protocol")),
    ("centered", "frozen", dict(color=OI["sky"],      ls="--", marker="v", lw=1.3, ms=4.5,
                                label="Frozen, centered protocol")),
    ("unit", "unbounded", dict(color="#3a3a3a",        ls=":",  marker="D", lw=1.5, ms=4,
                               label="Unbounded residual, main protocol")),
    ("centered", "unbounded", dict(color=OI["gray"],  ls="-.", marker="d", lw=1.5, ms=4,
                                   label="Unbounded residual, centered protocol")),
]
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))

for pi, (ax, key, ylab) in enumerate(((axes[0], "rmse", "RMSE (lower is better)"),
                                      (axes[1], "kendall", "Kendall $\\tau$ (higher is better)"))):
    for proto, meth, st in STYLES:
        ax.plot(deltas, curve(proto, meth, key), **st)
    ax.axvspan(0.3, 0.7, color="#E8C33C", alpha=0.16, lw=0)
    ax.set_xlabel(r"Residual bound $\delta_{\max}$")
    ax.set_ylabel(ylab)
    ax.set_xticks(deltas)
    ax.grid(alpha=0.25, lw=0.5)
    panel_label(ax, f"({'AB'[pi]})", dx=-0.13, dy=1.05)

axes[0].set_ylim(0.06, 0.15)
axes[1].set_ylim(0.1, 0.75)
handles = [plt.Line2D([], [], color=st["color"], ls=st["ls"], marker=st["marker"],
                      lw=st["lw"], ms=st["ms"] - 0.5, label=st["label"])
           for _, _, st in STYLES]
fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=2,
           frameon=False, columnspacing=1.4, handletextpad=0.6,
           bbox_to_anchor=(0.5, -0.10))
fig.tight_layout(rect=(0, 0.18, 1, 1))
save(fig, FIG, "figS2_delta_robustness")
plt.close(fig)
print("figs done")
