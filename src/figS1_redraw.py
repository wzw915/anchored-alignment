# -*- coding: utf-8 -*-
"""图S1 重绘: 忠实模仿 Pembek et al. (RecSys) Figure 2 的单位圆几何.
c: 冻结内容嵌入(单位范数), d: 可训练 delta(范数<=delta_max), e = c + d.
标注最小余弦相似度 sqrt(1 - delta_max^2); 图注引入 substitution matrix 术语.
输出: figures/figS1_anchor_geometry.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                     "axes.unicode_minus": False, "figure.dpi": 140,
                     "savefig.bbox": "tight", "font.size": 10})

fig, ax = plt.subplots(figsize=(5.6, 5.2))
th = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(th), np.sin(th), color="#9ca3af", lw=1.2, ls="--", zorder=1)

DELTA = 0.5
beta = np.deg2rad(65)                       # c 与 e 的夹角
c = np.array([1.0, 0.0])                    # 单位圆上的锚点
# e 在半径 delta 的"球"上, 取 d 垂直于 e 的最小余弦构型
# 最小相似构型: e 与 d 垂直, cos = sqrt(1-delta^2)
cos_min = np.sqrt(1 - DELTA**2)
# 放置: e 与 c 夹角 = beta (一般情形)
e = np.array([np.cos(beta), np.sin(beta)])
d = e - c
ax.annotate("", xy=e, xytext=c,
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.0), zorder=5)
ax.scatter(*c, s=60, color="#2563eb", zorder=6)
ax.scatter(*e, s=60, color="#dc2626", zorder=6)
ax.annotate("$c_i$ (冻结锚点)", c + np.array([0.03, -0.13]), fontsize=11,
            color="#2563eb")
ax.annotate("$e_i$ (最终表示)", e + np.array([0.03, 0.06]), fontsize=11,
            color="#dc2626")
# delta 球 (以 c 端点为圆心, 半径 delta_max 的虚线圆)
circ = plt.Circle(c, DELTA, fill=False, color="#f59e0b", lw=1.4, ls=":")
ax.add_patch(circ)
ax.annotate("$\\|d_i\\| \\leq \\delta_{\\max}$", c + np.array([-0.06, -DELTA - 0.14]),
            fontsize=11, color="#b45309", ha="center")
# d 标签
mid = c + d / 2
ax.annotate("$d_i$ (可训练残差)", mid + np.array([-0.32, 0.10]), fontsize=11,
            color="#7c2d12")
# 最小余弦构型: d 垂直于 e (在球面上)
d_perp = np.array([-np.sin(beta), np.cos(beta)]) * DELTA
e_min = c + d_perp
ax.annotate("", xy=e_min, xytext=c,
            arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.4), zorder=4)
ax.scatter(*e_min, s=36, color="#f59e0b", zorder=5)
ax.annotate("$e^{\\min}_i$", e_min + np.array([-0.16, -0.06]), fontsize=11,
            color="#b45309")
# 角 beta 弧线
th_arc = np.linspace(0, beta, 50)
ax.plot(0.35 * np.cos(th_arc), 0.35 * np.sin(th_arc), color="#374151", lw=1)
ax.annotate(r"$\theta$", (0.42 * np.cos(beta / 2), 0.42 * np.sin(beta / 2)),
            fontsize=12, color="#374151")
# 公式框
txt = (r"$\mathrm{sim}(c_i, e_i) = \sqrt{1-\delta_{\max}^2\sin^2\theta}$" + "\n"
       + r"$\min_\theta \mathrm{sim} = \sqrt{1-\delta_{\max}^2}$")
ax.text(-0.98, -0.62, txt, fontsize=11, color="#111827",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fef3c7", ec="#f59e0b"))
ax.set_xlim(-1.35, 1.45); ax.set_ylim(-1.15, 1.35)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("锚定微调的几何（沿用 Pembek et al., 2025 单位圆记号）\n"
             "表示被限制在锚点邻域球内；δ_max 直接控制与原始嵌入的最小余弦", fontsize=10)
fig.savefig(os.path.join(PROJ, "figures", "figS1_anchor_geometry.png"), dpi=300)
fig.savefig(os.path.join(PROJ, "figures", "figS1_anchor_geometry.pdf"))
plt.close(fig)
print("figS1 redrawn")
