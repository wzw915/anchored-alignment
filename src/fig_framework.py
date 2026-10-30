# -*- coding: utf-8 -*-
"""课程报告 §2.4 总体技术路线图 (中文, Okabe-Ito 配色)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PROJ, "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "axes.unicode_minus": False,
})

NAVY = "#1F497D"
VERM = "#D55E00"
SKY = "#56B4E9"
GRAY = "#7F7F7F"
GREEN = "#009E73"
LIGHT = "#EDF2F8"

fig, ax = plt.subplots(figsize=(9.2, 11.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13.4)
ax.axis("off")


def box(x, y, w, h, title, lines, face, edge, title_fs=11.5, fs=9.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                       facecolor=face, edgecolor=edge, linewidth=1.4, zorder=2)
    ax.add_patch(b)
    cy = y + h - 0.42
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color=edge, zorder=3)
    for ln in lines:
        cy -= 0.46
        ax.text(x + w / 2, cy, ln, ha="center", va="center", fontsize=fs,
                color="#333333", zorder=3)
    return (x + w / 2, y, y + h)


def arrow(p0, p1, color=GRAY, label=None):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                        color=color, lw=1.6, zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + 0.12, my, label, fontsize=9, color=color, ha="left",
                va="center")


# 层1: 数据层 (两个并列)
b1 = box(0.55, 11.1, 4.1, 1.9, "数据层 Ⅰ：消费者问卷",
         ["图书品类二阶选择", "9,265 个选择对（显式替代）"], LIGHT, NAVY)
b2 = box(5.35, 11.1, 4.1, 1.9, "数据层 Ⅱ：Amazon 行为数据",
         ["图书共购 249 万用户", "游戏共评论 275 万评论者"], LIGHT, NAVY)

# 层2: 分解算法
b3 = box(1.7, 8.6, 6.6, 1.75, "经验分解算法（改进点 1）",
         ["条件转移概率 S_ik = P(第二选择=k | 第一选择=i)",
          "共现图行归一化 → 替代份额矩阵"], LIGHT, VERM)

# 层3: 表示层
b4 = box(1.7, 6.1, 6.6, 1.75, "对比表示学习（改进点 2）",
         ["预训练多模态编码器 → 384 维嵌入（冻结锚点 c_j）",
          "可训练残差 e_j = c_j + Δ_j"], LIGHT, VERM)

# 层4: 优化层
b5 = box(1.7, 3.6, 6.6, 1.75, "范数有界投影梯度优化（改进点 3）",
         ["商品级交叉熵损失（温度化 softmax）",
          "投影约束 ‖Δ_j‖ ≤ δ_max（L2 范数），网格选界仅用训练行"], LIGHT, GREEN)

# 层5: 输出
b6 = box(2.6, 1.55, 4.8, 1.35, "锚定对齐替代矩阵",
         ["零样本商品：Δ = 0，退化为冻结相似度"], "#FDF3E7", VERM)

# 层6: 下游 (两个并列)
b7 = box(0.55, 0.0, 4.1, 1.1, "Bertrand 定价仿真", [], LIGHT, NAVY, fs=9)
b8 = box(5.35, 0.0, 4.1, 1.1, "并购筛查仿真", [], LIGHT, NAVY, fs=9)

arrow((2.6, 11.05), (3.4, 10.42))
arrow((7.4, 11.05), (6.6, 10.42))
arrow((5.0, 8.55), (5.0, 7.92))
arrow((5.0, 6.05), (5.0, 5.42))
arrow((5.0, 3.55), (5.0, 2.97))
arrow((4.0, 1.5), (2.6, 1.15))
arrow((6.0, 1.5), (7.4, 1.15))

fig.savefig(os.path.join(FIG, "fig_course_framework.png"), dpi=300,
            bbox_inches="tight", facecolor="white")
print("saved fig_course_framework.png")
