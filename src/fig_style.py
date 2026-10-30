# -*- coding: utf-8 -*-
"""统一期刊图样式: Okabe-Ito 色盲友好调色板 + DSS/Elsevier 图面规范.

规范要点 (DSS 投稿要求):
- 字体 Arial 8-10pt, 嵌入 PDF (pdf.fonttype 42);
- 矢量输出 PDF, PNG 300dpi 仅用于 docx/预览;
- 面板标签 (A)(B)(C) 加粗置于左上角;
- 黑白打印下仍可通过线型/marker 区分.
"""
import os
import matplotlib as mpl

# Okabe-Ito 色盲安全调色板
OI = dict(
    black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
    yellow="#F0E442", blue="#0072B2", vermillion="#D55E00",
    purple="#CC79A7", gray="#7F7F7F",
    navy="#1F497D",        # 深海军蓝 (品类一)
    slate="#5A6572",       # 暗板灰 (品类二)
    wine="#8C3041",        # 深酒红 (定价图主场景)
    lblue="#AFCFE8",       # 浅蓝 (定价图次场景)
)

ROLE = dict(
    ours="vermillion",     # anchored alignment (本文方法)
    baseline="blue",       # frozen embeddings
    unbounded="gray",      # unconstrained residual
    alt="orange",          # 第二协议 / 对照
    alt2="sky",
)


def apply_rc():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,   # 字体嵌入, 可编辑
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.30, "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "figure.dpi": 140,
    })


def panel_label(ax, s, dx=-0.10, dy=1.04):
    """加粗面板标签 (A)/(B)/(C), 置于轴左上角外侧."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom")


def save(fig, figdir, stem):
    """矢量 PDF (投稿) + 300dpi PNG (docx/预览)."""
    fig.savefig(os.path.join(figdir, stem + ".pdf"))
    fig.savefig(os.path.join(figdir, stem + ".png"), dpi=300)
