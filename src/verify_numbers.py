# -*- coding: utf-8 -*-
"""verify_numbers.py — 论文数字对账器（答辩现场演示用）

把论文正文与表格里出现的每一个关键数字，重新从 results/*.json 里读出来，
与论文中写的值逐一比对，现场打印对账表。

用法：
    python src/verify_numbers.py          # 全部对账
    python src/verify_numbers.py --brief  # 只看汇总

退出码：全部一致 0，否则 1。
设计约束：只读 results/ 下的结果文件，不训练、不改数据，秒级完成，
因此适合在答辩现场投屏运行。
"""
from __future__ import annotations

import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")

_CACHE: dict[str, object] = {}


def load(name: str):
    if name not in _CACHE:
        with open(os.path.join(RES, name), encoding="utf-8") as f:
            _CACHE[name] = json.load(f)
    return _CACHE[name]


def get(name: str, path: str) -> float:
    """按点号路径取值。键名本身可能含点（如 anchored_0.5、a0.5），
    因此逐层做最长匹配，而不是简单 split。"""
    node = load(name)
    parts = path.split(".")
    i = 0
    while i < len(parts):
        if isinstance(node, list):
            node = node[int(parts[i])]
            i += 1
            continue
        if not isinstance(node, dict):
            raise KeyError(path)
        for j in range(len(parts), i, -1):
            key = ".".join(parts[i:j])
            if key in node:
                node = node[key]
                i = j
                break
        else:
            raise KeyError(".".join(parts[i:]))
    return float(node)


def pad(text: str, width: int) -> str:
    """按显示宽度左对齐（中日韩字符按 2 列计）。"""
    shown = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - shown)


# (论文位置, 指标名, 论文写的值, 结果文件, 字段路径, 小数位)
CHECKS: list[tuple[str, str, float, str, str, int]] = [
    # ---- Table 1 · 留一商品评估（主协议） ----
    ("Table 1", "Frozen · RMSE", 0.0808, "ablation_loo.json", "protocols.unit.methods.frozen.rmse", 4),
    ("Table 1", "Frozen · Kendall τ", 0.4722, "ablation_loo.json", "protocols.unit.methods.frozen.kendall", 4),
    ("Table 1", "Full fine-tuning · RMSE", 0.1225, "ablation_loo.json", "protocols.unit.methods.fullft.rmse", 4),
    ("Table 1", "Full fine-tuning · τ", 0.4333, "ablation_loo.json", "protocols.unit.methods.fullft.kendall", 4),
    ("Table 1", "Unbounded residual · RMSE", 0.0788, "ablation_loo.json", "protocols.unit.methods.unbounded.rmse", 4),
    ("Table 1", "Unbounded residual · τ", 0.6556, "ablation_loo.json", "protocols.unit.methods.unbounded.kendall", 4),
    ("Table 1", "Anchored (δ=0.5) · RMSE", 0.0774, "ablation_loo.json", "protocols.unit.methods.anchored_0.5.rmse", 4),
    ("Table 1", "Anchored (δ=0.5) · τ", 0.6556, "ablation_loo.json", "protocols.unit.methods.anchored_0.5.kendall", 4),
    ("Table 1", "Anchored (δ=0.5) · Top-3", 0.9000, "ablation_loo.json", "protocols.unit.methods.anchored_0.5.top3", 4),
    # ---- 正文第 4.3 节的显著性 ----
    ("§4.3", "Anchored vs Frozen · ΔRMSE", -0.0034, "ablation_loo.json", "protocols.unit.tests.anchored_vs_frozen.d_rmse", 4),
    ("§4.3", "Anchored vs Frozen · p(RMSE)", 0.0352, "ablation_loo.json", "protocols.unit.tests.anchored_vs_frozen.p_rmse", 4),
    ("§4.3", "Anchored vs Frozen · Δτ", 0.1833, "ablation_loo.json", "protocols.unit.tests.anchored_vs_frozen.d_kendall", 4),
    ("§4.3", "Anchored vs Frozen · p(τ)", 0.0039, "ablation_loo.json", "protocols.unit.tests.anchored_vs_frozen.p_kendall", 4),
    ("§4.2", "Full fine-tuning vs Frozen · p(RMSE)", 0.0059, "ablation_loo.json", "protocols.unit.tests.fullft_vs_frozen.p_rmse", 4),
    # ---- Table 2 · 成分拆解（居中协议那一列） ----
    ("Table 2", "Centered Frozen · RMSE", 0.0725, "ablation_loo.json", "protocols.centered.methods.frozen.rmse", 4),
    ("Table 2", "Centered Unbounded · RMSE", 0.1399, "ablation_loo.json", "protocols.centered.methods.unbounded.rmse", 4),
    ("Table 2", "Centered Anchored δ=0.1 · RMSE", 0.0704, "ablation_loo.json", "protocols.centered.methods.anchored_0.1.rmse", 4),
    # ---- 图 2 · 同类余弦相似度 ----
    ("Figure 2", "Frozen · 同类余弦", 0.1845, "tsne_triptych.json", "quant.frozen.same_genre_cos", 4),
    ("Figure 2", "Full fine-tuning · 同类余弦", 0.0396, "tsne_triptych.json", "quant.fullft.same_genre_cos", 4),
    ("Figure 2", "Anchored · 同类余弦", 0.5474, "tsne_triptych.json", "quant.anchored.same_genre_cos", 4),
    ("Figure 2", "Anchored · 与真实替代列相关", 0.7747, "tsne_triptych.json", "quant.anchored.corr_div", 4),
    # ---- Table 3 / Figure 4 · 定价 ----
    ("Table 3", "Frozen · 单卖家平均损失 %", 0.21, "pricing_sim.json", "results.frozen.single.loss_mean_pct", 2),
    ("Table 3", "Full fine-tuning · 单卖家平均损失 %", 25.45, "pricing_sim.json", "results.fullft.single.loss_mean_pct", 2),
    ("Table 3", "Full fine-tuning · 最差单品 %", 35.54, "pricing_sim.json", "results.fullft.single.loss_max_pct", 2),
    ("Table 3", "Anchored · 单卖家平均损失 %", 0.17, "pricing_sim.json", "results.anchored.single.loss_mean_pct", 2),
    ("Table 3", "Frozen · 平台统一定价 %", 0.02, "pricing_sim.json", "results.frozen.platform.loss_pct", 2),
    ("Table 3", "Full fine-tuning · 平台统一定价 %", 1.67, "pricing_sim.json", "results.fullft.platform.loss_pct", 2),
    ("Table 3", "Anchored · 平台统一定价 %", 0.02, "pricing_sim.json", "results.anchored.platform.loss_pct", 2),
    # ---- Table 4 · 并购筛查 ----
    ("Table 4", "Frozen · 涨价预测 MAE (pp)", 4.74, "merger_sim.json", "methods.frozen.mae_pp", 2),
    ("Table 4", "Frozen · 误判笔数", 0, "merger_sim.json", "methods.frozen.decision_errors", 0),
    ("Table 4", "Full fine-tuning · MAE (pp)", 13.04, "merger_sim.json", "methods.fullft.mae_pp", 2),
    ("Table 4", "Full fine-tuning · 误判笔数", 9, "merger_sim.json", "methods.fullft.decision_errors", 0),
    ("Table 4", "Anchored · MAE (pp)", 5.32, "merger_sim.json", "methods.anchored.mae_pp", 2),
    ("Table 4", "Anchored · 误判笔数", 0, "merger_sim.json", "methods.anchored.decision_errors", 0),
    # ---- Table 6 / Figure 5 · 外部验证 ----
    ("Table 6", "Books · Frozen τ", 0.1595, "amazon_dataset2.json", "methods.frozen.kendall", 4),
    ("Table 6", "Books · Anchored τ", 0.2084, "amazon_dataset2.json", "methods.a0.5.kendall", 4),
    ("Table 6", "Books · Unbounded Top-3", 0.2057, "amazon_dataset2.json", "methods.inf.top3", 4),
    ("Table 6", "Books · Frozen Top-3", 0.4286, "amazon_dataset2.json", "methods.frozen.top3", 4),
    ("Table 6", "Video Games · Frozen τ", 0.1989, "amazon_dataset3.json", "methods.frozen.kendall", 4),
    ("Table 6", "Video Games · Anchored τ", 0.2572, "amazon_dataset3.json", "methods.a0.5.kendall", 4),
    # ---- Figure 5 Panel A · 语义-替代相关性 ----
    ("Figure 5A", "Books · 语义 vs 共购 Pearson", 0.2344, "amazon_dataset2.json", "semantic_vs_copurchase.pearson", 4),
    ("Figure 5A", "Video Games · 语义 vs 共评 Pearson", 0.2283, "amazon_dataset3.json", "semantic_vs_coreview.pearson", 4),
]


def main() -> int:
    brief = "--brief" in sys.argv
    ok = 0
    bad: list[tuple] = []
    dwidth = lambda s: sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)
    width = max(dwidth(c[0]) for c in CHECKS) + 2
    lwidth = max(dwidth(c[1]) for c in CHECKS) + 2

    if not brief:
        print()
        print(pad("论文位置", width) + pad("指标", lwidth)
              + f"{'论文写的':>10}{'json 读回':>12}   来源字段")
        print("-" * 132)
    for where, label, paper, fname, path, nd in CHECKS:
        try:
            trace = get(fname, path)
        except (KeyError, IndexError, FileNotFoundError) as exc:
            bad.append((where, label, paper, None, fname, path, repr(exc)))
            print(pad(where, width) + pad(label, lwidth)
                  + f"{paper:>10.{nd}f}{'—':>12}   !! {fname} 缺字段")
            continue
        tol = 0.5 * 10 ** (-nd) + 1e-9
        good = abs(trace - paper) <= tol
        ok += good
        if not good:
            bad.append((where, label, paper, trace, fname, path, ""))
        if not brief:
            mark = "OK " if good else "XX "
            print(mark + pad(where, width - 3) + pad(label, lwidth)
                  + f"{paper:>10.{nd}f}{trace:>12.{nd}f}   {fname}::{path}")

    total = len(CHECKS)
    print()
    print("=" * 132)
    if bad:
        print(f"对账结果：{ok}/{total} 一致，{len(bad)} 处不一致")
        for where, label, paper, trace, fname, path, err in bad:
            print(f"  · [{where}] {label}: 论文 {paper} vs json {trace}  ({fname}::{path}) {err}")
    else:
        print(f"对账结果：{ok}/{total} 全部一致 —— 论文中的每一个数字都能在 results/*.json 里追到出处。")
    print("=" * 132)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
