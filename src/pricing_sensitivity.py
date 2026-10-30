# -*- coding: utf-8 -*-
"""定价模拟的两轴敏感性：外部选项权重 × 毛利率（盲审 M6）。

论文 headline 的利润损失依赖三个外生设定：价格敏感 β=0.5（原实验已扫）、
外部选项权重 OUTSIDE=0.3、毛利率 MARGIN=0.4。本脚本固定 β=0.5，对

    OUTSIDE ∈ {0.1, 0.3, 0.5} × MARGIN ∈ {0.3, 0.4, 0.5}

共 9 个组合重跑"单卖主定价"主场景，检查 headline（锚定 25.5%、
微调更差、冻结居中）的排序是否依赖这些选择。

与 pricing_sim.py 的区别：直接加载已保存的 results/div_pred_*.npy，
不再重训 10 折 InfoNCE/残差，全脚本几十秒。

输出: results/pricing_sensitivity.json（不覆盖 pricing_sim.json）
"""
import os, json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pricing_sim import demand, optimize_single, single_true_profit

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
SEED = 2026
BETA = 0.5
SIGMA0 = 0.6
OUTSIDES = [0.1, 0.3, 0.5]
MARGINS = [0.3, 0.4, 0.5]

import pricing_sim as PS   # 复用其常量 GRID/SWEEPS 与函数


def run_grid():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    J = len(asins)
    S_true = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    S_true = S_true / S_true.sum(1, keepdims=True)
    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    fc = first[first.choice == 1]["asin"].value_counts(normalize=True)
    q0 = np.array([fc.get(a, 0.0) for a in asins]); q0 = q0 / q0.sum()
    rng = np.random.default_rng(SEED)
    p0 = rng.uniform(11.99, 19.99, size=J).round(2)

    mats = {}
    for m in ["frozen", "fullft", "anchored"]:
        f = os.path.join(RES, f"div_pred_{m}.npy")
        if not os.path.exists(f):
            sys.exit(f"缺少 {f}；先跑一次 pricing_sim.py 生成预测矩阵")
        mats[m] = np.load(f)

    grid = {}
    for out_w in OUTSIDES:
        PS.OUTSIDE = out_w          # demand() 读取模块级常量
        for mg in MARGINS:
            c = p0 * (1 - mg)
            row = {}
            for m, D in mats.items():
                p_m, _ = optimize_single(D, q0, p0, c, BETA)
                p_t, _ = optimize_single(S_true, q0, p0, c, BETA)
                pi_t = np.array([single_true_profit(p_t[j], q0, p0, c, BETA,
                                                    S_true, j) for j in range(J)])
                pi_w = np.array([single_true_profit(p_m[j], q0, p0, c, BETA,
                                                    S_true, j) for j in range(J)])
                row[m] = round(float(np.mean(100 * (1 - pi_w /
                                                    np.clip(pi_t, 1e-9, None)))), 2)
            grid[f"outside={out_w}|margin={mg}"] = row
            print(f"  outside={out_w} margin={mg}: {row}", flush=True)
    return grid


def main():
    # 真实 diversion 一并重跑（作为每个格子的最优基准在函数内处理）
    grid = run_grid()
    # headline 排序检查：每个格子里 loss 的排序是否保持 anchored < frozen < fullft
    rank_ok = all(
        (v["anchored"] < v["frozen"] < v["fullft"]) or
        (v["anchored"] <= v["frozen"] and v["frozen"] <= v["fullft"])
        for v in grid.values())
    out = dict(beta=BETA, sigma0=SIGMA0,
               outside_grid=OUTSIDES, margin_grid=MARGINS,
               scenario="single-seller",
               loss_pct=grid, rank_stable=bool(rank_ok))
    json.dump(out, open(os.path.join(RES, "pricing_sensitivity.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n排序 anchored < frozen < fullft 在所有 9 个格子成立: {rank_ok}")
    print("写入 results/pricing_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
