# -*- coding: utf-8 -*-
"""商业定价模拟 (Pricing Simulation) —— DSS 审稿人视角的决策损失实验

三个 diversion 矩阵 (Frozen / Full-FT / Anchored) 各自驱动定价优化,
用真实 (经验) diversion 矩阵评估利润 -> 利润损失 % (vs 真实最优定价).

需求系统 (固定替代结构 Luce 流模型, diversion 为原始项):
  顾客类型 i (质量 q0_i) 在产品间按权重重新分配:
    w_ij = D[i,j] * exp(-beta * dp_j)   (j != i)
    w_ii = rho * exp(-beta * dp_i),  rho = sigma0/(1-sigma0)
  q_j(P) = sum_i q0_i w_ij / sum_k w_ik;  pi = sum_j (p_j - c_j) q_j
  sigma0=0.6 保证 p=p0 时需求近似基线份额.
定价: 逐产品坐标上升 (网格 0.6x-1.4x p0, 12 轮) 用候选 D 优化;
评估: 价格向量代入真实 D (diversion_empirical) 计算真实利润.

输出: results/pricing_sim.json + figures/fig11_profit_loss.png
"""
import os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_v2 import load_pairs, train_infoNCE          # 个体级 InfoNCE
from align_anchored import train_delta                    # 锚定残差

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
SPEC = "reviews_ST_chunkmean"
SEED = 2026
TAU_EVAL = 0.3
DELTA = 0.5                      # 论文主结果的稳定平台中点
SIGMA0 = 0.6                     # 基线留存率
OUTSIDE = 0.3                    # 外部选项权重 (涨价可流失出品类)
BETAS = [0.25, 0.5, 1.0]         # 价格敏感度鲁棒性 (每 $1)
MARGIN = 0.4                     # 毛利率 (成本 = 60% 价格)
GRID = np.linspace(0.6, 1.4, 81) # 相对 p0 的价格网格
SWEEPS = 12


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-10, None)


def col_softmax(Sim, i, j0):
    """产品 i 行 softmax 去自身后, 转向 j0 的份额 (与评估协议一致)."""
    r = Sim[i].copy() / TAU_EVAL
    r[i] = -1e9
    e = np.exp(r - r.max())
    return e[j0] / e.sum()


def assemble(method, E, S, pair, asins, asin_idx):
    """按 LOO 协议组装 10x10 预测 diversion 矩阵 (列 j0 来自排除 j0 的训练折)."""
    J = len(asins)
    P = np.full((J, J), np.nan)
    for j0 in range(J):
        keep = [i for i in range(J) if i != j0]
        if method == "frozen":
            Zfull = E
        elif method == "fullft":
            ptr = pair[(pair.a1 != asins[j0]) & (pair.a2 != asins[j0])]
            W = train_infoNCE(E, ptr, asin_idx)          # 384->8 线性投影
            Zfull = unit(E @ W)
        elif method == "anchored":
            Zk = train_delta(E, S, keep, DELTA)
            Zfull = E.copy(); Zfull[keep] = Zk
        Sim = Zfull @ Zfull.T
        for i in keep:
            P[i, j0] = col_softmax(Sim, i, j0)
    # 对角 = 留存概率 = 1 - 离开概率
    off = P.copy(); np.fill_diagonal(off, 0.0)
    diag = np.clip(1.0 - off.sum(1), 0.02, 1.0)
    np.fill_diagonal(P, diag)
    return P


def demand(P0, q0, D, p, p0, beta):
    """固定替代结构 Luce 流模型 + 外部选项 (涨价流失出品类)."""
    dp = p - p0
    e = np.exp(-beta * dp)
    q = np.zeros(len(q0))
    for i in range(len(q0)):
        w = np.append(D[i] * e, OUTSIDE)             # 内部 J 项 + 外部
        w[i] = (SIGMA0 / (1 - SIGMA0)) * e[i]        # 自身留存权重
        q += q0[i] * w[:len(q0)] / w.sum()
    return q


def optimize_prices(D, q0, p0, c, beta):
    """坐标上升: 用候选 D 最大化平台总利润, 返回价格向量."""
    p = p0.copy()
    J = len(p0)
    for _ in range(SWEEPS):
        changed = False
        for j in range(J):
            best_p, best_pi = p[j], None
            for g in GRID:
                pj = p0[j] * g
                pp = p.copy(); pp[j] = pj
                q = demand(p0, q0, D, pp, p0, beta)
                pi = float(((pp - c) * q).sum())
                if best_pi is None or pi > best_pi:
                    best_pi, best_p = pi, pj
            if abs(best_p - p[j]) > 1e-6:
                p[j] = best_p; changed = True
        if not changed:
            break
    return p


def optimize_single(D, q0, p0, c, beta):
    """独立卖家: 每本书的卖家只卖该书, 各自定最优价 (其余保持 p0).
    返回每个产品的 (最优价, 单品利润)."""
    J = len(p0)
    prices, profits = [], []
    for j in range(J):
        best_p, best_pi = p0[j], None
        for g in GRID:
            pj = p0[j] * g
            pp = p0.copy(); pp[j] = pj
            q = demand(p0, q0, D, pp, p0, beta)
            pi = (pj - c[j]) * q[j]
            if best_pi is None or pi > best_pi:
                best_pi, best_p = pi, pj
        prices.append(best_p)
        profits.append(float(best_pi))
    return np.array(prices), np.array(profits)


def single_true_profit(pv, q0, p0, c, beta, D_true, j):
    """真实 D 下, 单品 j 定价 pv 的真实利润."""
    pp = p0.copy(); pp[j] = pv
    q = demand(p0, q0, D_true, pp, p0, beta)
    return float((pv - c[j]) * q[j])


def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    asin_idx = {a: i for i, a in enumerate(asins)}
    J = len(asins)
    S_true = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    S_true = S_true / S_true.sum(1, keepdims=True)       # 行归一
    E = unit(np.load(os.path.join(CACHE, "emb", SPEC + ".npy")))
    pair, _ = load_pairs()

    # 基线份额 = 第一选市场份额
    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    fc = first[first.choice == 1]["asin"].value_counts(normalize=True)
    q0 = np.array([fc.get(a, 0.0) for a in asins]); q0 = q0 / q0.sum()

    rng = np.random.default_rng(SEED)
    p0 = rng.uniform(11.99, 19.99, size=J).round(2)      # 图书基线价
    c = p0 * (1 - MARGIN)                                # 成本

    mats, rows = {}, {}
    p_opt_plat = optimize_prices(S_true, q0, p0, c, BETAS[1])
    pi_plat = float(((p_opt_plat - c) *
                     demand(p0, q0, S_true, p_opt_plat, p0, BETAS[1])).sum())
    for method in ["frozen", "fullft", "anchored"]:
        D = assemble(method, E, S_true, pair, asins, asin_idx)
        np.save(os.path.join(RES, f"div_pred_{method}.npy"), D)
        mats[method] = D
        # ---- 场景 1: 平台统一定价 ----
        p_m = optimize_prices(D, q0, p0, c, BETAS[1])
        pi_m = float(((p_m - c) *
                      demand(p0, q0, S_true, p_m, p0, BETAS[1])).sum())
        by_beta = {}
        for b in BETAS:
            pb = optimize_prices(S_true, q0, p0, c, b)
            pi_b = float(((pb - c) * demand(p0, q0, S_true, pb, p0, b)).sum())
            pmb = optimize_prices(D, q0, p0, c, b)
            pi_mb = float(((pmb - c) *
                           demand(p0, q0, S_true, pmb, p0, b)).sum())
            by_beta[str(b)] = dict(loss_pct=round(100 * (1 - pi_mb / pi_b), 2),
                                   profit_opt=round(pi_b, 2))
        # ---- 场景 2: 独立卖家单品定价 (主场景) ----
        p_single, _ = optimize_single(D, q0, p0, c, BETAS[1])
        true_single, _ = optimize_single(S_true, q0, p0, c, BETAS[1])
        pi_true = np.array([
            single_true_profit(true_single[j], q0, p0, c, BETAS[1], S_true, j)
            for j in range(J)])
        pi_wrong = np.array([
            single_true_profit(p_single[j], q0, p0, c, BETAS[1], S_true, j)
            for j in range(J)])
        loss_each = 100 * (1 - pi_wrong / np.clip(pi_true, 1e-9, None))
        sb = {}
        for b in BETAS:
            ts_b, _ = optimize_single(S_true, q0, p0, c, b)
            ms_b, _ = optimize_single(D, q0, p0, c, b)
            pit = np.array([single_true_profit(ts_b[j], q0, p0, c, b,
                                               S_true, j) for j in range(J)])
            piw = np.array([single_true_profit(ms_b[j], q0, p0, c, b,
                                               S_true, j) for j in range(J)])
            sb[str(b)] = round(float(np.mean(100 * (1 - piw /
                                                    np.clip(pit, 1e-9, None)))), 2)
        rows[method] = dict(
            rmse_vs_true=round(float(np.sqrt(np.mean(
                (D - S_true)[~np.eye(J, dtype=bool)] ** 2))), 4),
            platform=dict(profit=round(pi_m, 2), profit_opt=round(pi_plat, 2),
                          loss_pct=round(100 * (1 - pi_m / pi_plat), 2),
                          avg_price=round(float(np.mean(p_m)), 2),
                          by_beta=by_beta),
            single=dict(loss_mean_pct=round(float(loss_each.mean()), 2),
                        loss_max_pct=round(float(loss_each.max()), 2),
                        loss_each=[round(float(x), 2) for x in loss_each],
                        avg_price=round(float(np.mean(p_single)), 2),
                        by_beta=sb))

    # 基准: 沿用基线价 p0 不优化
    pi_p0 = float(((p0 - c) * demand(p0, q0, S_true, p0, p0, BETAS[1])).sum())
    out = dict(spec=SPEC, delta=DELTA, beta_base=BETAS[1], margin=MARGIN,
               sigma0=SIGMA0, outside=OUTSIDE,
               profit_baseline_price=round(pi_p0, 2),
               prices_p0=p0.tolist(), results=rows)
    with open(os.path.join(RES, "pricing_sim.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 图: 利润损失条形
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fig_style import OI, apply_rc, save
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
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            emph = v > 20     # 25.45% 主结果柱：加粗放大，一眼可见
            ax.annotate(f"{v:.2f}%",
                        (b.get_x() + b.get_width() / 2, v + 0.15),
                        ha="center", fontsize=12 if emph else 10.5,
                        fontweight="bold" if emph else "normal",
                        color=OI["wine"] if emph else "#333333")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Profit loss vs. true-optimal pricing (%)")
    ax.set_ylim(0, max(max(l_single), max(l_plat)) * 1.18)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    save(fig, FIG, "fig11_profit_loss")
    plt.close(fig)

    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
