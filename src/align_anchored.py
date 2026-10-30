# -*- coding: utf-8 -*-
"""锚定微调 (Anchored Fine-tuning): 有界残差 + 产品级 diversion 监督

z_j = normalize(c_j + Delta_j),  ||Delta_j||_2 <= delta_max (训练中投影回球面)
损失: 估计书行 softmax 交叉熵 vs 实测 diversion 行 (产品级, 非个体级)
评估: LOO —— 留出书不带 Delta (零样本冻结), 估计书带 Delta,
      预测转向留出书的份额列, 与实测对比 (RMSE / Kendall / Top-3)
delta_max 网格 {0.1, 0.3, 0.5, 0.7}; 每折用训练行选 delta (诚实协议)
输出: results/align_anchored.json
"""
import os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
SPEC = "reviews_ST_chunkmean"
SEED = 2026
EPOCHS = 800
LR = 1e-2
DELTAS = [0.1, 0.3, 0.5, 0.7]
TAU_EVAL = 0.3


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-10, None)


def train_delta(E, S, keep, delta_max):
    """为 keep 中的书训练有界 Delta, 目标=keep 行 x keep 列 diversion。"""
    torch.manual_seed(SEED)
    n = len(keep)
    P = torch.nn.Parameter(torch.zeros(n, E.shape[1]))
    opt = torch.optim.Adam([P], lr=LR)
    Ct = torch.tensor(E[keep], dtype=torch.float32)
    St = torch.tensor(S[np.ix_(keep, keep)], dtype=torch.float32)
    idx = torch.arange(n)
    for ep in range(EPOCHS):
        Z = F.normalize(Ct + P, dim=1)
        logits = (Z @ Z.T) / TAU_EVAL
        logp = F.log_softmax(logits, dim=1)
        tgt = St / St.sum(1, keepdims=True).clamp_min(1e-10)
        loss = -(tgt * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                       # 投影回 delta_max 球
            norms = P.norm(dim=1, keepdim=True).clamp_min(1e-12)
            P.mul_(torch.clamp(norms, max=delta_max) / norms)
    with torch.no_grad():
        return F.normalize(Ct + P, dim=1).numpy()


def metrics(pred_col, obs):
    rmse = float(np.sqrt(np.mean((pred_col - obs) ** 2)))
    tau = float(kendalltau(pred_col, obs).statistic)
    order = np.argsort(-pred_col)
    best = int(np.argmax(obs))
    return rmse, tau, int(order[0] == best), int(best in order[:3])


def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    J = len(asins)
    S = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    E = unit(np.load(os.path.join(CACHE, "emb", SPEC + ".npy")))

    # 每折训练行上调 tau_eval 与 delta (协议: 训练行 RMSE 最小者)
    per_delta = {d: [] for d in DELTAS}
    pick = []                                    # 每折按训练行选的 delta
    for j0 in range(J):
        keep = [i for i in range(J) if i != j0]
        obs = np.array([S[i, j0] for i in keep])
        best = None
        for d in DELTAS:
            Zk = train_delta(E, S, keep, d)
            # 评估: 估计书(带 Delta) x 全部书(留出书冻结)
            Zfull = E.copy()
            Zfull[keep] = Zk
            Sm = Zfull @ Zfull.T
            pred = np.array([row_softmax(Sm[i], i, j0) for i in keep])
            rmse, tau, t1, t3 = metrics(pred, obs)
            per_delta[d].append((rmse, tau, t1, t3))
            # 训练行 RMSE (keep 行 x keep 列, 去自身) —— 用于选 delta
            tr_obs = S[np.ix_(keep, keep)]
            tr_scores = []
            for a, i in enumerate(keep):
                row = Sm[i, keep] / TAU_EVAL
                row[a] = -1e9
                e = np.exp(row - row.max())
                p = e / e.sum()
                tgt = tr_obs[a] / tr_obs[a].sum()
                tr_scores.append(np.sqrt(np.mean((p - tgt) ** 2)))
            tr_rmse = float(np.mean(tr_scores))
            if best is None or tr_rmse < best[0]:
                best = (tr_rmse, d, (rmse, tau, t1, t3))
        pick.append(best)
    agg = {d: dict(rmse=round(float(np.mean([r[0] for r in rows])), 4),
                   kendall=round(float(np.mean([r[1] for r in rows])), 4),
                   top1=round(float(np.mean([r[2] for r in rows])), 3),
                   top3=round(float(np.mean([r[3] for r in rows])), 3))
           for d, rows in per_delta.items()}
    sel = dict(rmse=round(float(np.mean([p[2][0] for p in pick])), 4),
               deltas_chosen=[p[1] for p in pick])
    # 基线
    frozen_rows = []
    for j0 in range(J):
        keep = [i for i in range(J) if i != j0]
        obs = np.array([S[i, j0] for i in keep])
        Sm = E @ E.T
        pred = np.array([row_softmax(Sm[i], i, j0) for i in keep])
        frozen_rows.append(metrics(pred, obs))
    frozen = dict(rmse=round(float(np.mean([r[0] for r in frozen_rows])), 4),
                  kendall=round(float(np.mean([r[1] for r in frozen_rows])), 4),
                  top1=round(float(np.mean([r[2] for r in frozen_rows])), 3),
                  top3=round(float(np.mean([r[3] for r in frozen_rows])), 3))

    out = dict(spec=SPEC, tau_eval=TAU_EVAL, per_delta=agg,
               train_selected=sel, frozen=frozen)
    with open(os.path.join(RES, "align_anchored.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def row_softmax(row, self_i, target_j):
    r = row.copy() / TAU_EVAL
    r[self_i] = -1e9
    e = np.exp(r - r.max())
    return e[target_j] / e.sum()


if __name__ == "__main__":
    sys.exit(main())
