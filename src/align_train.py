# -*- coding: utf-8 -*-
"""需求对齐训练 · 投影头（方案第二周模块）

思路: 学一个低秩线性投影 W (d->8), 使投影后余弦相似度的逐行 softmax
逼近实测 diversion 矩阵 (行交叉熵)。评估 = 逐列留一 CV:
  - 训练时遮住被留出产品的 diversion 行与列
  - 用训练好的 W 预测"转向该留出产品"的份额, 与实测对比
  - 基线 = 冻结嵌入(同样的 softmax, tau 在训练行上调) + 均匀分配
输出: results/align_train.json
"""
import os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
SPEC = "reviews_ST_chunkmean"       # 与主线最优规格一致
D_OUT = 8                            # 低秩投影维度
EPOCHS = 800
WD = 0.05
SEED = 2026
TAUS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def cosine(E):
    E = E - E.mean(0, keepdims=True)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-10, None)


def softmax_row(logits, idx_keep, idx_alt, tau):
    """对每个 i: 在备选集 idx_alt(排除自身)上 softmax, 返回 {i: {k: 份额}}。"""
    z = torch.tensor(logits) / tau
    out = {}
    for i in idx_keep:
        cols = [k for k in idx_alt if k != i]
        p = F.softmax(z[i, cols], dim=0)
        out[i] = {k: float(p[n]) for n, k in enumerate(cols)}
    return out


def col_pred(pred, keep, j0):
    return np.array([pred[i][j0] for i in keep])


def train_rows_obs(S, keep):
    """训练行的归一化实测矩阵 (去掉自身列)。"""
    return np.array([[S[i, k] / S[i, [k for k in keep if k != i]].sum()
                      for k in keep if k != i] for i in keep])


def train_projection(E, S, mask_rowcol):
    """在遮住 mask_rowcol 行/列的 diversion 上训练投影 W, 返回 (W, tau)。"""
    keep = [i for i in range(len(S)) if i != mask_rowcol]
    Et = torch.tensor(E, dtype=torch.float32)
    St = torch.tensor(S, dtype=torch.float32)
    torch.manual_seed(SEED)
    W = (torch.randn(E.shape[1], D_OUT) * 0.01)
    W.requires_grad_(True)
    opt = torch.optim.Adam([W], lr=5e-3, weight_decay=WD)
    for ep in range(EPOCHS):
        Z = F.normalize(Et @ W, dim=1)
        logits = Z @ Z.T
        loss = 0.0
        for i in keep:
            cols = [k for k in keep if k != i]
            tgt = St[i, cols] / St[i, cols].sum()
            loss = loss - (tgt * F.log_softmax(logits[i, cols], dim=0)).sum()
        loss = loss / len(keep)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Z = F.normalize(Et @ W, dim=1).numpy()
    # tau 在训练行上调 (不含留出列)
    best_tau, best_rmse = None, 1e9
    for tau in TAUS:
        pred = softmax_row(Z @ Z.T, keep, keep, tau)
        obs = train_rows_obs(S, keep)
        obs_arr = np.array([[obs[a][n] for n in range(len(keep) - 1)]
                            for a in range(len(keep))])
        pred_arr = np.array([[pred[i][k] for k in keep if k != i] for i in keep])
        rm = float(np.sqrt(np.mean((pred_arr - obs_arr) ** 2)))
        if rm < best_rmse:
            best_tau, best_rmse = tau, rm
    return Z, best_tau


def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    J = len(asins)
    S = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    E = cosine(np.load(os.path.join(CACHE, "emb", SPEC + ".npy")))

    folds = []
    for j0 in range(J):
        keep = [i for i in range(J) if i != j0]
        Z, tau = train_projection(E, S, j0)
        # 评估: 预测列 j0 (转向留出产品), 行为其余 9 个产品
        pred = softmax_row(Z @ Z.T, keep, list(range(J)), tau)
        obs = np.array([S[i, j0] for i in keep])
        pred_col = col_pred(pred, keep, j0)
        rmse_al = float(np.sqrt(np.mean((pred_col - obs) ** 2)))

        # 冻结基线: 原始嵌入, tau 在训练行上调
        best_tau_f, best_rmse_f = None, 1e9
        obs_tr = train_rows_obs(S, keep)
        obs_arr = np.array([[obs_tr[a][n] for n in range(len(keep) - 1)]
                            for a in range(len(keep))])
        for tau in TAUS:
            p_tr = softmax_row(E @ E.T, keep, keep, tau)
            pred_arr = np.array([[p_tr[i][k] for k in keep if k != i] for i in keep])
            rm = float(np.sqrt(np.mean((pred_arr - obs_arr) ** 2)))
            if rm < best_rmse_f:
                best_tau_f, best_rmse_f = tau, rm
        pred_f = softmax_row(E @ E.T, keep, list(range(J)), best_tau_f)
        pred_f_col = col_pred(pred_f, keep, j0)
        rmse_fr = float(np.sqrt(np.mean((pred_f_col - obs) ** 2)))

        uni = np.full(len(keep), 1.0 / (J - 1))
        rmse_uni = float(np.sqrt(np.mean((uni - obs) ** 2)))
        folds.append(dict(heldout=asins[j0], tau=tau, tau_frozen=best_tau_f,
                          rmse_aligned=round(rmse_al, 4),
                          rmse_frozen=round(rmse_fr, 4),
                          rmse_uniform=round(rmse_uni, 4)))
        print(f"[fold {j0}] {asins[j0]}: aligned={rmse_al:.4f} "
              f"frozen={rmse_fr:.4f} uniform={rmse_uni:.4f}", flush=True)

    agg = {k: round(float(np.mean([f[k] for f in folds])), 4)
           for k in ["rmse_aligned", "rmse_frozen", "rmse_uniform"]}
    # 全量训练后的对齐相似度 vs 实测 diversion 相关 (诊断对照, 有过拟合嫌疑需标注)
    Zfull, _ = train_projection(E, S, mask_rowcol=-1 if False else J)  # 无留出
    # mask_rowcol=J 等价于不遮任何行 (keep = 全部)
    Sal = Zfull @ Zfull.T
    from scipy.stats import pearsonr
    off = ~np.eye(J, dtype=bool)
    r_al = float(pearsonr(Sal[off], S[off])[0])
    r_fr = float(pearsonr((E @ E.T)[off], S[off])[0])

    out = dict(spec=SPEC, d_out=D_OUT, folds=folds, aggregate=agg,
               corr_full=dict(aligned=round(r_al, 4), frozen=round(r_fr, 4),
                              note="全量训练后的相关含过拟合风险, 以 LOO RMSE 为准"))
    with open(os.path.join(RES, "align_train.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(aggregate=agg, corr_full=out["corr_full"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
