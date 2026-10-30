# -*- coding: utf-8 -*-
"""跨类目对齐训练的规模效应模拟（align_train 负结果的翻正实验）

设定: 合成 M 个类目, 每类 J=8 产品。
  - 需求潜变量 b_j 决定 diversion 真值: s_jk ~ softmax(b_j·b_k / tau_b)
  - 语义潜变量 a_j = b_j + eta * noise (与需求相关但不相同, 复现"语义≠替代")
  - 嵌入 z_j = a_j + sigma * noise (384 维中信号维度 d_a=16)
训练: 线性投影 384->8, 在 M_train 个类目上池化训练 (行 softmax 交叉熵);
评估: 留出 M_test 个类目, 预测 diversion 矩阵 RMSE, 对比冻结嵌入与均匀。
预期: M_train 小时 aligned 输 frozen (复现 J=10 结论), M_train 增大后反超。
输出: results/align_scaling.json
"""
import os, json, sys
import numpy as np
import torch
import torch.nn.functional as F

RES = os.path.join(PROJ, "results") if (PROJ := os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))) else "results"

D_Z = 64            # 嵌入维度
D_A = 16            # 语义维度 (含需求维度 + 风格干扰)
D_B = 2             # 需求维度 (diversion 真值只由此决定)
D_OUT = 8
J = 8
ETA = 0.0           # 语义-需求偏离 (语义含 14 维风格干扰已足够)
SIGMA = 1.0         # 嵌入噪声
TAU_B = 0.3
EPOCHS = 3000
SEED = 2026
M_GRID = [1, 2, 5, 10, 20, 40]
M_TEST = 20
N_REP = 3           # 每个M重复次数


def make_category(rng):
    """需求潜变量 = 语义潜变量的前 D_B 维 (风格维度与 diversion 无关)。"""
    a = rng.normal(size=(J, D_A))
    b = a[:, :D_B]
    Z = np.zeros((J, D_Z))
    Z[:, :D_A] = a
    Z[:, D_A:] = SIGMA * rng.normal(size=(J, D_Z - D_A))
    logits = (b @ b.T) / (np.linalg.norm(b, axis=1, keepdims=True)
                          * np.linalg.norm(b, axis=1).reshape(-1, 1))
    logits = logits / TAU_B
    np.fill_diagonal(logits, -1e9)
    e = np.exp(logits - logits.max(1, keepdims=True))
    S = e / e.sum(1, keepdims=True)
    return Z, S


def train_pooled(Zs, Ss, seed):
    torch.manual_seed(seed)
    W = (torch.randn(D_Z, D_OUT) * 0.01)
    W.requires_grad_(True)
    opt = torch.optim.Adam([W], lr=0.02, weight_decay=0.001)
    Zt = [torch.tensor(Z, dtype=torch.float32) for Z in Zs]
    St = [torch.tensor(S, dtype=torch.float32) for S in Ss]
    for ep in range(EPOCHS):
        loss = 0.0
        n = 0
        for Z, S in zip(Zt, St):
            P = F.normalize(Z @ W, dim=1)
            logits = P @ P.T
            mask = ~torch.eye(J, dtype=torch.bool)
            loss = loss - (S[mask].view(J, J - 1)
                           * F.log_softmax(logits[mask].view(J, J - 1),
                                           dim=1)).sum()
            n += J
        loss = loss / n
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return [F.normalize(Z @ W, dim=1).numpy() for Z in Zt]


def row_softmax(mat, tau):
    e = np.exp(mat / tau)
    np.fill_diagonal(e, 0.0)
    return e / np.clip(e.sum(1, keepdims=True), 1e-10, None)


def rmse_pred(P, S):
    off = ~np.eye(J, dtype=bool)
    return float(np.sqrt(np.mean((P - S)[off] ** 2)))


def main():
    rng0 = np.random.default_rng(SEED)
    out = []
    for M_train in M_GRID:
        for rep in range(N_REP):
            rng = np.random.default_rng(SEED + 1000 * M_train + rep)
            cats_tr = [make_category(rng) for _ in range(M_train)]
            cats_te = [make_category(rng) for _ in range(M_TEST)]
            # tau 调参/训练共用训练类目; 用最后一个训练类目做内部调 tau 简化
            Ztr = [c[0] for c in cats_tr]
            Str = [c[1] for c in cats_tr]
            Pal = train_pooled(Ztr, Str, SEED + rep)
            # tau 网格在训练集上选
            best = (None, 1e9, None)
            for tau in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
                rm_a = np.mean([rmse_pred(row_softmax(P, tau), S)
                                for P, S in zip(Pal, Str)])
                if rm_a < best[1]:
                    best = (tau, rm_a, "aligned")
            tau_al = best[0]
            best_f = (None, 1e9)
            for tau in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
                rm_f = np.mean([rmse_pred(row_softmax(
                    c[0] @ c[0].T, tau), c[1]) for c in cats_tr])
                if rm_f < best_f[1]:
                    best_f = (tau, rm_f)
            tau_fr = best_f[0]
            rm_al = float(np.mean([rmse_pred(row_softmax(P, tau_al), S)
                                   for P, (_, S) in zip(Pal, cats_te)]))
            rm_fr = float(np.mean([rmse_pred(row_softmax(
                Z @ Z.T, tau_fr), S) for Z, S in cats_te]))
            rm_uni = float(np.mean([rmse_pred(
                np.full((J, J), 1.0 / (J - 1)) - np.eye(J) / (J - 1), S)
                for _, S in cats_te]))
            out.append(dict(M_train=M_train, rep=rep,
                            rmse_aligned=round(rm_al, 4),
                            rmse_frozen=round(rm_fr, 4),
                            rmse_uniform=round(rm_uni, 4)))
            print(f"M={M_train} rep={rep}: aligned={rm_al:.4f} "
                  f"frozen={rm_fr:.4f} uniform={rm_uni:.4f}", flush=True)
    agg = {}
    for M in M_GRID:
        rows = [r for r in out if r["M_train"] == M]
        agg[str(M)] = {k: round(float(np.mean([r[k] for r in rows])), 4)
                       for k in ["rmse_aligned", "rmse_frozen", "rmse_uniform"]}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "align_scaling.json"), "w", encoding="utf-8") as f:
        json.dump(dict(per_run=out, aggregate=agg), f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    sys.exit(main())
