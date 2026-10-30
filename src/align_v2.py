# -*- coding: utf-8 -*-
"""极简对齐 v2 (按老师四步指示)

第一步: 极简/无参数基线 —— 对角度量学习(D 个参数) + 零参数 ZCA 白化
第二步: 个体级对齐 —— InfoNCE, 正样本=(被试第一选,第二选), 负样本=未选书中采 3 本
        训练样本 90 对 -> ~9265 对
第四步: 评估协议 —— LOO 留出产品的 diversion 列预测,
        指标 RMSE + Kendall tau + Top-3 命中; 另附 1..9 本 few-shot 曲线
输出: results/align_v2.json
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
TAU_NCE = 0.1
N_NEG = 3
EPOCHS = 600
LR = 1e-2
WD = 0.01
FEWSHOT_GRID = [1, 2, 3, 5, 7, 9]


def load_pairs():
    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    second = pd.read_csv(os.path.join(CACHE, "second_choice.csv"))
    fc = first[first.choice == 1].set_index("respondent_id")["asin"]
    sc = second[second.choice == 1].set_index("respondent_id")["asin"]
    pair = pd.DataFrame({"a1": fc, "a2": sc}).dropna()
    # 每个被试的未选书集合
    allb = sorted(first["asin"].unique())
    pair = pair.reset_index()
    pair["negs"] = pair.apply(
        lambda r: [b for b in allb if b not in (r.a1, r.a2)], axis=1)
    return pair, allb


def zca_whiten(E, eps=1e-3):
    C = np.cov(E.T) + eps * np.eye(E.shape[1])
    evals, evecs = np.linalg.eigh(C)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    Z = E @ W
    return Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-10, None)


def cosine(E):
    E = E - E.mean(0, keepdims=True)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-10, None)


def sim_matrix(Z, w=None):
    """w: 对角权重(None=普通余弦); Z 已行归一化。"""
    if w is None:
        return Z @ Z.T
    Zw = Z * w
    return Zw @ Z.T


def train_infoNCE(E_norm, pair_train, asin_idx, model="lin", d_out=8,
                  seed=SEED):
    """个体级 InfoNCE。model: 'lin'(384->d_out 投影) 或 'diag'(384 对角权重)。
    返回可用于 sim_matrix 的对象 (Wn 矩阵或 w 向量)。"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if model == "lin":
        P = torch.nn.Parameter(torch.randn(E_norm.shape[1], d_out) * 0.01)
    else:
        P = torch.nn.Parameter(torch.ones(E_norm.shape[1]))
    opt = torch.optim.Adam([P], lr=LR, weight_decay=WD)
    Et = torch.tensor(E_norm, dtype=torch.float32)
    js, ks, ng = [], [], []
    for _, r in pair_train.iterrows():
        negs = list(r["negs"]); rng.shuffle(negs)
        js.append(asin_idx[r.a1]); ks.append(asin_idx[r.a2])
        ng.append([asin_idx[n] for n in negs[:N_NEG]])
    jt = torch.tensor(js); kt = torch.tensor(ks)
    nt = torch.tensor(ng)                                    # (n, N_NEG)
    for ep in range(EPOCHS):
        if model == "lin":
            Zp = F.normalize(Et @ P, dim=1)
        else:
            Zp = Et * P
        S = Zp @ Zp.T
        pos = S[jt, kt].unsqueeze(1)                         # (n,1)
        neg = S[jt.unsqueeze(1).expand(-1, N_NEG), nt]       # (n,N_NEG)
        logits = torch.cat([pos, neg], dim=1) / TAU_NCE
        loss = -F.log_softmax(logits, dim=1)[:, 0].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return P.detach().numpy().copy()


def evaluate_column(pred_col, obs):
    """指标: RMSE, Kendall tau, top1/top3 (obs 最优产品是否在 pred 前 1/3)。"""
    rmse = float(np.sqrt(np.mean((pred_col - obs) ** 2)))
    tau = float(kendalltau(pred_col, obs).statistic) if len(obs) > 2 else np.nan
    order = np.argsort(-pred_col)
    best = int(np.argmax(obs))
    top1 = int(order[0] == best)
    top3 = int(best in order[:3])
    return rmse, tau, top1, top3


def loo_eval(sim_fn, S, J):
    """sim_fn(i0) -> 训练后整个 sim 矩阵的构造方式; 这里统一:
    对每个留出 j0, 用排除 j0 的 pair 训练, 预测列 j0。
    sim_fn: (pair_train) -> callable 得到 10x10 sim (留出品也需嵌入)。"""
    out = []
    for j0 in range(J):
        ptr = pair_all[(pair_all.a1 != asins[j0]) & (pair_all.a2 != asins[j0])]
        Sm = sim_fn(ptr)
        keep = [i for i in range(J) if i != j0]
        for i in keep:
            row = Sm[i].copy() / TAU_EVAL
            row[i] = -1e9
            e = np.exp(row - row.max())
            pc = e / e.sum()
            out.append((j0, i, float(pc[j0]), float(S[i, j0])))
    return out


def metrics_from_loo(rows):
    if not rows:
        return dict(rmse=np.nan, kendall=np.nan, top1=np.nan, top3=np.nan)
    obs = np.array([r[3] for r in rows])
    pred = np.array([r[2] for r in rows])
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    tau = float(kendalltau(pred, obs).statistic)
    # top-k: 对每个 j0, obs 最大的 i 是否在 pred 前 1/3
    top1 = top3 = n = 0
    by_j0 = {}
    for j0, i, p, o in rows:
        by_j0.setdefault(j0, []).append((i, p, o))
    for j0, lst in by_j0.items():
        order = sorted(lst, key=lambda t: -t[1])          # 按预测降序
        best = max(lst, key=lambda t: t[2])[0]            # 实测最优的产品
        rank = [t[0] for t in order].index(best)
        n += 1
        top1 += int(rank == 0)
        top3 += int(rank < 3)
    return dict(rmse=round(rmse, 4), kendall=round(tau, 4),
                top1=round(top1 / n, 3), top3=round(top3 / n, 3))


def main():
    global pair_all, asins, asin_idx, TAU_EVAL
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    asin_idx = {a: i for i, a in enumerate(asins)}
    J = len(asins)
    S = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    E = cosine(np.load(os.path.join(CACHE, "emb", SPEC + ".npy")))
    pair_all, _ = load_pairs()
    TAU_EVAL = 0.3

    results = {}

    # ---- 基线 1: 冻结嵌入 (零参数) ----
    results["frozen"] = metrics_from_loo(loo_eval(
        lambda ptr: E @ E.T, S, J))
    # ---- 基线 2: 均匀 ----
    uni_rows = [(j0, i, 1.0 / (J - 1), S[i, j0])
                for j0 in range(J) for i in range(J) if i != j0]
    results["uniform"] = metrics_from_loo(uni_rows)
    # ---- 基线 3: ZCA 白化 (无监督零参数) ----
    Ez = zca_whiten(np.load(os.path.join(CACHE, "emb", SPEC + ".npy")))
    results["zca"] = metrics_from_loo(loo_eval(lambda ptr: Ez @ Ez.T, S, J))

    # ---- 方法 1: 对角度量学习 (个体级 InfoNCE, 384 参数) ----
    def sim_diag(ptr):
        w = train_infoNCE(E, ptr, asin_idx, model="diag")
        return sim_matrix(E, w)
    results["diag_metric"] = metrics_from_loo(loo_eval(sim_diag, S, J))

    # ---- 方法 2: 线性投影 (个体级 InfoNCE, 384*8 参数) ----
    def sim_lin(ptr):
        W = train_infoNCE(E, ptr, asin_idx, model="lin")
        Zp = cosine(E @ W)
        return Zp @ Zp.T
    results["linear_probing"] = metrics_from_loo(loo_eval(sim_lin, S, J))

    # ---- few-shot 曲线 (线性投影): 训练可见书本数 1..9 ----
    rng = np.random.default_rng(SEED)
    fs = {}
    for m in FEWSHOT_GRID:
        sub = list(rng.choice(asins, size=m, replace=False))
        ptr = pair_all[(pair_all.a1.isin(sub)) & (pair_all.a2.isin(sub))]
        rows = []
        for j0 in range(J):
            ptr2 = ptr[(ptr.a1 != asins[j0]) & (ptr.a2 != asins[j0])]
            if len(ptr2) < 10:
                continue
            W = train_infoNCE(E, ptr2, asin_idx, model="lin", seed=SEED + m)
            Zp = cosine(E @ W)
            Sm = Zp @ Zp.T
            keep = [i for i in range(J) if i != j0]
            for i in keep:
                row = Sm[i].copy() / TAU_EVAL
                row[i] = -1e9
                e = np.exp(row - row.max()); pc = e / e.sum()
                rows.append((j0, i, float(pc[j0]), float(S[i, j0])))
        fs[str(m)] = dict(n_pairs=int(len(ptr)),
                          **metrics_from_loo(rows))
        fs[str(m)] = {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v)
                         else v) for k, v in fs[str(m)].items()}
        print(f"[fewshot m={m}] {fs[str(m)]}", flush=True)

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "align_v2.json"), "w", encoding="utf-8") as f:
        json.dump(dict(results=results, fewshot=fs), f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
