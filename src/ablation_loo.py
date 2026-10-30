# -*- coding: utf-8 -*-
"""LOO 协议下的两项补充实验 + 不确定性量化

回应盲审意见：
  M1 表 2 的 FullFT 行在 LOO 协议下没有原始结果 → 重跑个体级 InfoNCE（lin 投影）
  M4 缺「无界残差 + 产品级损失」消融 → δ_max = inf，不投影
  M3 无不确定性量化 → 每折指标均值±SD + 逐折配对置换检验（2^10 全枚举）

输出 results/ablation_loo.json
"""
import os, sys, json, itertools
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
TAU_EVAL = 0.3
TAU_NCE = 0.07
N_NEG = 3
WD = 1e-4
EPOCHS_NCE = 600
EPOCHS_PROD = 800
LR = 1e-2


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-10, None)


def cosine(E):
    E = E - E.mean(0, keepdims=True)
    return unit(E)


def load_pairs():
    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    second = pd.read_csv(os.path.join(CACHE, "second_choice.csv"))
    fc = first[first.choice == 1].set_index("respondent_id")["asin"]
    sc = second[second.choice == 1].set_index("respondent_id")["asin"]
    pair = pd.DataFrame({"a1": fc, "a2": sc}).dropna().reset_index()
    allb = sorted(first["asin"].unique())
    pair["negs"] = pair.apply(lambda r: [b for b in allb if b not in (r.a1, r.a2)], axis=1)
    return pair


def train_infoNCE(E_norm, pair_train, asin_idx, d_out=8, seed=SEED):
    """个体级 InfoNCE + 线性投影 (384 -> d_out)，与 align_v2 的 'lin' 同构。"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    P = torch.nn.Parameter(torch.randn(E_norm.shape[1], d_out) * 0.01)
    opt = torch.optim.Adam([P], lr=LR, weight_decay=WD)
    Et = torch.tensor(E_norm, dtype=torch.float32)
    js, ks, ng = [], [], []
    for _, r in pair_train.iterrows():
        negs = list(r["negs"]); rng.shuffle(negs)
        js.append(asin_idx[r.a1]); ks.append(asin_idx[r.a2])
        ng.append([asin_idx[n] for n in negs[:N_NEG]])
    jt, kt, nt = torch.tensor(js), torch.tensor(ks), torch.tensor(ng)
    for _ in range(EPOCHS_NCE):
        Zp = F.normalize(Et @ P, dim=1)
        Sim = Zp @ Zp.T
        pos = Sim[jt, kt].unsqueeze(1)
        neg = Sim[jt.unsqueeze(1).expand(-1, N_NEG), nt]
        logits = torch.cat([pos, neg], dim=1) / TAU_NCE
        loss = -F.log_softmax(logits, dim=1)[:, 0].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return F.normalize(Et @ P, dim=1).numpy()


def train_residual(E, S, keep, delta_max):
    """产品级监督残差；delta_max = inf 时为无界残差（消融 M4）。"""
    torch.manual_seed(SEED)
    n = len(keep)
    P = torch.nn.Parameter(torch.zeros(n, E.shape[1]))
    opt = torch.optim.Adam([P], lr=LR)
    Ct = torch.tensor(E[keep], dtype=torch.float32)
    St = torch.tensor(S[np.ix_(keep, keep)], dtype=torch.float32)
    for _ in range(EPOCHS_PROD):
        Z = F.normalize(Ct + P, dim=1)
        logits = (Z @ Z.T) / TAU_EVAL
        logp = F.log_softmax(logits, dim=1)
        tgt = St / St.sum(1, keepdims=True).clamp_min(1e-10)
        loss = -(tgt * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if np.isfinite(delta_max):
            with torch.no_grad():
                norms = P.norm(dim=1, keepdim=True).clamp_min(1e-12)
                P.mul_(torch.clamp(norms, max=delta_max) / norms)
    with torch.no_grad():
        return F.normalize(Ct + P, dim=1).numpy()


def col_metrics(pred, obs):
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    tau = float(kendalltau(pred, obs).statistic)
    order = np.argsort(-pred)
    best = int(np.argmax(obs))
    return dict(rmse=rmse, kendall=tau,
                top1=int(order[0] == best), top3=int(best in order[:3]))


def predict_column(Z, keep, j0):
    Sm = Z @ Z.T
    pred = []
    for i in keep:
        r = Sm[i] / TAU_EVAL
        r[i] = -1e9
        e = np.exp(r - r.max())
        pred.append(float(e[j0] / e.sum()))
    return np.array(pred)


def perm_paired(a, b, n_iter=None):
    """逐折配对置换检验（符号翻转全枚举，2^10）。返回双尾 p。"""
    d = np.array(a) - np.array(b)
    obs = abs(d.mean())
    k = len(d)
    cnt = tot = 0
    for signs in itertools.product([1, -1], repeat=k):
        tot += 1
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def summarize(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    J = len(asins)
    asin_idx = {a: i for i, a in enumerate(asins)}
    S = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    Eraw = np.load(os.path.join(CACHE, "emb", SPEC + ".npy"))
    Ec = cosine(Eraw)
    pair_all = load_pairs()

    per_fold = {k: [] for k in ["frozen", "fullft", "unbounded", "anchored"]}
    for j0 in range(J):
        keep = [i for i in range(J) if i != j0]
        obs = np.array([S[i, j0] for i in keep])

        # 1) frozen
        per_fold["frozen"].append(col_metrics(predict_column(Ec, keep, j0), obs))

        # 2) FullFT: 个体级 InfoNCE + 线性投影（LOO：训练对排除留出书）
        ptr = pair_all[(pair_all.a1 != asins[j0]) & (pair_all.a2 != asins[j0])]
        W = train_infoNCE(unit(Eraw), ptr, asin_idx)
        Zf = W.copy()
        per_fold["fullft"].append(col_metrics(predict_column(Zf, keep, j0), obs))

        # 3) unbounded residual + product-level loss
        Zu = Ec.copy(); Zu[keep] = train_residual(Ec, S, keep, np.inf)
        per_fold["unbounded"].append(col_metrics(predict_column(Zu, keep, j0), obs))

        # 4) anchored δ=0.5
        Za = Ec.copy(); Za[keep] = train_residual(Ec, S, keep, 0.5)
        per_fold["anchored"].append(col_metrics(predict_column(Za, keep, j0), obs))
        print(f"fold {j0} done", flush=True)

    out = {"spec": SPEC, "tau_eval": TAU_EVAL, "n_folds": J, "methods": {}}
    for m, rows in per_fold.items():
        entry = {}
        for key in ["rmse", "kendall", "top1", "top3"]:
            mu, sd = summarize(rows, key)
            entry[key] = round(mu, 4)
            entry[key + "_sd"] = round(sd, 4)
        entry["per_fold"] = [{k: round(v, 4) if isinstance(v, float) else v
                              for k, v in r.items()} for r in rows]
        out["methods"][m] = entry

    # 配对置换检验：anchored vs frozen / anchored vs unbounded / fullft vs frozen
    def col(m, key):
        return [r[key] for r in per_fold[m]]
    tests = {}
    for name, (x, y) in {
        "anchored_vs_frozen": ("anchored", "frozen"),
        "anchored_vs_unbounded": ("anchored", "unbounded"),
        "fullft_vs_frozen": ("fullft", "frozen"),
    }.items():
        tests[name] = {
            "d_rmse": round(float(np.mean(col(x, "rmse")) - np.mean(col(y, "rmse"))), 4),
            "p_rmse": round(perm_paired(col(x, "rmse"), col(y, "rmse")), 4),
            "d_kendall": round(float(np.mean(col(x, "kendall")) - np.mean(col(y, "kendall"))), 4),
            "p_kendall": round(perm_paired(col(x, "kendall"), col(y, "kendall")), 4),
        }
    out["paired_permutation"] = tests

    with open(os.path.join(RES, "ablation_loo.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_fold"}
                      for k, v in out["methods"].items()}, ensure_ascii=False, indent=2))
    print(json.dumps(tests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
