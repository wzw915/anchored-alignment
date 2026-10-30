# -*- coding: utf-8 -*-
"""双协议消融：预处理 × 方法，全程一致，附逐折置换检验

协议 U (primary): 行归一化，不中心化 —— 与 align_anchored.py 一致
协议 C (sensitivity): 去均值后行归一化 —— 常见的 cosine 预处理

方法: frozen / fullft(个体级 InfoNCE+线性投影) / unbounded(δ=∞, 产品级) / anchored(δ 网格)
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
LR_NCE = 1e-2
LR_PROD = 1e-2
DELTAS = [0.1, 0.3, 0.5, 0.7]


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-10, None)


def centered_unit(X):
    return unit(X - X.mean(0, keepdims=True))


def load_pairs():
    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    second = pd.read_csv(os.path.join(CACHE, "second_choice.csv"))
    fc = first[first.choice == 1].set_index("respondent_id")["asin"]
    sc = second[second.choice == 1].set_index("respondent_id")["asin"]
    pair = pd.DataFrame({"a1": fc, "a2": sc}).dropna().reset_index()
    allb = sorted(first["asin"].unique())
    pair["negs"] = pair.apply(lambda r: [b for b in allb if b not in (r.a1, r.a2)], axis=1)
    return pair


def train_infonce(E, pair_train, asin_idx, d_out=8):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    P = torch.nn.Parameter(torch.randn(E.shape[1], d_out) * 0.01)
    opt = torch.optim.Adam([P], lr=LR_NCE, weight_decay=WD)
    Et = torch.tensor(E, dtype=torch.float32)
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
    torch.manual_seed(SEED)
    n = len(keep)
    P = torch.nn.Parameter(torch.zeros(n, E.shape[1]))
    opt = torch.optim.Adam([P], lr=LR_PROD)
    Ct = torch.tensor(E[keep], dtype=torch.float32)
    St = torch.tensor(S[np.ix_(keep, keep)], dtype=torch.float32)
    for _ in range(EPOCHS_PROD):
        Z = F.normalize(Ct + P, dim=1)
        logp = F.log_softmax(Z @ Z.T / TAU_EVAL, dim=1)
        tgt = St / St.sum(1, keepdims=True).clamp_min(1e-10)
        loss = -(tgt * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if np.isfinite(delta_max):
            with torch.no_grad():
                nrm = P.norm(dim=1, keepdim=True).clamp_min(1e-12)
                P.mul_(torch.clamp(nrm, max=delta_max) / nrm)
    with torch.no_grad():
        return F.normalize(Ct + P, dim=1).numpy()


def pred_col(Z, keep, j0):
    Sm = Z @ Z.T
    out = []
    for i in keep:
        r = Sm[i] / TAU_EVAL
        r[i] = -1e9
        e = np.exp(r - r.max())
        out.append(float(e[j0] / e.sum()))
    return np.array(out)


def met(p, o):
    order = np.argsort(-p)
    best = int(np.argmax(o))
    return dict(rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                kendall=float(kendalltau(p, o).statistic),
                top1=int(order[0] == best), top3=int(best in order[:3]))


def perm_paired(a, b):
    d = np.array(a) - np.array(b)
    obs = abs(d.mean())
    cnt = tot = 0
    for signs in itertools.product([1, -1], repeat=len(d)):
        tot += 1
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def summarize(rows):
    out = {}
    for k in ["rmse", "kendall", "top1", "top3"]:
        v = np.array([r[k] for r in rows], float)
        out[k] = round(float(v.mean()), 4)
        out[k + "_sd"] = round(float(v.std(ddof=1)), 4)
    out["per_fold"] = rows
    return out


def run_protocol(E, S, pair_all, asins, asin_idx, J):
    rows = {"frozen": [], "fullft": [], "unbounded": []}
    for d in DELTAS:
        rows[f"anchored_{d}"] = []
    for j0 in range(J):
        keep = [i for i in range(J) if i != j0]
        obs = np.array([S[i, j0] for i in keep])
        rows["frozen"].append(met(pred_col(E, keep, j0), obs))

        ptr = pair_all[(pair_all.a1 != asins[j0]) & (pair_all.a2 != asins[j0])]
        W = train_infonce(E, ptr, asin_idx)
        rows["fullft"].append(met(pred_col(W, keep, j0), obs))

        Zu = E.copy(); Zu[keep] = train_residual(E, S, keep, np.inf)
        rows["unbounded"].append(met(pred_col(Zu, keep, j0), obs))

        for d in DELTAS:
            Za = E.copy(); Za[keep] = train_residual(E, S, keep, d)
            rows[f"anchored_{d}"].append(met(pred_col(Za, keep, j0), obs))
        print(f"  fold {j0} done", flush=True)
    return rows


def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    asin_idx = {a: i for i, a in enumerate(asins)}
    J = len(asins)
    S = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    Eraw = np.load(os.path.join(CACHE, "emb", SPEC + ".npy"))
    pair_all = load_pairs()

    out = {"spec": SPEC, "tau_eval": TAU_EVAL, "n_folds": J, "protocols": {}}
    for name, E in [("unit", unit(Eraw)), ("centered", centered_unit(Eraw))]:
        print(f"protocol {name}", flush=True)
        rows = run_protocol(E, S, pair_all, asins, asin_idx, J)
        methods = {k: summarize(v) for k, v in rows.items()}
        def col(m, key):
            return [r[key] for r in rows[m]]
        tests = {}
        for label, (x, y) in {
            "anchored_vs_frozen": ("anchored_0.5", "frozen"),
            "unbounded_vs_frozen": ("unbounded", "frozen"),
            "anchored_vs_unbounded": ("anchored_0.5", "unbounded"),
            "fullft_vs_frozen": ("fullft", "frozen"),
        }.items():
            tests[label] = {
                "d_rmse": round(float(np.mean(col(x, "rmse")) - np.mean(col(y, "rmse"))), 4),
                "p_rmse": round(perm_paired(col(x, "rmse"), col(y, "rmse")), 4),
                "d_kendall": round(float(np.mean(col(x, "kendall")) - np.mean(col(y, "kendall"))), 4),
                "p_kendall": round(perm_paired(col(x, "kendall"), col(y, "kendall")), 4),
            }
        out["protocols"][name] = {"methods": methods, "tests": tests}

    with open(os.path.join(RES, "ablation_loo.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for name, blk in out["protocols"].items():
        print(f"\n== {name} ==")
        for m, v in blk["methods"].items():
            print(f"{m:14s} rmse={v['rmse']:.4f}±{v['rmse_sd']:.4f}  tau={v['kendall']:.4f}±{v['kendall_sd']:.4f}  top3={v['top3']:.2f}")
        print(json.dumps(blk["tests"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
