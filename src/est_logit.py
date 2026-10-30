"""mixed logit 估计与二选验证（对齐论文 Compiani, Morozov & Seiler 2026）。

效用设定 (论文式 1):
    u_ij = delta_j + alpha_i * price_ij + gamma * rank_ij + theta_i' * PC_j + eps_ij
    - delta_j: 产品固定效应 (基准 = product_id 47)
    - alpha_i ~ N(a_bar, sd_a) (价格随机系数), theta_i ~ N(0, sd_p) (PC 随机系数)
    - eps_ij ~ Type I Extreme Value
    - 仅用第一选择估计 (MLE, Halton 抽样模拟)

验证 (论文 3.3 节):
    预测反事实第二选择 diversion: 去掉产品 j 后重新计算概率,
    s_{j->k} = E_i[(s_k^{(j)} - s_k) / s_j], 与实验观测的二选转移矩阵比较。
    指标: RMSE (论文式 2, 去对角), MAE, top-1/top-3 最近替代命中率 (本文新增)。
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from xlogit import MixedLogit, MultinomialLogit

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
os.makedirs(RES, exist_ok=True)

N_PC = 5                 # 候选主成分数 (论文 P=6; N=10 书时 PC5+ 方差解释率 <2%)
BASE_FE = 47             # 基准产品 (The Art of Letting Go)
N_DRAWS = 100            # Halton 抽样数 (与论文一致)
N_RESTARTS = 1           # 搜索阶段起点数; 最终模型用 5
SEED = 1024


def fe_names(pids):
    return [f"product_id_{p}" for p in sorted(pids) if p != BASE_FE]


def load_first():
    df = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    return df.sort_values(["choice_id", "product_id"]).reset_index(drop=True)


def merge_pcs(df, spec, books=None, P=N_PC):
    """把 spec 的前 P 个主成分(标准化)合并进选择数据, 返回 (df, pc_names)。"""
    if books is None:
        books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    E = np.load(os.path.join(CACHE, "emb", spec + ".npy"))
    E = E - E.mean(0, keepdims=True)
    from sklearn.decomposition import PCA
    pc = PCA(n_components=P).fit_transform(E)
    pc = (pc - pc.mean(0)) / pc.std(0)
    names = [f"pc{i+1}" for i in range(P)]
    m = {a: pc[i] for i, a in enumerate(books["asin"])}
    for c in names:
        df[c] = df["asin"].map(lambda a, c=c: m[a][list(names).index(c)])
    return df, names


def _fit(data, varnames, randvars, n_draws=N_DRAWS, halton=True,
         random_state=SEED, init_coeff=None):
    cls = MixedLogit if randvars else MultinomialLogit
    m = cls()
    kw = dict(X=data[varnames], y=data["choice"], varnames=varnames,
              ids=data["choice_id"], alts=data["product_id"], verbose=0)
    if randvars:
        kw.update(n_draws=n_draws, halton=halton, randvars=randvars,
                  random_state=random_state)
        if init_coeff is not None:
            kw["init_coeff"] = init_coeff
    else:
        kw["random_state"] = random_state
    m.fit(**kw)
    return m


def aic(model):
    return float(2 * len(model.coeff_names) - 2 * model.loglikelihood)


def predict_probs(model, data, varnames, avail=None, halton=True):
    kw = dict(X=data[varnames], varnames=varnames,
              ids=data["choice_id"], alts=data["product_id"],
              avail=avail, return_proba=True, random_state=1, verbose=0)
    if isinstance(model, MixedLogit):
        kw["halton"] = halton
        kw["n_draws"] = N_DRAWS          # 与拟合一致 (xlogit 默认 1000, 内存爆炸)
    _, probs = model.predict(**kw)
    return np.asarray(probs)


def diversion_matrix(model, first, varnames, base_probs=None):
    """预测 diversion 矩阵 (移除法, 逐个体, 对齐论文实现)。"""
    J = first["product_id"].nunique()
    prods = sorted(first["product_id"].unique())
    if base_probs is None:
        base_probs = predict_probs(model, first, varnames)
    out = np.zeros((J, J))
    col = {p: i for i, p in enumerate(prods)}
    for j in prods:
        avail = (first["product_id"] != j).astype(int).to_numpy()
        pj = predict_probs(model, first, varnames, avail=avail)
        diff = (pj - base_probs) / np.clip(base_probs[:, col[j]], 1e-10, None)[:, None]
        diff[:, col[j]] = 0.0
        out[col[j]] = diff.mean(axis=0)
    return out


def div_metrics(pred, emp):
    """RMSE(去对角, 论文式2), MAE, top-1/top-3 最近替代命中率。"""
    J = emp.shape[0]
    mask = ~np.eye(J, dtype=bool)
    rmse = float(np.sqrt(np.mean((emp - pred)[mask] ** 2)))
    mae = float(np.mean(np.abs((emp - pred)[mask])))
    top1 = top3 = 0
    for j in range(J):
        obs_best = int(np.argmax(emp[j]))
        pred_rank = np.argsort(-pred[j])
        top1 += int(pred_rank[0] == obs_best)
        top3 += int(obs_best in pred_rank[:3])
    return dict(rmse=rmse, mae=mae, top1=top1 / J, top3=top3 / J)


def evaluate_model(model, first, varnames, emp):
    probs = predict_probs(model, first, varnames)
    pred = diversion_matrix(model, first, varnames, base_probs=probs)
    m = div_metrics(pred, emp)
    m["aic"] = aic(model)
    m["ll"] = float(model.loglikelihood)
    m["n_params"] = len(model.coeff_names)
    return m, pred


def algorithm1(first, books, spec, emp, n_pc=N_PC, n_draws=N_DRAWS, verbose=True):
    """论文 Algorithm 1: 逐步加入随机系数, 按 AIC 选规格。

    候选集 = {price, pc1..pcP}; K=0 为 plain logit。
    每步估计所有 C(cand, K) 组合; AIC 不再下降即停止。
    返回 (best_vars, best_randvars, history)。
    """
    df, pc_names = merge_pcs(first.copy(), spec, books, n_pc)
    fes = fe_names(sorted(df["product_id"].unique()))
    cand = ["price"] + pc_names
    base_vars = fes + ["price", "position"]
    hist = []

    m0 = _fit(df, base_vars, {})
    best = dict(k=0, vars=[], randvars={}, aic=aic(m0), model=m0)
    if verbose:
        print(f"  [{spec}] K=0 plain: AIC={best['aic']:.1f}", flush=True)
    hist.append(dict(k=0, spec="plain", aic=best["aic"]))

    for K in range(1, n_pc + 2):
        improved = False
        for combo in itertools.combinations(cand, K):
            rv = {v: "n" for v in combo}
            vars_ = list(dict.fromkeys(base_vars + list(combo)))
            try:
                m = _fit(df, vars_, rv, n_draws=n_draws)
            except Exception as e:
                print(f"  [{spec}] fit fail {combo}: {e}", flush=True)
                continue
            a = aic(m)
            hist.append(dict(k=K, spec=",".join(combo), aic=a))
            if a < best["aic"]:
                best = dict(k=K, vars=list(combo), randvars=rv, aic=a, model=m)
                improved = True
            if verbose:
                print(f"  [{spec}] K={K} {combo}: AIC={a:.1f}"
                      f"{' *' if a == best['aic'] else ''}", flush=True)
        if not improved:
            break
    return dict(spec=spec, best_k=best["k"], best_vars=best["vars"],
                randvars=best["randvars"], aic=best["aic"], model=best["model"],
                df=df, pc_names=pc_names, hist=hist)


def attribute_search(first, emp, n_draws=N_DRAWS):
    """属性 mixed logit (论文对照组): 候选 {price, year, pages, genre_mystery, genre_scifi}。"""
    fes = fe_names(sorted(first["product_id"].unique()))
    cand = ["price", "year", "pages", "genre_mystery", "genre_scifi"]
    base_vars = fes + ["price", "position"]
    hist = []
    m0 = _fit(first, base_vars, {})
    best = dict(vars=[], randvars={}, aic=aic(m0), model=m0)
    hist.append(dict(k=0, spec="plain", aic=best["aic"]))
    for K in range(1, len(cand) + 1):
        improved = False
        for combo in itertools.combinations(cand, K):
            rv = {v: "n" for v in combo}
            m = _fit(first, list(dict.fromkeys(base_vars + list(combo))), rv,
                     n_draws=n_draws)
            a = aic(m)
            hist.append(dict(k=K, spec=",".join(combo), aic=a))
            if a < best["aic"]:
                best = dict(vars=list(combo), randvars=rv, aic=a, model=m)
                improved = True
        if not improved:
            break
    return dict(spec="attributes", best_k=len(best["vars"]),
                best_vars=best["vars"], randvars=best["randvars"],
                aic=best["aic"], model=best["model"], hist=hist)
