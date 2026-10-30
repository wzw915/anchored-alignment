"""裂缝-组件-验证：Algorithm 1 的 AIC 选型在类目内失灵，用 K 折 CV 似然替换。

裂缝（流水线已两次暴露）：
  - 图 7 / 4.6 节：AIC 最优的降噪规格二选 RMSE 反而全场最差（0.0928 vs 冠军 0.0667）；
  - 4.7 节：论文口径 P=6 下 AIC 再降 1.9，二选 RMSE 反而恶化 2.1%。

填缝组件（来源清楚）：Stone (1977) 证明 AIC 是留一交叉验证的渐近近似；
小样本下直接用 5 折 CV 对数似然做选型（按被试分折，杜绝同一个人跨折）。

验证（二选反事实）：同一候选池上，AIC 选出的结构 vs CV 选出的结构，
比较已知的全量拟合二选 RMSE；并比较 AIC 与 CV-LL 各自和 RMSE 的相关。

用法:
  python src/cv_selection.py --smoke     # plain × 2 折冒烟
  python src/cv_selection.py             # 全池 10 结构 × 5 折 (约 40 分钟, 串行)
结果增量写入 results/cv_selection.json。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import est_logit as EL  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
CACHE = os.path.join(PROJ, "cache")
OUT = os.path.join(RES, "cv_selection.json")

N_FOLDS = 5
FOLD_SEED = 2026

DISPLAY = {
    "plain": "Plain Logit", "attributes": "属性(pages)",
    "reviews_ST_K1": "评论拼接 K=1", "reviews_ST_K3": "评论拼接 K=3",
    "reviews_ST_P6_K4": "评论拼接 K=4(P6)", "reviews_ST_chunkmean_K3": "分块均值 K=3",
    "reviews_ST_chunkattn_K4": "注意力池化 K=4", "reviews_ST_denoised_K2": "降噪 K=2(AIC冠军)",
    "description_ST_K1": "简介拼接 K=1", "description_ST_denoised_K2": "简介降噪 K=2",
}

# 候选池: (label, emb_spec or None, n_pc, extra_vars, randvars)
# extra_vars: 除 fes+price+position 之外的变量; price 进 randvars 则也列在 extra
POOL = [
    ("plain", None, 0, [], {}),
    ("attributes", None, 0, ["pages"], {"pages": "n"}),
    ("reviews_ST_K1", "reviews_ST", 5, ["pc1"], {"pc1": "n"}),
    ("reviews_ST_K3", "reviews_ST", 5, ["price", "pc1", "pc3"],
     {"price": "n", "pc1": "n", "pc3": "n"}),
    ("reviews_ST_P6_K4", "reviews_ST", 6, ["price", "pc1", "pc3", "pc6"],
     {"price": "n", "pc1": "n", "pc3": "n", "pc6": "n"}),
    ("reviews_ST_chunkmean_K3", "reviews_ST_chunkmean", 5, ["pc1", "pc2", "pc3"],
     {"pc1": "n", "pc2": "n", "pc3": "n"}),
    ("reviews_ST_chunkattn_K4", "reviews_ST_chunkattn", 5, ["pc1", "pc2", "pc3", "pc4"],
     {"pc1": "n", "pc2": "n", "pc3": "n", "pc4": "n"}),
    ("reviews_ST_denoised_K2", "reviews_ST_denoised", 5, ["pc2", "pc3"],
     {"pc2": "n", "pc3": "n"}),
    ("description_ST_K1", "description_ST", 5, ["pc1"], {"pc1": "n"}),
    ("description_ST_denoised_K2", "description_ST_denoised", 5, ["pc1", "pc5"],
     {"pc1": "n", "pc5": "n"}),
]

# 各结构的全量拟合已知结果 (来自 results/model_*.json 与 replication_checks.json)
KNOWN = {
    "plain": (40998.7, 0.0913), "attributes": (40996.4, 0.0752),
    "reviews_ST_K1": (40997.5, 0.0865), "reviews_ST_K3": (40990.5, 0.0776),
    "reviews_ST_P6_K4": (40988.6, 0.0793), "reviews_ST_chunkmean_K3": (40988.0, 0.0667),
    "reviews_ST_chunkattn_K4": (40989.9, 0.0739), "reviews_ST_denoised_K2": (40984.1, 0.0928),
    "description_ST_K1": (40993.7, 0.0670), "description_ST_denoised_K2": (40993.5, 0.0742),
}


def _load_out():
    return json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}


def _save_out(d):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, default=float)


def make_folds(df):
    """按被试分 5 折: 同一人的所有行在同一折, 杜绝个体异质性泄漏。"""
    rid = df["choice_id"] // 2
    uniq = np.array(sorted(rid.unique()))
    rng = np.random.default_rng(FOLD_SEED)
    perm = rng.permutation(uniq)
    fold_of = {r: i % N_FOLDS for i, r in enumerate(perm)}
    return rid.map(fold_of).to_numpy()


def build_df(first, emb_spec, n_pc):
    if emb_spec is None:
        return first.copy()
    df, _ = EL.merge_pcs(first.copy(), emb_spec, None, n_pc)
    return df


def holdout_ll(model, data, varnames):
    """折外对数似然: 用 100 draws 模拟概率, 取被选备选的 log p 求和。
    注: predict_probs 返回 (选择数, 备选数), 每行对应一个 choice_id。
    """
    probs = np.asarray(EL.predict_probs(model, data, varnames))
    prods = sorted(data["product_id"].unique())
    col = {p: i for i, p in enumerate(prods)}
    chosen = data[data["choice"] == 1]          # load_first 已按 choice_id 排序
    idx = np.array([col[p] for p in chosen["product_id"]])
    if len(idx) != probs.shape[0]:
        raise ValueError(f"选择数 {len(idx)} != 概率行数 {probs.shape[0]}")
    p_sel = np.clip(probs[np.arange(probs.shape[0]), idx], 1e-12, None)
    return float(np.log(p_sel).sum())


def run_structure(label, emb_spec, n_pc, extra, rv, folds_arr, first, only_folds=None):
    df = build_df(first, emb_spec, n_pc)
    fes = EL.fe_names(sorted(df["product_id"].unique()))
    varnames = list(dict.fromkeys(fes + ["price", "position"] + extra))
    per_fold = []
    t0 = time.time()
    for f in range(N_FOLDS):
        if only_folds is not None and f not in only_folds:
            continue
        test = folds_arr == f
        tr, te = df[~test], df[test]
        try:
            m = EL._fit(tr, varnames, rv)
            ll = holdout_ll(m, te, varnames)
        except Exception as e:
            print(f"  [{label}] fold{f} 失败: {e}", flush=True)
            return None
        per_fold.append(dict(fold=f, ll=ll, n_test=int(test.sum())))
        print(f"  [{label}] fold{f}: holdout LL={ll:.1f} ({time.time()-t0:.0f}s)", flush=True)
    cv_ll = sum(x["ll"] for x in per_fold)
    return dict(label=label, cv_ll=cv_ll, folds=per_fold,
                n_folds=len(per_fold), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    first = EL.load_first()
    folds_arr = make_folds(first)
    out = _load_out()
    # choice_id = respondent_id * 2 + (task - 1)，且 first 只含 task=1，
    # 因此 choice_id.nunique() 就等于受访者数。旧代码误写为 // 2，
    # 把 9,265 人报成 4,632 人（2026-10-29 审计修正）。
    out.setdefault("meta", dict(n_folds=N_FOLDS, fold_seed=FOLD_SEED,
                                n_respondents=int(first["respondent_id"].nunique())))

    done = {r["label"] for r in out.get("pool", [])}
    for label, spec, npc, extra, rv in POOL:
        if label in done:
            print(f"[cv] {label} 已完成, 跳过", flush=True)
            continue
        only = {0, 1} if a.smoke else None
        r = run_structure(label, spec, npc, extra, rv, folds_arr, first, only_folds=only)
        if r is None:
            continue
        r["aic_full"], r["rmse_full"] = KNOWN[label]
        out.setdefault("pool", []).append(r)
        _save_out(out)
        print(f"[cv] {label}: CV-LL={r['cv_ll']:.1f} "
              f"(AIC={r['aic_full']:.1f}, RMSE={r['rmse_full']:.4f})", flush=True)

    pool = out.get("pool", [])
    if len(pool) >= 3:
        from scipy.stats import pearsonr, spearmanr
        aic = [r["aic_full"] for r in pool]
        rmse = [r["rmse_full"] for r in pool]
        cv = [r["cv_ll"] for r in pool]
        out["summary"] = dict(
            n=len(pool),
            aic_pick=min(pool, key=lambda r: r["aic_full"])["label"],
            cv_pick=max(pool, key=lambda r: r["cv_ll"])["label"],
            rmse_best=min(pool, key=lambda r: r["rmse_full"])["label"],
            corr_aic_rmse=pearsonr(aic, rmse)[0],
            corr_cv_rmse=pearsonr(cv, rmse)[0],
            spear_aic_rmse=spearmanr(aic, rmse)[0],
            spear_cv_rmse=spearmanr(cv, rmse)[0],
        )
        s = out["summary"]
        print(f"[sum] AIC 选 {s['aic_pick']} / CV 选 {s['cv_pick']} / "
              f"RMSE 最优 {s['rmse_best']}", flush=True)
        print(f"[sum] corr(AIC,RMSE)={s['corr_aic_rmse']:.3f} "
              f"corr(CV,RMSE)={s['corr_cv_rmse']:.3f}", flush=True)
        _save_out(out)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
