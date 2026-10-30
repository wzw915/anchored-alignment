"""复现校准与独立验证: 把"代码跑通"升级为"可核对的复现"。

对应论文复现的四类判据:
  A. 识别性检查      —— PC 得分等比缩放不改变似然 (logit 的度量事实), 证明方法被真正理解,
                        同时解释 M1 为何不能用"缩放"做降噪;
  B. AIC-反事实校准  —— 复算论文"一选 AIC 与二选 RMSE 相关 0.78"在本文样本上的取值;
  C. 选型一致性      —— AIC 选中的规格是否就是反事实 RMSE 最优;
  D. 敏感性          —— P=6 (论文口径) 与 Halton 随机种子扰动下的结论是否稳定。

用法:
  python src/replication_checks.py --phase fast     # A/B/C, 数分钟
  python src/replication_checks.py --phase p6       # D1: 论文 P=6 重跑 (耗时)
  python src/replication_checks.py --phase seeds    # D2: 种子扰动 (耗时)
  python src/replication_checks.py --phase all
结果写入 results/replication_checks.json (各 phase 增量合并)。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import est_logit as EL  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
CACHE = os.path.join(PROJ, "cache")
OUT = os.path.join(RES, "replication_checks.json")

# 重点规格: M2 的核心对照对 + 论文口径最佳
KEY_SPECS = ["reviews_ST_chunkmean", "reviews_ST", "description_ST"]


def _load_out():
    if os.path.exists(OUT):
        return json.load(open(OUT, encoding="utf-8"))
    return {}


def _save_out(d):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, default=float)


def _emp():
    return np.load(os.path.join(CACHE, "diversion_empirical.npy"))


def _collect():
    """读取全部有效结果 -> 按 AIC 排序的行。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "model_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if "error" in d or not d.get("rmse"):
            continue
        rows.append(dict(spec=os.path.basename(f)[len("model_"):-len(".json")],
                         aic=d["aic"], rmse=d["rmse"], mae=d.get("mae"),
                         top1=d.get("top1"), best_k=d.get("best_k")))
    rows.sort(key=lambda r: r["aic"])
    return rows


# ---------------- A. 识别性: PC 等比缩放 ----------------
def check_identification(spec="description_ST"):
    """把 PC 得分乘 k 后重估同一结构: 似然应完全不变, 系数与方差按 1/k 缩放。

    若成立 -> 说明"对 PC 得分做均匀缩放"本质上只是重标定方差参数, 不可识别,
    因此 M1 降噪必须改变嵌入的相对取值 (多测量平均), 而不能靠缩放。
    """
    from sklearn.decomposition import PCA
    first = EL.load_first()
    books = __import__("pandas").read_csv(os.path.join(CACHE, "books_clean.csv"))
    base = json.load(open(os.path.join(RES, f"model_{spec}.json"), encoding="utf-8"))
    used = [c for c in (base.get("best_vars") or []) if str(c).startswith("pc")] or ["pc1"]
    P = max(int(c[2:]) for c in used)

    E = np.load(os.path.join(CACHE, "emb", spec + ".npy"))
    E = E - E.mean(0, keepdims=True)
    pc = PCA(n_components=P).fit_transform(E)
    pc = (pc - pc.mean(0)) / pc.std(0)

    out = dict(spec=spec, used_pcs=used, n_pc=P, runs={})
    for k in (1.0, 2.0):
        df = first.copy()
        m = {a: pc[i] for i, a in enumerate(books["asin"])}
        for i in range(P):
            name = f"pc{i+1}"
            df[name] = df["asin"].map(lambda a, i=i: m[a][i] * k)
        fes = EL.fe_names(sorted(df["product_id"].unique()))
        vars_ = list(dict.fromkeys(fes + ["price", "position"] + used))
        rv = {v: "n" for v in used}
        mod = EL._fit(df, vars_, rv)
        coef = {n: float(c) for n, c in zip(mod.coeff_names, mod.coeff_)}
        out["runs"][f"k={k:g}"] = dict(
            ll=float(mod.loglikelihood), aic=EL.aic(mod),
            sd={v: coef.get(f"{v}.sd", coef.get(f"sd.{v}")) for v in used},
        )
    a, b = out["runs"]["k=1"], out["runs"]["k=2"]
    out["ll_diff"] = abs(a["ll"] - b["ll"])
    out["aic_diff"] = abs(a["aic"] - b["aic"])
    out["ratio_expected"] = 2.0
    out["ratio_observed"] = {
        v: (a["sd"][v] / b["sd"][v] if b["sd"].get(v) else None) for v in used}
    out["pass"] = bool(out["ll_diff"] < 1e-3)
    print(f"[A] 识别性 {spec}: LL {a['ll']:.4f} vs {b['ll']:.4f} "
          f"(diff={out['ll_diff']:.2e}) ratio={out['ratio_observed']} -> "
          f"{'PASS' if out['pass'] else 'CHECK'}", flush=True)
    return out


# ---------------- B/C. AIC 与反事实表现的相关 ----------------
def check_aic_calibration():
    from scipy.stats import pearsonr, spearmanr

    rows = _collect()
    est = [r for r in rows if (r["best_k"] or 0) > 0]

    def _stat(sub):
        aic = [r["aic"] for r in sub]
        rmse = [r["rmse"] for r in sub]
        pr = pearsonr(aic, rmse)
        sp = spearmanr(aic, rmse)
        return dict(n=len(sub), pearson=float(pr[0]), pearson_p=float(pr[1]),
                    spearman=float(sp[0]), spearman_p=float(sp[1]))

    best_aic = min(rows, key=lambda r: r["aic"])["spec"]
    best_rmse = min(rows, key=lambda r: r["rmse"])["spec"]
    out = dict(
        paper_value=0.78,
        paper_scope="40 个 Amazon 类目之间 (每类目一个数据集)",
        all_specs=_stat(rows),
        estimated_only=_stat(est),
        spec_with_min_aic=best_aic,
        spec_with_min_rmse=best_rmse,
        aic_selects_best=bool(best_aic == best_rmse),
        rows=rows,
    )
    print(f"[B] 全部规格 Pearson={out['all_specs']['pearson']:.3f} "
          f"(p={out['all_specs']['pearson_p']:.3f}, n={out['all_specs']['n']}); "
          f"仅随机系数 Pearson={out['estimated_only']['pearson']:.3f} "
          f"(p={out['estimated_only']['pearson_p']:.3f}, n={out['estimated_only']['n']})",
          flush=True)
    print(f"[C] AIC 最优={best_aic} / RMSE 最优={best_rmse} -> "
          f"{'一致' if out['aic_selects_best'] else '背离'}", flush=True)
    return out


# ---------------- D1. P=6 (论文口径) 重跑 ----------------
def _fit_and_eval(spec, n_pc, n_draws=EL.N_DRAWS, restarts=(7, 77, 777), seed=None):
    first = EL.load_first()
    emp = _emp()
    r = EL.algorithm1(first, None, spec, emp, n_pc=n_pc, n_draws=n_draws)
    df, best_rand, best_vars = r["df"], r["randvars"], r["best_vars"]
    fes = EL.fe_names(sorted(df["product_id"].unique()))
    extra = [c for c in best_vars if str(c).startswith("pc")]
    vars_ = list(dict.fromkeys(fes + ["price", "position"] + extra))
    m = r["model"]
    if best_rand:
        ll_best, m_best = m.loglikelihood, m
        seq = (seed,) if seed is not None else restarts
        for rs in seq:
            try:
                m2 = EL._fit(df, vars_, best_rand, random_state=rs,
                             n_draws=n_draws)
                if m2.loglikelihood > ll_best + 1e-6:
                    ll_best, m_best = m2.loglikelihood, m2
            except Exception:
                pass
        m = m_best
    met, _ = EL.evaluate_model(m, df, vars_, emp)
    return dict(spec=spec, n_pc=n_pc, n_draws=n_draws, best_k=r["best_k"],
                best_vars=list(best_vars), **met)


def phase_p6():
    out = _load_out()
    res = out.get("p6", [])
    done = {r["spec"] for r in res}
    for spec in KEY_SPECS:
        if spec in done:
            print(f"[D1] {spec} 已有 P=6 结果, 跳过", flush=True)
            continue
        t0 = time.time()
        try:
            r = _fit_and_eval(spec, n_pc=6)
        except Exception as e:
            print(f"[D1] {spec} 失败: {e}", flush=True)
            continue
        r["seconds"] = round(time.time() - t0, 1)
        res.append(r)
        out["p6"] = res
        _save_out(out)
        print(f"[D1] P=6 {spec}: K={r['best_k']} {r['best_vars']} "
              f"AIC={r['aic']:.1f} RMSE={r['rmse']:.4f} ({r['seconds']}s)", flush=True)
    out["p6"] = res
    _save_out(out)
    return out


# ---------------- D2. 种子扰动 ----------------
def phase_seeds(seeds=(7, 77, 777)):
    out = _load_out()
    res = out.get("seeds", [])
    have = {(r["spec"], r["seed"]) for r in res}
    first = EL.load_first()
    emp = _emp()
    for spec in KEY_SPECS:
        base = json.load(open(os.path.join(RES, f"model_{spec}.json"), encoding="utf-8"))
        rv = base.get("randvars") or {}
        used = [c for c in (base.get("best_vars") or []) if str(c).startswith("pc")]
        if not rv:
            print(f"[D2] {spec} 无随机系数, 种子无影响, 跳过", flush=True)
            continue
        df, _ = EL.merge_pcs(first.copy(), spec, None, EL.N_PC)
        fes = EL.fe_names(sorted(df["product_id"].unique()))
        vars_ = list(dict.fromkeys(fes + ["price", "position"] + used))
        for sd in seeds:
            if (spec, sd) in have:
                continue
            t0 = time.time()
            try:
                m = EL._fit(df, vars_, rv, random_state=sd)
                met, _ = EL.evaluate_model(m, df, vars_, emp)
            except Exception as e:
                print(f"[D2] {spec} seed={sd} 失败: {e}", flush=True)
                continue
            row = dict(spec=spec, seed=sd, **met,
                       seconds=round(time.time() - t0, 1))
            res.append(row)
            out["seeds"] = res
            _save_out(out)
            print(f"[D2] {spec} seed={sd}: AIC={row['aic']:.1f} "
                  f"RMSE={row['rmse']:.4f}", flush=True)
    out["seeds"] = res
    _save_out(out)
    return out


def phase_draws(specs=("description_ST", "reviews_ST_chunkmean"), draws=(300, 500)):
    """D3: Halton 抽样数敏感性。xlogit 的 Halton 序列是确定性的, 换 random_state
    不改变任何数值, 真正的模拟误差来源是抽样数本身, 故补做 n_draws 扰动。
    说明: 仅拟合用高抽样数 (预测仍用 100 draws 以控制内存), 比较对象是 LL / AIC。
    """
    out = _load_out()
    res = out.get("draws", [])
    have = {(r["spec"], r["n_draws"]) for r in res}
    first = EL.load_first()
    for spec in specs:
        base = json.load(open(os.path.join(RES, f"model_{spec}.json"), encoding="utf-8"))
        rv = base.get("randvars") or {}
        if not rv:
            continue
        used = [c for c in (base.get("best_vars") or []) if str(c).startswith("pc")]
        df, _ = EL.merge_pcs(first.copy(), spec, None, EL.N_PC)
        fes = EL.fe_names(sorted(df["product_id"].unique()))
        vars_ = list(dict.fromkeys(fes + ["price", "position"] + used))
        for nd in draws:
            if (spec, nd) in have:
                continue
            t0 = time.time()
            try:
                m = EL._fit(df, vars_, rv, n_draws=nd)
            except Exception as e:
                print(f"[D3] {spec} n_draws={nd} 失败: {e}", flush=True)
                continue
            row = dict(spec=spec, n_draws=nd, ll=float(m.loglikelihood),
                       aic=EL.aic(m), base_ll=float(base["ll"]),
                       base_aic=float(base["aic"]),
                       base_n_draws=EL.N_DRAWS,
                       seconds=round(time.time() - t0, 1))
            res.append(row)
            out["draws"] = res
            _save_out(out)
            print(f"[D3] {spec} n_draws={nd}: LL={row['ll']:.2f} "
                  f"AIC={row['aic']:.1f} (100 draws: {row['base_aic']:.1f}, "
                  f"Δ={row['aic']-row['base_aic']:+.1f})", flush=True)
    out["draws"] = res
    _save_out(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="fast",
                    choices=["fast", "p6", "seeds", "draws", "all"])
    a = ap.parse_args()
    out = _load_out()
    if a.phase in ("fast", "all"):
        out["identification"] = check_identification()
        out["calibration"] = check_aic_calibration()
        _save_out(out)
    if a.phase in ("p6", "all"):
        phase_p6()
    if a.phase in ("seeds", "all"):
        phase_seeds()
    if a.phase in ("draws", "all"):
        phase_draws()
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
