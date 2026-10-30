"""实验编排: 搜索(Algorithm 1) -> 多起点精修 -> 二选验证, 按 spec 并行。

用法:
  python src/run_experiments.py                 # 全部
  python src/run_experiments.py --only reviews_ST,image_resnet50
  python src/run_experiments.py --procs 4
"""
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import est_logit as EL  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
CACHE = os.path.join(PROJ, "cache")
os.makedirs(RES, exist_ok=True)

SPECS = [
    # 论文口径
    "title_ST", "title_COUNT", "description_ST", "reviews_ST",
    "reviews_COUNT", "reviews_TFIDF",
    "image_resnet50", "image_inceptionv3",
    # M2 分块聚合
    "reviews_ST_chunkmean", "reviews_ST_chunkmax", "reviews_ST_chunkattn",
    "description_ST_chunkmean",
    # M1 多折降噪
    "reviews_ST_denoised", "description_ST_denoised", "image_resnet50_denoised",
]


def run_one(args):
    """单个 spec 的完整流程。"""
    kind, spec, force = args
    done = os.path.join(RES, f"model_{spec}.json")
    if os.path.exists(done) and not force:
        try:
            prev = json.load(open(done, encoding="utf-8"))
            if "error" not in prev:
                print(f"[run] {spec}: 已有结果, 跳过", flush=True)
                return prev
        except Exception:
            pass
    t0 = time.time()
    first = EL.load_first()
    emp = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    out = {"kind": kind, "spec": spec}
    df = None
    best_rand, best_vars = {}, []
    try:
        if kind == "plain":
            fes = EL.fe_names(sorted(first["product_id"].unique()))
            varnames = fes + ["price", "position"]
            m = EL._fit(first, varnames, {})
            best_rand, best_vars, df = {}, [], None
        elif kind == "attributes":
            r = EL.attribute_search(first, emp)
            m, best_rand, best_vars = r["model"], r["randvars"], r["best_vars"]
            varnames = None
            out["hist"] = r["hist"]
        else:
            r = EL.algorithm1(first, None, spec, emp)
            m, best_rand, best_vars = r["model"], r["randvars"], r["best_vars"]
            df = r["df"]
            varnames = None
            out["hist"] = r["hist"]
            out["best_k"] = r["best_k"]

        # ---- 多起点精修 (mixed logit 非凸; plain 无需) ----
        if kind != "plain" and best_rand:
            if df is None:
                df = first
            fes = EL.fe_names(sorted(df["product_id"].unique()))
            if kind == "attributes":
                extra = best_vars
            else:
                extra = [c for c in best_vars if c.startswith("pc")]
            vars_ = list(dict.fromkeys(fes + ["price", "position"] + extra))
            ll_best, m_best = m.loglikelihood, m
            for rs in (7, 77, 777):
                try:
                    m2 = EL._fit(df, vars_, best_rand, random_state=rs)
                    if m2.loglikelihood > ll_best + 1e-6:
                        ll_best, m_best = m2.loglikelihood, m2
                except Exception:
                    pass
            m = m_best

        # ---- 统一变量名与数据 ----
        if df is None:
            df = first
        fes = EL.fe_names(sorted(df["product_id"].unique()))
        if kind == "plain":
            varnames = fes + ["price", "position"]
        elif kind == "attributes":
            varnames = list(dict.fromkeys(fes + ["price", "position"] + best_vars))
        else:
            varnames = fes + ["price", "position"] + \
                [c for c in best_vars if c.startswith("pc")]

        # ---- 二选验证 ----
        metrics, pred = EL.evaluate_model(m, df, varnames, emp)
        out.update(metrics)
        out["randvars"] = best_rand
        out["best_vars"] = best_vars
        out["coeff"] = {n: float(c) for n, c in zip(m.coeff_names, m.coeff_)}
        np.save(os.path.join(RES, f"diversion_{spec}.npy"), pred)
        out["seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(RES, f"model_{spec}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=float)
        print(f"[run] {spec}: AIC={out['aic']:.1f} RMSE={out['rmse']:.4f} "
              f"top1={out['top1']:.1f} ({out['seconds']}s)", flush=True)
        return out
    except Exception:
        out["error"] = traceback.format_exc()
        with open(os.path.join(RES, f"model_{spec}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[run] {spec} FAILED:\n{out['error']}", flush=True)
        return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔的 spec 列表")
    ap.add_argument("--force", default=None, help="强制重跑的 spec（覆盖已有结果）")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    force = set(a.force.split(",")) if a.force else set()

    tasks = [("plain", "plain"), ("attributes", "attributes")]
    specs = SPECS if not a.only else a.only.split(",")
    tasks += [("emb", s) for s in specs]
    tasks = [t for t in tasks if t[1] in specs]
    if a.only:
        tasks = [t for t in tasks if t[1] in a.only.split(",")]
    tasks = [(k, s, s in force) for k, s in tasks]

    print(f"[run] 共 {len(tasks)} 个任务, 并行 {a.procs}, 强制 {sorted(force)}", flush=True)
    if a.procs > 1:
        with Pool(a.procs) as pool:
            pool.map(run_one, tasks)
    else:
        for t in tasks:
            run_one(t)
    print("[run] 全部完成", flush=True)


if __name__ == "__main__":
    main()
