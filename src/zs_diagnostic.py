# -*- coding: utf-8 -*-
"""零样本替代推演 · 第一步诊断实验（零训练成本）

对应实验设计（老师/专家方案）:
  第一周诊断: 语义相似性 vs 替代相似性 的相关性矩阵
      Corr(sim_USE(j,k), diversion(j,k)) 等, 用置换检验给显著性
  第四周预演: 零样本 diversion RMSE
      留出 2 本产品(K=2: 同作者 The Inmate + 系列续作 The Ashes),
      从第二选数据构造经验 s_{j->new}, 与 softmax(z_j·z_new/tau) 预测对比;
      基线 = 均匀分配 + 原论文拼接嵌入(丰富条件近似)
输出: results/zs_diagnostic.json
"""
import os, json, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")

# 留出产品: 同作者对之一(The Inmate) + 系列续作(Ashes); 对应保留 Housemaid / Serpent
HELDOUT_TITLES = ["The Inmate", "The Ashes & The Star Cursed King"]
EMB_SPECS = ["title_ST", "description_ST", "reviews_ST", "reviews_ST_chunkmean",
             "image_inceptionv3"]


def offdiag(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


def perm_p(x, y, n=5000, seed=2026):
    """置换检验 p 值 (Pearson r)。"""
    r0 = pearsonr(x, y)[0]
    rng = np.random.default_rng(seed)
    cnt = 0
    y = np.asarray(y)
    for _ in range(n):
        if abs(pearsonr(x, rng.permutation(y))[0]) >= abs(r0):
            cnt += 1
    return r0, (cnt + 1) / (n + 1)


def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    J = len(asins)
    emp = np.load(os.path.join(CACHE, "diversion_empirical.npy"))

    # ---------- 1) 诊断: 语义相似性 vs 替代相似性 ----------
    diag = []
    for spec in EMB_SPECS:
        E = np.load(os.path.join(CACHE, "emb", spec + ".npy"))
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-10, None)
        S = E @ E.T
        r, p = perm_p(offdiag(S), offdiag(emp))
        rs = spearmanr(offdiag(S), offdiag(emp)).statistic
        diag.append(dict(spec=spec, pearson=round(float(r), 4),
                         spearman=round(float(rs), 4), perm_p=round(float(p), 4)))
    # 多模态拼接 (图像 + 评论, 均值融合)
    Et = np.load(os.path.join(CACHE, "emb", "reviews_ST_chunkmean.npy"))
    Ei = np.load(os.path.join(CACHE, "emb", "image_inceptionv3.npy"))
    for name, (A, B) in {"multi_text+image": (Et, Ei)}.items():
        A = (A - A.mean(0)) / np.clip(A.std(0), 1e-10, None)
        B = (B - B.mean(0)) / np.clip(B.std(0), 1e-10, None)
        C = np.hstack([A, B])
        C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-10, None)
        S = C @ C.T
        r, p = perm_p(offdiag(S), offdiag(emp))
        diag.append(dict(spec=name, pearson=round(float(r), 4),
                         spearman=round(float(spearmanr(offdiag(S), offdiag(emp)).statistic), 4),
                         perm_p=round(float(p), 4)))

    # ---------- 2) 零样本预演: 留出产品的 diversion ----------
    idx = {a: i for i, a in enumerate(asins)}
    heldout = [idx[books.loc[books.title == t, "asin"].iloc[0]] for t in HELDOUT_TITLES]
    keep = [i for i in range(J) if i not in heldout]

    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    second = pd.read_csv(os.path.join(CACHE, "second_choice.csv"))
    # 经验 s_{j->new}: 第一选=j 的被试中第二选=new 的比例 (按 respondent_id 连接;
    # 第二选任务已移除该被试的第一选产品, 故第二选=new 可直接观测)
    fc = first[first["choice"] == 1].set_index("respondent_id")["asin"]
    sc = second[second["choice"] == 1].set_index("respondent_id")["asin"]
    pair = pd.DataFrame({"a1": fc, "a2": sc}).dropna()

    zs = []
    for h in heldout:
        h_asin = asins[h]
        n1 = pair[pair.a1 != h_asin].groupby("a1").size()          # 第一选=j(不含h) 的人数
        n2 = pair[(pair.a1 != h_asin) & (pair.a2 == h_asin)].groupby("a1").size()
        emp_row = {a: n2.get(a, 0) / max(n1.get(a, 0), 1) for a in n1.index}
        # 各嵌入条件下的 softmax 预测
        for spec in EMB_SPECS + ["multi_text+image"]:
            if spec == "multi_text+image":
                A = (Et - Et.mean(0)) / np.clip(Et.std(0), 1e-10, None)
                B = (Ei - Ei.mean(0)) / np.clip(Ei.std(0), 1e-10, None)
                E = np.hstack([A, B])
                E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-10, None)
            else:
                E = np.load(os.path.join(CACHE, "emb", spec + ".npy"))
                E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-10, None)
            sims = {a: float(E[idx[a]] @ E[h]) for a in n1.index}
            best_tau, best_rmse, best_pred = None, 1e9, None
            for tau in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
                ex = np.exp(np.array([sims[a] for a in n1.index]) / tau)
                pred = ex / ex.sum()
                obs = np.array([emp_row[a] for a in n1.index])
                rm = float(np.sqrt(np.mean((pred - obs) ** 2)))
                if rm < best_rmse:
                    best_tau, best_rmse, best_pred = tau, rm, pred
            zs.append(dict(heldout=HELDOUT_TITLES[heldout.index(h)], spec=spec,
                           tau=best_tau, rmse=round(best_rmse, 4),
                           n_obs=int(len(n1.index))))
        # 均匀基线
        uni = np.full(len(n1.index), 1.0 / len(n1.index))
        obs = np.array([emp_row[a] for a in n1.index])
        zs.append(dict(heldout=HELDOUT_TITLES[heldout.index(h)], spec="uniform",
                       tau=None, rmse=round(float(np.sqrt(np.mean((uni - obs) ** 2))), 4),
                       n_obs=int(len(n1.index))))
        print(f"[zero-shot] {h_asin}: 完成条件对比", flush=True)

    out = dict(diagnostic=diag, zero_shot=zs,
               heldout_asins=[asins[i] for i in heldout])
    with open(os.path.join(RES, "zs_diagnostic.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
