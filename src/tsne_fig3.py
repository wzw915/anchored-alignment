# -*- coding: utf-8 -*-
"""三联 t-SNE: 冻结 / 完全微调 / 锚定 的表示空间对比 (全样本训练表示).
输出: figures/fig10b_tsne_triptych.png + results/tsne_triptych.json
"""
import os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsne_fig import cosine, load_pairs, train_infoNCE
from align_anchored import train_delta
from fig_style import OI, apply_rc, panel_label, save

apply_rc()

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
SPEC = "reviews_ST_chunkmean"
SEED = 2026
DELTA = 0.5


def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    asin_idx = {a: i for i, a in enumerate(asins)}
    S = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    E = cosine(np.load(os.path.join(CACHE, "emb", SPEC + ".npy")))
    pair, _ = load_pairs()

    W = train_infoNCE(E, pair, asin_idx)                 # 全量个体级 InfoNCE
    E_ft = cosine(E @ W)
    Zk = train_delta(E, S, list(range(len(asins))), DELTA)  # 全量锚定
    E_an = Zk

    titles = [t if len(t) <= 20 else t[:18] + "…" for t in books["title"]]
    perp = min(5, len(asins) - 1)
    ts = [TSNE(n_components=2, perplexity=perp, random_state=SEED,
               init="pca").fit_transform(X) for X in (E, E_ft, E_an)]

    # 期刊版: 不逐点标注 (审稿人高频吐槽). 每品类仅标 1 本代表书目 (共 2 个),
    # 联合选取在三个面板中都相距最远的一对, 保证任何面板内标签都不碰撞.
    import numpy as _np
    from itertools import product
    genre_arr = books["genre"].tolist()
    cats = sorted(set(genre_arr))
    idx_cat = {g: [i for i in range(len(asins)) if genre_arr[i] == g]
               for g in cats}
    label_ids = []
    if len(cats) == 2:
        best_pair, best_score = None, -1
        for i, j in product(idx_cat[cats[0]], idx_cat[cats[1]]):
            score = min(_np.linalg.norm(ts[p][i] - ts[p][j])
                        for p in range(3))
            if score > best_score:
                best_score, best_pair = score, (i, j)
        label_ids = list(best_pair)
    else:
        for g in cats:
            label_ids.append(idx_cat[g][0])

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    genre = books["genre"].tolist()
    cmap = {"Mystery, Thriller & Suspense": OI["navy"],
            "Science Fiction & Fantasy": OI["slate"]}
    markers = {"Mystery, Thriller & Suspense": "o",
               "Science Fiction & Fantasy": "^"}
    short = {"Mystery, Thriller & Suspense": "Mystery, thriller & suspense",
             "Science Fiction & Fantasy": "Science fiction & fantasy"}
    for k, (ax, T) in enumerate(zip(axes, ts)):
        for g in cmap:
            pts = [i for i in range(len(asins)) if genre[i] == g]
            ax.scatter(T[pts, 0], T[pts, 1], s=34, color=cmap[g],
                       marker=markers[g], zorder=3, linewidths=0.4,
                       edgecolors="white")
        for i in label_ids:
            ax.annotate(titles[i], T[i], fontsize=7,
                        xytext=(5, 4), textcoords="offset points",
                        color="#333333")
        panel_label(ax, f"({'ABC'[k]})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    handles = [plt.Line2D([], [], marker=markers[g], ls="", color=c,
                          label=short[g], markersize=6)
               for g, c in cmap.items()]
    fig.legend(handles=handles, fontsize=8.5, loc="lower center", ncol=2,
               frameon=False, columnspacing=1.6, handletextpad=0.3,
               bbox_to_anchor=(0.5, -0.06))

    # 量化: 同类目平均余弦 + 与真实 diversion 的相关
    same = [(i, j) for i in range(len(asins)) for j in range(len(asins))
            if i < j and genre[i] == genre[j]]
    off = ~np.eye(len(asins), dtype=bool)
    quant = {}
    for name, X in [("frozen", E), ("fullft", E_ft), ("anchored", E_an)]:
        quant[name] = dict(
            same_genre_cos=round(float(np.mean([X[i] @ X[j] for i, j in same])), 4),
            corr_div=round(float(np.corrcoef((X @ X.T)[off], S[off])[0, 1]), 4))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    save(plt.gcf(), FIG, "fig10b_tsne_triptych")
    plt.close()
    out = dict(delta=DELTA, quant=quant)
    with open(os.path.join(RES, "tsne_triptych.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False))
    print("[fig10b] -> figures/fig10b_tsne_triptych.png")


if __name__ == "__main__":
    sys.exit(main())
