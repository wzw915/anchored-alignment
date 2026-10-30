# -*- coding: utf-8 -*-
"""图 10: t-SNE 对比 —— 冻结嵌入 vs 个体级 InfoNCE 微调后嵌入
预期效果: 冻结嵌入呈清晰产品聚类(替代结构), 微调后被打散成按个人喜好排列的碎片。
输出: figures/fig10_tsne.png + results/tsne_data.json
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
SPEC = "reviews_ST_chunkmean"
SEED = 2026
TAU_NCE, N_NEG, EPOCHS, LR, WD = 0.1, 3, 600, 1e-2, 0.01


def cosine(E):
    E = E - E.mean(0, keepdims=True)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-10, None)


def load_pairs():
    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    second = pd.read_csv(os.path.join(CACHE, "second_choice.csv"))
    fc = first[first.choice == 1].set_index("respondent_id")["asin"]
    sc = second[second.choice == 1].set_index("respondent_id")["asin"]
    pair = pd.DataFrame({"a1": fc, "a2": sc}).dropna().reset_index()
    allb = sorted(first["asin"].unique())
    pair["negs"] = pair.apply(
        lambda r: [b for b in allb if b not in (r.a1, r.a2)], axis=1)
    return pair, allb


def train_infoNCE(E_norm, pair, asin_idx):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    P = torch.nn.Parameter(torch.randn(E_norm.shape[1], 8) * 0.01)
    opt = torch.optim.Adam([P], lr=LR, weight_decay=WD)
    Et = torch.tensor(E_norm, dtype=torch.float32)
    js, ks, ng = [], [], []
    for _, r in pair.iterrows():
        negs = list(r["negs"]); rng.shuffle(negs)
        js.append(asin_idx[r.a1]); ks.append(asin_idx[r.a2])
        ng.append([asin_idx[n] for n in negs[:N_NEG]])
    jt, kt, nt = torch.tensor(js), torch.tensor(ks), torch.tensor(ng)
    for ep in range(EPOCHS):
        Zp = F.normalize(Et @ P, dim=1)
        S = Zp @ Zp.T
        pos = S[jt, kt].unsqueeze(1)
        neg = S[jt.unsqueeze(1).expand(-1, N_NEG), nt]
        logits = torch.cat([pos, neg], dim=1) / TAU_NCE
        loss = -F.log_softmax(logits, dim=1)[:, 0].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return P.detach().numpy()


def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    asin_idx = {a: i for i, a in enumerate(asins)}
    E = cosine(np.load(os.path.join(CACHE, "emb", SPEC + ".npy")))
    pair, _ = load_pairs()
    W = train_infoNCE(E, pair, asin_idx)
    Ef = cosine(E @ W)

    titles = [t if len(t) <= 22 else t[:20] + "…" for t in books["title"]]
    perplexity = min(5, len(asins) - 1)
    ts1 = TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
               init="pca").fit_transform(E)
    ts2 = TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
               init="pca").fit_transform(Ef)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    genre = books["genre"].tolist()
    cmap = {"Mystery, Thriller & Suspense": "#c0392b",
            "Science Fiction & Fantasy": "#2980b9"}
    for ax, T, title in ((axes[0], ts1, "(A) 冻结大模型嵌入：清晰的替代聚类"),
                         (axes[1], ts2, "(B) 个体级 InfoNCE 微调后：结构被打散")):
        for i in range(len(asins)):
            ax.scatter(*T[i], s=140, color=cmap.get(genre[i], "#7f8c8d"),
                       zorder=3, edgecolor="white", linewidth=1.2)
            ax.annotate(titles[i], T[i], fontsize=7.5,
                        xytext=(6, 4), textcoords="offset points")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c,
                          label=g) for g, c in cmap.items()]
    axes[0].legend(handles=handles, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig10_tsne.png"), dpi=140, bbox_inches="tight")
    plt.close()

    # 量化: 同类目对的平均相似度保留比
    same = [(i, j) for i in range(len(asins)) for j in range(len(asins))
            if i < j and genre[i] == genre[j]]
    r_f = float(np.mean([E[i] @ E[j] for i, j in same]))
    r_a = float(np.mean([Ef[i] @ Ef[j] for i, j in same]))
    out = dict(same_genre_cos_frozen=round(r_f, 4),
               same_genre_cos_aligned=round(r_a, 4),
               note="同类目对平均余弦: 微调后下降即结构打散的量化证据")
    with open(os.path.join(RES, "tsne_data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False))
    print("[fig10] -> figures/fig10_tsne.png")


if __name__ == "__main__":
    sys.exit(main())
