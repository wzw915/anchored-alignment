# -*- coding: utf-8 -*-
"""Amazon Books 反向验证: 对齐 vs 冻结嵌入的规模交叉曲线 (老师任务一)

设计:
  - 真值: 用户级共购边 (amazon_build.py 输出), 300-400 本热门书
  - 嵌入: SentenceTransformer (与主线同一模型) 编码 标题+描述+特征
  - 训练: 线性投影 W (384->64), InfoNCE 正样本=共购边, 负样本=随机非边对
  - 评估: 冷启动设定 —— 训练书与评估书完全分离; 在留出书内部,
    用 AUC 衡量"共购伙伴的嵌入相似度高于非伙伴"的程度
  - 曲线: x=训练书数, y=留出 AUC; 冻结嵌入为水平参考线
输出: results/amazon_scaling.json + figures/fig11_amazon.png
"""
import os, json, sys, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
MODEL_DIR = os.path.join(PROJ, "models", "st_minilm")
SEED = 2026
D_OUT = 64
EPOCHS = 400
LR = 1e-2
WD = 1e-3
TAU = 0.1
N_NEG = 5
TRAIN_SIZES = [50, 100, 200, 300]


def cosine(E):
    E = E - E.mean(0, keepdims=True)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-10, None)


def load_graph():
    d = json.load(open(os.path.join(AMZ, "copurchase_edges.json")))
    return d["books"], d["edges"]


def load_texts(asins):
    want = set(asins)
    texts = {}
    with open(os.path.join(AMZ, "meta_books_head.jsonl"), encoding="utf-8",
              errors="replace") as f:
        for line in f:
            if not want:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            a = d.get("parent_asin")
            if a in want:
                parts = [d.get("title") or ""]
                parts += [x for x in (d.get("description") or [])[:2]]
                parts += [x for x in (d.get("features") or [])[:2]]
                texts[a] = " ".join(parts)[:1500]
                want.discard(a)
    return texts


def auc(sim_row, partners, idx):
    """sim_row: 对其余书的相似度; partners: 共购伙伴下标集合。"""
    pos = [sim_row[j] for j in partners if j in idx]
    neg = [sim_row[j] for j in idx if j not in partners and j != idx]
    if not pos or not neg:
        return None
    pos, neg = np.array(pos), np.array(neg)
    return float((pos[:, None] > neg[None, :]).mean())


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    books, edges = load_graph()
    bidx = {a: i for i, a in enumerate(books)}
    texts = load_texts(books)
    books = [a for a in books if a in texts and texts[a].strip()]
    books_set = set(books)
    edges = [[a, b, c] for a, b, c in edges if a in books_set and b in books_set]
    bidx = {a: i for i, a in enumerate(books)}
    J = len(books)
    print(f"books with text: {J}, edges: {len(edges)}")

    # 嵌入
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL_DIR, device="cpu")
    E = st.encode([texts[a] for a in books], batch_size=64,
                  show_progress_bar=False, normalize_embeddings=True)
    E = np.asarray(E)
    E = cosine(E)

    # 邻接表
    nbrs = defaultdict(set)
    for a, b, _ in edges:
        nbrs[bidx[a]].add(bidx[b])
        nbrs[bidx[b]].add(bidx[a])
    # 只保留有 >=3 个伙伴的书进入评估池
    eval_pool = [i for i in range(J) if len(nbrs[i]) >= 3]
    print("eval pool:", len(eval_pool))

    results = []
    for m in TRAIN_SIZES:
        random.shuffle(eval_pool)
        test_books = eval_pool[:120]
        train_books = [i for i in eval_pool[120:]][:m]
        train_set = set(train_books)
        pos_pairs = [[a, b] for a, b, _ in edges
                     if a in train_set and b in train_set]
        if len(pos_pairs) < 50:
            continue
        pt = torch.tensor(pos_pairs)
        allb = torch.arange(J)
        W = torch.nn.Parameter(torch.randn(E.shape[1], D_OUT) * 0.01)
        opt = torch.optim.Adam([W], lr=LR, weight_decay=WD)
        Et = torch.tensor(E, dtype=torch.float32)
        for ep in range(EPOCHS):
            Zp = F.normalize(Et @ W, dim=1)
            S = Zp @ Zp.T
            i_idx = pt[:, 0]; j_idx = pt[:, 1]
            neg = torch.randint(0, J, (len(pt), N_NEG))
            pos = S[i_idx, j_idx].unsqueeze(1)
            negs = S[i_idx.unsqueeze(1).expand(-1, N_NEG), neg]
            logits = torch.cat([pos, negs], dim=1) / TAU
            loss = -F.log_softmax(logits, dim=1)[:, 0].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        Wn = W.detach().numpy()
        Za = cosine(E @ Wn)

        for name, Z in (("frozen", E), ("aligned", Za)):
            Sfull = Z @ Z.T
            aucs = []
            for b in test_books:
                partners = {j for j in nbrs[b] if j not in train_set}
                universe = [j for j in test_books if j != b]
                v = auc(Sfull[b], partners, universe)
                if v is not None:
                    aucs.append(v)
            results.append(dict(train_size=m, method=name,
                                auc=round(float(np.mean(aucs)), 4),
                                n_eval=len(aucs)))
            print(f"m={m} {name}: AUC={np.mean(aucs):.4f} (n={len(aucs)})",
                  flush=True)

    # 曲线
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, color, marker in (("frozen", "#27ae60", "s"),
                                ("aligned", "#c0392b", "o")):
        pts = sorted([(r["train_size"], r["auc"]) for r in results
                      if r["method"] == name])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker=marker, color=color, label=
                    "冻结嵌入(水平参考)" if name == "frozen" else "对齐投影")
    ax.set_xlabel("训练书数 m")
    ax.set_ylabel("留出书共购伙伴排序 AUC")
    ax.set_title("Amazon Books 交叉验证: 对齐 vs 冻结 (冷启动)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig11_amazon.png"))
    with open(os.path.join(RES, "amazon_scaling.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[fig11] saved")


if __name__ == "__main__":
    from collections import defaultdict
    sys.exit(main())
