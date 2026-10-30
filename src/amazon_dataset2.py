# -*- coding: utf-8 -*-
"""Dataset 2: Amazon Books co-purchase external validation (generalization).

Items: 175 bestseller books with text in the downloaded meta prefix.
Diversion proxy: row-normalized co-purchase weights (induced subgraph,
6,906 edges from 2.49M users; symmetric).
Embeddings: same local SentenceTransformer as the main study (384-d).
Protocol: identical to the main LOO pipeline — leave one item out,
tau_eval = 0.3, product-level supervision on remaining rows.
Methods: uniform / frozen / anchored(delta grid) / unbounded (delta=inf).
Outputs: results/amazon_dataset2.json (per-fold values + permutation tests).
"""
import os, json, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
MODEL_DIR = os.path.join(PROJ, "models", "st_minilm")
TAU_EVAL = 0.3
EPOCHS = 800
LR = 1e-2
DELTAS = [0.1, 0.3, 0.5, 0.7]
SEED = 2026


def load_texts(asins):
    """Extract title+description+features for the wanted asins from meta prefix."""
    cache = os.path.join(CACHE, "amazon_texts.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    want = set(asins)
    texts = {}
    for fn in ["meta_books_head.jsonl", "meta_books_full.jsonl"]:
        with open(os.path.join(AMZ, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                if line[0] != "{":
                    continue
                i = line.find('"parent_asin"')
                if i < 0:
                    continue
                j = line.find('"', i + 14)
                k = line.find('"', j + 1)
                a = line[j + 1:k] if 0 < j < k else None
                if a not in want or a in texts:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                parts = [d.get("title") or ""]
                desc = d.get("description")
                if isinstance(desc, list):
                    parts.append(" ".join(x for x in desc if isinstance(x, str)))
                elif isinstance(desc, str):
                    parts.append(desc)
                feats = d.get("features")
                if isinstance(feats, list):
                    parts.append(" ".join(x for x in feats if isinstance(x, str)))
                texts[a] = " [SEP] ".join(p for p in parts if p.strip())[:4000]
        print(f"  after {fn}: {len(texts)} texts", flush=True)
    json.dump(texts, open(cache, "w", encoding="utf-8"))
    return texts


def embed(texts, asins):
    """Embed item texts with the local MiniLM (same encoder as main study)."""
    cache_npy = os.path.join(CACHE, "amazon_item_emb.npy")
    if os.path.exists(cache_npy):
        return np.load(cache_npy)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL_DIR, device="cpu")
    vecs = st.encode([texts[a] if texts.get(a) else "unknown" for a in asins],
                     batch_size=32, show_progress_bar=False,
                     normalize_embeddings=False)
    E = np.asarray(vecs, dtype=np.float32)
    np.save(cache_npy, E)
    return E


def build_diversion(asins):
    """Symmetric co-purchase weight matrix, row-normalized."""
    g = json.load(open(os.path.join(AMZ, "copurchase_edges.json")))
    idx = {a: i for i, a in enumerate(asins)}
    n = len(asins)
    W = np.zeros((n, n), dtype=np.float64)
    for u, v, w in g["edges"]:
        if u in idx and v in idx and u != v:
            W[idx[u], idx[v]] += w
            W[idx[v], idx[u]] += w
    S = W / np.clip(W.sum(1, keepdims=True), 1e-12, None)
    return S


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-10, None)


def train_anchor(E, S, keep, delta, epochs=EPOCHS, lr=LR, seed=SEED):
    """Bounded-residual product-level training (identical to main pipeline)."""
    torch.manual_seed(seed)
    n = len(keep)
    P = torch.nn.Parameter(torch.zeros(n, E.shape[1]))
    opt = torch.optim.Adam([P], lr=lr)
    Ct = torch.tensor(E[keep], dtype=torch.float32)
    St = torch.tensor(S[np.ix_(keep, keep)], dtype=torch.float32)
    for _ in range(epochs):
        Z = F.normalize(Ct + P, dim=1)
        lp = F.log_softmax(Z @ Z.T / TAU_EVAL, dim=1)
        tgt = St / St.sum(1, keepdims=True).clamp_min(1e-10)
        loss = -(tgt * lp).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if np.isfinite(delta):
                nn_ = P.norm(dim=1, keepdim=True).clamp_min(1e-12)
                P.mul_(torch.clamp(nn_, max=delta) / nn_)
    with torch.no_grad():
        return F.normalize(Ct + P, dim=1).numpy()


def pred_col(Z, keep, j0):
    Sm = Z @ Z.T
    out = []
    for i in keep:
        r = Sm[i] / TAU_EVAL
        r[i] = -1e9
        e = np.exp(r - r.max())
        out.append(e[j0] / e.sum())
    return np.array(out)


def met(p, o):
    order = np.argsort(-p)
    best = int(np.argmax(o))
    return dict(rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                kendall=float(kendalltau(p, o).statistic),
                top3=int(best in order[:3]))


def perm_test(a, b, alternative="less", n_samples=20000, seed=7):
    """Paired sign-flip permutation. Exact enumeration when n <= 16,
    otherwise Monte-Carlo sampling of n_samples random sign vectors."""
    d = np.array(a) - np.array(b)
    n = len(d)
    obs = d.mean()
    if n <= 16:
        better = total = 0
        for m in range(1 << n):
            s = 0.0
            for i in range(n):
                s += d[i] if (m >> i) & 1 else -d[i]
            better += (s / n >= obs) if alternative == "greater" else (s / n <= obs)
            total += 1
        return better / total
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_samples, n))
    stats = (signs * d).mean(1)
    if alternative == "greater":
        return float((stats >= obs).mean())
    return float((stats <= obs).mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="copurchase_edges.json",
                    help="边表文件名；--edges copurchase_edges_full.json 跑无截断敏感性版")
    ap.add_argument("--tag", default="",
                    help="非空则输出 results/amazon_dataset2_<tag>.json，避免覆盖正式结果")
    a = ap.parse_args()

    t0 = time.time()
    g = json.load(open(os.path.join(AMZ, a.edges)))
    covered = json.load(open(os.path.join(AMZ, "covered_asins.json")))
    books = set(g["books"])
    asins = [x for x in covered if x in books]
    print(f"items: {len(asins)}  edges: {g['n_edges']:,}  ({a.edges})", flush=True)

    texts = load_texts(asins)
    print(f"texts: {len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    Eraw = embed(texts, asins)
    print(f"embedded: {Eraw.shape} ({time.time()-t0:.0f}s)", flush=True)

    S = build_diversion(asins)
    n = len(asins)

    # semantic-similarity vs co-purchase association (analogue of the r=0.64 story)
    Eu = unit(Eraw)
    C = Eu @ Eu.T
    iu = np.triu_indices(n, 1)
    r_pearson = pearsonr(C[iu], S[iu])
    r_spear = spearmanr(C[iu], S[iu])
    print(f"cosine vs co-purchase: pearson={r_pearson.statistic:.4f} "
          f"spearman={r_spear.statistic:.4f}", flush=True)

    methods = ["frozen"] + [f"a{d}" for d in DELTAS] + ["inf"]
    rows = {m: [] for m in methods}
    for j0 in range(n):
        keep = [i for i in range(n) if i != j0]
        obs = np.array([S[i, j0] for i in keep])
        rows["frozen"].append(met(pred_col(Eu, keep, j0), obs))
        for d in DELTAS:
            Z = Eu.copy()
            Z[keep] = train_anchor(Eu, S, keep, d)
            rows[f"a{d}"].append(met(pred_col(Z, keep, j0), obs))
        Z = Eu.copy()
        Z[keep] = train_anchor(Eu, S, keep, np.inf)
        rows["inf"].append(met(pred_col(Z, keep, j0), obs))
        if (j0 + 1) % 25 == 0:
            print(f"  fold {j0+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    def agg(v):
        return dict(
            rmse=float(np.mean([x["rmse"] for x in v])),
            rmse_sd=float(np.std([x["rmse"] for x in v], ddof=1)),
            kendall=float(np.mean([x["kendall"] for x in v])),
            kendall_sd=float(np.std([x["kendall"] for x in v], ddof=1)),
            top3=float(np.mean([x["top3"] for x in v])),
        )

    methods_out = {m: agg(rows[m]) for m in methods}
    print("\n=== Amazon Books (dataset 2, LOO n=%d) ===" % n)
    for m, v in methods_out.items():
        print(f"  {m:6s} rmse={v['rmse']:.4f}±{v['rmse_sd']:.4f}  "
              f"tau={v['kendall']:.4f}±{v['kendall_sd']:.4f}  top3={v['top3']:.2f}")

    fr = [x["rmse"] for x in rows["frozen"]]
    an = [x["rmse"] for x in rows["a0.5"]]
    tests = dict(
        rmse_frozen_vs_anchored_p=perm_test(fr, an, "less"),
        rmse_frozen_vs_unbounded_p=perm_test(fr, [x["rmse"] for x in rows["inf"]], "less"),
    )
    print("tests:", tests)

    out = dict(
        dataset="amazon_books_copurchase",
        n_items=n, n_edges=int((S > 0).sum() // 2),
        n_users_multi=g.get("n_users_multi"),
        tau_eval=TAU_EVAL, epochs=EPOCHS,
        semantic_vs_copurchase=dict(pearson=float(r_pearson.statistic),
                                    pearson_p=float(r_pearson.pvalue),
                                    spearman=float(r_spear.statistic)),
        methods=methods_out, tests=tests,
        per_fold={m: rows[m] for m in methods},
    )
    outname = f"amazon_dataset2_{a.tag}.json" if a.tag else "amazon_dataset2.json"
    out = dict(edges_file=a.edges, **out)
    json.dump(out, open(os.path.join(RES, outname), "w"),
              ensure_ascii=False, indent=1)
    print(f"saved {outname} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
