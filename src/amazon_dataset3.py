# -*- coding: utf-8 -*-
"""Dataset 3: Amazon Video Games co-review external validation.

Items: top-200 games by review count.
Diversion proxy: row-normalized co-review weights (users who reviewed
both games), threshold MIN_EDGES, from vg_coreview.json (2.75M users).
Embeddings + protocol: identical to datasets 1-2 (MiniLM, LOO,
tau_eval = 0.3, product-level supervision, bounded residual).
Outputs: results/amazon_dataset3.json.
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
MIN_EDGES = 3  # minimum co-review weight to keep an edge


def unit(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-10, None)


def build_graph():
    g = json.load(open(os.path.join(AMZ, "vg_coreview.json")))
    g["edges"] = [e for e in g["edges"] if e[2] >= MIN_EDGES]
    g["n_edges"] = len(g["edges"])
    print(f"graph: {len(g['n_items'])} items, {g['n_edges']} edges "
          f"(weight >= {MIN_EDGES}; users={g['n_users']})", flush=True)
    return g


def load_texts(asins):
    cache = os.path.join(CACHE, "vg_texts.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    want = set(asins)
    texts = {}
    with open(os.path.join(AMZ, "vg_meta_Video_Games.jsonl"),
              encoding="utf-8", errors="replace") as f:
        for line in f:
            if line[0] != "{":
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            a = d.get("parent_asin")
            if a not in want or a in texts:
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
    json.dump(texts, open(cache, "w", encoding="utf-8"))
    return texts


def embed(texts, asins):
    cache_npy = os.path.join(CACHE, "vg_item_emb.npy")
    if os.path.exists(cache_npy):
        return np.load(cache_npy)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL_DIR, device="cpu")
    vecs = st.encode([texts.get(a) or "unknown" for a in asins],
                     batch_size=32, show_progress_bar=False,
                     normalize_embeddings=False)
    E = np.asarray(vecs, dtype=np.float32)
    np.save(cache_npy, E)
    return E


def build_diversion(asins, g):
    idx = {a: i for i, a in enumerate(asins)}
    n = len(asins)
    W = np.zeros((n, n), dtype=np.float64)
    for u, v, w in g["edges"]:
        if u in idx and v in idx and u != v:
            W[idx[u], idx[v]] += w
            W[idx[v], idx[u]] += w
    return W / np.clip(W.sum(1, keepdims=True), 1e-12, None)


def train_anchor(E, S, keep, delta, epochs=EPOCHS, lr=LR, seed=SEED):
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
    t0 = time.time()
    g = build_graph()
    asins = g["n_items"]
    texts = load_texts(asins)
    n_have = sum(1 for a in asins if texts.get(a))
    print(f"texts: {n_have}/{len(asins)} ({time.time()-t0:.0f}s)", flush=True)
    Eraw = embed(texts, asins)
    S = build_diversion(asins, g)
    n = len(asins)

    # exclude items with empty diversion row (no edges)
    has = S.sum(1) > 1e-12
    asins2 = [a for a, h in zip(asins, has) if h]
    Eraw = Eraw[has]
    S = S[np.ix_(has, has)]
    n = len(asins2)
    print(f"items with diversion rows: {n}", flush=True)

    Eu = unit(Eraw)
    C = Eu @ Eu.T
    iu = np.triu_indices(n, 1)
    r_p = pearsonr(C[iu], S[iu])
    r_s = spearmanr(C[iu], S[iu])
    print(f"cosine vs co-review: pearson={r_p.statistic:.4f} "
          f"spearman={r_s.statistic:.4f}", flush=True)

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
        if (j0 + 1) % 50 == 0:
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
    print(f"\n=== Video Games (dataset 3, LOO n={n}) ===")
    for m, v in methods_out.items():
        print(f"  {m:6s} rmse={v['rmse']:.4f}±{v['rmse_sd']:.4f}  "
              f"tau={v['kendall']:.4f}±{v['kendall_sd']:.4f}  top3={v['top3']:.2f}")

    tests = dict(
        rmse_frozen_vs_anchored_p=perm_test(
            [x["rmse"] for x in rows["frozen"]],
            [x["rmse"] for x in rows["a0.5"]], "less"),
    )
    print("tests:", tests)

    out = dict(
        dataset="amazon_video_games_coreview",
        n_items=n, n_edges=g["n_edges"], n_users=g["n_users"],
        tau_eval=TAU_EVAL, epochs=EPOCHS,
        semantic_vs_coreview=dict(pearson=float(r_p.statistic),
                                  pearson_p=float(r_p.pvalue),
                                  spearman=float(r_s.statistic)),
        methods=methods_out, tests=tests,
        per_fold={m: rows[m] for m in methods},
    )
    json.dump(out, open(os.path.join(RES, "amazon_dataset3.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"saved ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
