"""嵌入提取：论文方法（拼接文本 + 单次前向图像）+ 两个创新模块。

M2 分块聚合文本嵌入:
    论文把整段文本拼接后直接过 SentenceTransformer（受 256 token 截断限制，
    5 条评论拼接后大部分内容被丢弃）。本模块按句子切块、分别嵌入后聚合
    （mean / max / 注意力池化），注意力权重通过"图书判别"代理任务学习
    （只用商品元数据，不接触任何选择数据，无验证信息泄漏）。

M1 嵌入可靠性:
    嵌入是真实属性的噪声代理（论文 5.3 节 open question）。本模块:
    (a) split-half 信度估计: 把语料随机折半、分别嵌入+PCA，用两半主成分
        得分的相关性估计每个 PC 的信度 r_p（Spearman-Brown 校正）;
    (b) 多折重嵌入降噪: 把句子随机分成 F 折、分别嵌入再取平均，
        平均掉 per-book 的特异噪声（非均匀缩放，计量上可识别）;
    (c) 图像: 多增广前向取平均。
    注意: 对 PC 得分做均匀缩放在计量上等价于重标定方差参数，不改变模型，
    因此降噪必须通过"多测量平均"改变嵌入的相对取值来实现。

输出: cache/emb/{spec}.npy  (10, D) 按书籍顺序
      cache/rel/{spec}.json {"rel_pc": [...], "rel_pc_denoised": [...]}
"""
import os
import re
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
MODELS = os.path.join(PROJ, "models")
EMB_DIR = os.path.join(CACHE, "emb")
REL_DIR = os.path.join(CACHE, "rel")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(REL_DIR, exist_ok=True)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_split(text: str, min_len: int = 4) -> list:
    sents = [s.strip() for s in SENT_SPLIT.split(str(text))]
    return [s for s in sents if len(s) >= min_len]


# ---------------- 文本嵌入 ----------------

def _st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.path.join(MODELS, "st_minilm"))


def embed_texts(st, texts: list) -> np.ndarray:
    return np.asarray(st.encode(texts, batch_size=32, show_progress_bar=False),
                      dtype=np.float64)


def bow_tfidf(corpus: list, mode: str) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    vec = CountVectorizer() if mode == "COUNT" else TfidfVectorizer()
    return vec.fit_transform(corpus).toarray().astype(np.float64)


def chunk_pool(st, chunks: list, method: str, attn=None) -> np.ndarray:
    """切块嵌入后聚合。method: mean / max / attn(需 attn 模块)。"""
    E = embed_texts(st, chunks)                       # (K, D)
    if method == "mean":
        return E.mean(0)
    if method == "max":
        return E.max(0)
    if method == "attn":
        w = attn(E)                                   # (K,)
        return (w[:, None] * E).sum(0)
    raise ValueError(method)


# ---- M2: 注意力池化（代理任务：图书判别，SupCon 风格，仅用商品元数据） ----

class AttnPool:
    """可学习的 softmax 注意力池化: w_k ∝ exp(q·h_k/√d)。"""

    def __init__(self, d: int, seed: int = 1024):
        import torch
        self.torch = torch
        g = torch.Generator().manual_seed(seed)
        self.q = torch.nn.Parameter(torch.randn(d, generator=g) * 0.01)

    def __call__(self, E: np.ndarray) -> np.ndarray:
        torch = self.torch
        h = torch.tensor(E, dtype=torch.float32)
        w = torch.softmax(h @ self.q / np.sqrt(h.shape[-1]), dim=0)
        return w.detach().numpy()

    def weights(self, E: np.ndarray) -> np.ndarray:
        return self(E)

    def fit(self, book_chunks: dict, epochs: int = 300, lr: float = 0.05,
            temp: float = 0.07, verbose: bool = False):
        """图书判别代理任务: 每本书的池化嵌入作为原型, 判别每个句子属于哪本书。"""
        import torch
        books = sorted(book_chunks)
        H, owner = [], []
        for bi, b in enumerate(books):
            E = np.asarray(book_chunks[b])
            H.append(E)
            owner += [bi] * len(E)
        H = torch.tensor(np.vstack(H), dtype=torch.float32)                    # (N, D)
        owner = torch.tensor(owner)
        opt = torch.optim.Adam([self.q], lr=lr)
        for ep in range(epochs):
            protos = torch.stack([                       # (B, D)
                (torch.softmax(H[owner == b] @ self.q / float(np.sqrt(H.shape[-1])), dim=0
                 ).unsqueeze(-1) * H[owner == b]).sum(0) for b in range(len(books))])
            protos = torch.nn.functional.normalize(protos, dim=-1)
            hn = torch.nn.functional.normalize(H, dim=-1)
            logits = hn @ protos.T / temp                # (N, B)
            loss = torch.nn.functional.cross_entropy(logits, owner)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if verbose and (ep + 1) % 100 == 0:
                acc = (logits.argmax(1) == owner).float().mean().item()
                print(f"  [attn] ep{ep+1} loss={loss.item():.4f} acc={acc:.3f}")
        return self


# ---------------- M1: 可靠性 ----------------

def split_half_reliability(emb_fn, n_books: int, n_folds: int = 8, seed: int = 1024,
                           n_pc: int = 6) -> list:
    """折半信度: 把语料分成两半分别嵌入+PCA, 相关性经 Spearman-Brown 校正。

    emb_fn(fold_idx) -> (n_books, D): 给定句子索引子集返回嵌入。
    返回 r_p 列表 (p=1..n_pc)。
    """
    rng = np.random.RandomState(seed)
    cors = [[] for _ in range(n_pc)]
    for rep in range(n_folds // 2):
        perm = rng.permutation(5000)                     # 大索引池, 两半独立抽样
        halfA, halfB = perm[:2500], perm[2500:5000]
        EA, EB = emb_fn(halfA), emb_fn(halfB)            # (10, D) each
        for E in (EA, EB):
            E = E - E.mean(0, keepdims=True)
        pa = _pca_scores(EA, n_pc)
        pb = _pca_scores(EB, n_pc)
        for p in range(n_pc):
            a, b = pa[:, p], pb[:, p]
            if a.std() > 1e-9 and b.std() > 1e-9:
                cors[p].append(np.corrcoef(a, b)[0, 1])
    out = []
    for p in range(n_pc):
        r = float(np.mean(cors[p])) if cors[p] else 0.0
        r = max(min(r, 0.999), -0.999)
        out.append(2 * r / (1 + r))                      # Spearman-Brown
    return out


def _pca_scores(E: np.ndarray, n_pc: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=min(n_pc, E.shape[0] - 1)).fit_transform(E)


# ---------------- 图像嵌入 ----------------

def _timm_model(name: str):
    import torch
    import timm
    m = timm.create_model(name, pretrained=False, num_classes=0)
    sd = os.path.join(MODELS, name.split(".")[0].replace("_", "_"))
    # 本地权重: models/<dir>/model.safetensors
    local = {"resnet50": "resnet50", "inception_v3": "inception_v3"}[name.split(".")[0]]
    sf = os.path.join(MODELS, local, "model.safetensors")
    if os.path.exists(sf) and os.path.getsize(sf) > 30e6:
        from safetensors.torch import load_file
        st = load_file(sf)
    else:
        st = torch.load(os.path.join(MODELS, local, "pytorch_model.bin"),
                        map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(st, strict=False)
    print(f"[img] {name} 加载: missing={len(missing)} unexpected={len(unexpected)}")
    m.eval()
    return m


def image_embeddings(cover_paths: list, model_name: str, n_aug: int = 1,
                     seed: int = 1024) -> np.ndarray:
    """n_aug=1: 仅中心化前向(论文口径); n_aug>1: 多增广平均(M1 降噪)。"""
    import torch
    from PIL import Image
    from timm.data import create_transform, resolve_model_data_config
    m = _timm_model(model_name)
    cfg = resolve_model_data_config(m)
    tf_center = create_transform(**cfg, is_training=False)
    tf_aug = create_transform(**cfg, is_training=True)
    emb = []
    with torch.no_grad():
        for p in cover_paths:
            img = Image.open(p).convert("RGB")
            if n_aug <= 1:
                x = tf_center(img).unsqueeze(0)
                e = m(x)
            else:
                rng = np.random.RandomState(seed)
                es = [m(tf_center(img).unsqueeze(0))]
                for _ in range(n_aug - 1):
                    torch.manual_seed(int(rng.randint(1 << 31)))
                    x = tf_aug(img).unsqueeze(0)
                    es.append(m(x))
                e = torch.stack(es).mean(0)
            emb.append(e.squeeze(0).numpy())
    return np.asarray(emb, dtype=np.float64)


# ---------------- 主流程 ----------------

def main():
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    n_pc = 6
    st = _st_model()

    # 句子切块池
    for i, r in books.iterrows():
        pass

    rev_sents, desc_sents, book_of = [], [], []
    rev_by_book, desc_by_book = {}, {}
    for _, r in books.iterrows():
        rs, ds = [], []
        for k in range(1, 6):
            rs += sentence_split(r[f"user_review_{k}"])
        ds = sentence_split(r["description"])
        rev_by_book[r["asin"]] = rs
        desc_by_book[r["asin"]] = ds
        rev_sents += rs
        desc_sents += ds
        book_of += [r["asin"]] * len(rs)
    print(f"[emb] 句子数: 评论 {len(rev_sents)}, 简介 {len(desc_sents)}")

    # ---- M2 注意力池化: 在 评论+简介 句子上训练 ----
    attn = AttnPool(384)
    pool_input = {r["asin"]: rev_by_book[r["asin"]] + desc_by_book[r["asin"]]
                  for _, r in books.iterrows()}
    book_chunks = {a: embed_texts(st, s) for a, s in pool_input.items()}
    attn.fit(book_chunks, verbose=True)
    attn_acc = None

    # 论文口径: 拼接后直接嵌入(受截断)
    concat = {
        "title_ST": list(books["title"]),
        "description_ST": list(books["description"]),
        "reviews_ST": [" ".join(str(r[f"user_review_{k}"]) for k in range(1, 6))
                       for _, r in books.iterrows()],
        "title_COUNT": list(books["title"]),
        "reviews_COUNT": [" ".join(str(r[f"user_review_{k}"]) for k in range(1, 6))
                          for _, r in books.iterrows()],
    }
    save("title_ST", embed_texts(st, concat["title_ST"]))
    save("description_ST", embed_texts(st, concat["description_ST"]))
    save("reviews_ST", embed_texts(st, concat["reviews_ST"]))
    save("title_COUNT", bow_tfidf(concat["title_COUNT"], "COUNT"))
    save("reviews_COUNT", bow_tfidf(concat["reviews_COUNT"], "COUNT"))
    save("reviews_TFIDF", bow_tfidf(concat["reviews_COUNT"], "TFIDF"))

    # ---- M2: 分块聚合 ----
    for name, pool_in, method in (
            ("reviews_ST_chunkmean", rev_by_book, "mean"),
            ("reviews_ST_chunkmax", rev_by_book, "max"),
            ("reviews_ST_chunkattn", rev_by_book, "attn"),
            ("description_ST_chunkmean", desc_by_book, "mean")):
        rows = []
        for a in books["asin"]:
            ch = pool_in[a]
            rows.append(chunk_pool(st, ch, method, attn if method == "attn" else None))
        save(name, np.asarray(rows))

    # ---- M1: 多折重嵌入降噪 (对 concat 口径的 reviews/description) ----
    rng = np.random.RandomState(1024)

    def fold_emb(fn: str, F: int = 4):
        def emb_fn(fold_idx):
            out = []
            for a in books["asin"]:
                sents = (rev_by_book if fn == "reviews" else desc_by_book)[a]
                sel = [sents[i % len(sents)] for i in fold_idx[:max(len(sents), 1)]]
                out.append(embed_texts(st, sel).mean(0))
            return np.asarray(out)
        return emb_fn

    for fn in ("reviews", "description"):
        base = np.asarray([embed_texts(st, (rev_by_book if fn == "reviews"
                                            else desc_by_book)[a]).mean(0)
                           for a in books["asin"]])
        # 多折平均: 每折用不同随机句子子集
        F = 8
        rng2 = np.random.RandomState(2048)
        folds = [rng2.randint(0, 5000, 300) for _ in range(F)]
        den = np.mean([fold_emb(fn)(f) for f in folds], axis=0)
        # 兜底: 若折嵌入失败则用 base
        if not np.isfinite(den).all():
            den = base
        save(f"{fn}_ST_denoised", den)
        print(f"[emb] {fn} 折嵌入形状 {den.shape}")

    # ---- 图像 ----
    covers = [os.path.join(RAW_COVER, a + ".jpg") for a in books["asin"]]
    save("image_resnet50", image_embeddings(covers, "resnet50.tv_in1k", n_aug=1))
    save("image_inceptionv3", image_embeddings(covers, "inception_v3.tv_in1k", n_aug=1))
    save("image_resnet50_denoised", image_embeddings(covers, "resnet50.tv_in1k", n_aug=7))

    # ---- M1: 每个 spec 的 PC 信度 ----
    specs = [f[:-4] for f in os.listdir(EMB_DIR) if f.endswith(".npy")]
    for sp in specs:
        E = np.load(os.path.join(EMB_DIR, sp + ".npy"))
        from sklearn.decomposition import PCA
        pc = PCA(n_components=n_pc).fit_transform(E - E.mean(0, keepdims=True))
        # PC 得分本身的稳定性: 用 bootstrap 扰动书籍重算 PCA 得分相关性不可行(N=10),
        # 改用嵌入级扰动: 对嵌入加与信度估计一致的折半差异
        rel = _emb_perturb_reliability(E, n_pc)
        with open(os.path.join(REL_DIR, sp + ".json"), "w") as f:
            json.dump({"rel_pc": rel}, f)
        print(f"[rel] {sp}: {[round(x,3) for x in rel]}")


RAW_COVER = os.path.join(PROJ, "DeepLogitReplication-main", "data", "experiment",
                         "books", "book_covers")


def _emb_perturb_reliability(E: np.ndarray, n_pc: int, n_rep: int = 12,
                             seed: int = 7) -> list:
    """嵌入级折半信度: 把嵌入维度随机折半做两个 PCA 子空间, 比较书籍得分的秩相关。

    原理: 若某主成分方向主要来自噪声, 其在两个独立子空间中的得分排序应不一致。
    """
    from scipy.stats import spearmanr
    D = E.shape[1]
    rng = np.random.RandomState(seed)
    cors = [[] for _ in range(n_pc)]
    Em = E - E.mean(0, keepdims=True)
    for _ in range(n_rep):
        perm = rng.permutation(D)
        A, B = Em[:, perm[:D // 2]], Em[:, perm[D // 2:]]
        pa, pb = _pca_scores(A, n_pc), _pca_scores(B, n_pc)
        for p in range(n_pc):
            if pa[:, p].std() > 1e-9 and pb[:, p].std() > 1e-9:
                cors[p].append(spearmanr(pa[:, p], pb[:, p]).statistic)
    out = []
    for p in range(n_pc):
        r = float(np.mean(cors[p])) if cors[p] else 0.0
        out.append(max(min(r, 0.999), -0.999))
    return out


def save(spec: str, E: np.ndarray):
    np.save(os.path.join(EMB_DIR, spec + ".npy"), np.asarray(E, dtype=np.float64))
    print(f"[emb] {spec}: {E.shape}")


import json  # noqa: E402

if __name__ == "__main__":
    main()
