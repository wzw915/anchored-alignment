"""汇总实验结果: 主表 + 论文级图表。

图:
  fig1_dataset.png   数据集刻画: 经验 diversion 热力图 + 一选份额 + PC 空间书籍分布
  fig2_rmse.png      各规格二选 RMSE 对比 (论文图2风格)
  fig3_m2.png        M2 消融: 拼接 vs 分块聚合 (mean/max/注意力)
  fig4_m1.png        M1 消融: 单次 vs 降噪 + PC 信度
  fig5_example.png   最近替代示例 (Dopamine Detox 行)
  fig6_topk.png      top-1 / top-3 命中率
表:
  results/summary.json  全部规格汇总
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
CACHE = os.path.join(PROJ, "cache")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.dpi": 140, "savefig.bbox": "tight",
    "font.size": 9.5,
})
C = dict(base="#4C78A8", ours="#D1495B", alt="#7A9E7E", gray="#9AA0A6", dark="#2F3437")

DISPLAY = {
    "plain": "Plain Logit", "attributes": "属性 Mixed Logit",
    "title_ST": "标题(ST)", "title_COUNT": "标题(BoW)",
    "description_ST": "简介(ST 拼接)", "description_ST_chunkmean": "简介(ST 分块均值)",
    "description_ST_denoised": "简介(ST 降噪★M1)",
    "reviews_ST": "评论(ST 拼接)", "reviews_COUNT": "评论(BoW)", "reviews_TFIDF": "评论(TF-IDF)",
    "reviews_ST_chunkmean": "评论(ST 分块均值)", "reviews_ST_chunkmax": "评论(ST 分块max)",
    "reviews_ST_chunkattn": "评论(ST 注意力池化★M2)",
    "reviews_ST_denoised": "评论(ST 降噪★M1)",
    "image_resnet50": "封面(ResNet50)", "image_inceptionv3": "封面(InceptionV3)",
    "image_resnet50_denoised": "封面(ResNet50 降噪★M1)",
}
ORDER = list(DISPLAY)


def load_all():
    rows = []
    for f in sorted(os.listdir(RES)):
        if f.startswith("model_") and f.endswith(".json"):
            d = json.load(open(os.path.join(RES, f), encoding="utf-8"))
            if "error" in d:
                print(f"[summary] {f} 出错, 跳过")
                continue
            rows.append(d)
    df = pd.DataFrame(rows)
    df["disp"] = df["spec"].map(DISPLAY)
    df["d_aic"] = df["aic"] - df.loc[df["spec"] == "plain", "aic"].iloc[0]
    base_rmse = df.loc[df["spec"] == "plain", "rmse"].iloc[0]
    df["rmse_cut"] = (base_rmse - df["rmse"]) / base_rmse * 100
    return df


def fig1(df):
    emp = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

    ax = axes[0]
    short = {3: "PleaseTellMe", 4: "Housemaid", 8: "Ritual", 9: "Inmate",
             21: "Serpent", 23: "Ashes", 29: "CourtRavens", 45: "Dopamine",
             46: "DontBelieve", 47: "ArtLettingGo"}
    labels = [short[p] for p in sorted(books["product_id"])]
    im = ax.imshow(emp, cmap="Blues", vmin=0, vmax=0.5)
    ax.set_xticks(range(10), labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(10), labels, fontsize=6.5)
    ax.set_title("(a) 经验二选 diversion 矩阵\nP(第二选=k | 第一选=j)", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    share = first[first["choice"] == 1]["product_id"].value_counts().reindex(
        sorted(books["product_id"])).to_numpy()
    share = share / share.sum()
    genre_color = {"Science Fiction & Fantasy": C["base"],
                   "Mystery, Thriller & Suspense": C["ours"],
                   "Self-Help": C["alt"]}
    colors = [genre_color[g] for g in books["genre"]]
    ax.barh(range(10), share, color=colors)
    ax.set_yticks(range(10), labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_title("(b) 一选市场份额", fontsize=9.5)
    ax.set_xlabel("份额")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v, label=k.split(",")[0])
               for k, v in genre_color.items()]
    ax.legend(handles=handles, fontsize=6.5, loc="lower right")

    ax = axes[2]
    E = np.load(os.path.join(CACHE, "emb", "reviews_ST_chunkmean.npy"))
    from sklearn.decomposition import PCA
    P = PCA(n_components=2).fit_transform(E - E.mean(0, keepdims=True))
    for g, c in genre_color.items():
        m = (books["genre"] == g).to_numpy()
        ax.scatter(P[m, 0], P[m, 1], c=c, s=42, label=g.split(",")[0])
    for i, t in enumerate(books["title"]):
        ax.annotate(short[sorted(books["product_id"])[i]], (P[i, 0], P[i, 1]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_title("(c) 评论分块均值嵌入的 PC 空间\n(论文图4风格)", fontsize=9.5)
    ax.legend(fontsize=6.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_dataset.png"))
    plt.close()


def fig2(df):
    d = df.set_index("spec").reindex([s for s in ORDER if s in set(df["spec"])])
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    colors = [C["gray"] if s in ("plain", "attributes") else
              (C["ours"] if ("denoised" in s or "chunkattn" in s) else C["base"])
              for s in d.index]
    bars = ax.bar(range(len(d)), d["rmse"], color=colors)
    base = df.loc[df["spec"] == "plain", "rmse"].iloc[0]
    ax.axhline(base, color=C["dark"], ls="--", lw=0.8)
    ax.text(len(d) - 0.4, base + 0.001, "plain logit 基线", fontsize=7.5, ha="right")
    for i, (rmse, cut) in enumerate(zip(d["rmse"], d["rmse_cut"])):
        ax.text(i, rmse + 0.0008, f"{cut:+.0f}%", ha="center", fontsize=7)
    ax.set_xticks(range(len(d)), [DISPLAY[s] for s in d.index],
                  rotation=38, ha="right", fontsize=7.5)
    ax.set_ylabel("二选 diversion RMSE (越低越好)")
    ax.set_title("各规格的反事实第二选择预测误差 (论文式(2)口径)", fontsize=10.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
               ((C["gray"], "基准"), (C["base"], "论文方法"),
                (C["ours"], "本文创新(★)"))]
    ax.legend(handles=handles, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_rmse.png"))
    plt.close()


def fig3(df):
    """M2 消融: 评论嵌入的四种构造方式。"""
    specs = ["reviews_ST", "reviews_COUNT", "reviews_TFIDF",
             "reviews_ST_chunkmean", "reviews_ST_chunkmax", "reviews_ST_chunkattn"]
    d = df.set_index("spec").reindex([s for s in specs if s in set(df["spec"])])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    labels = [DISPLAY[s] for s in d.index]
    colors = [C["gray"] if "BoW" in l or "TF-IDF" in l else
              (C["ours"] if "注意力" in l else C["base"]) for l in labels]
    ax.barh(range(len(d)), d["rmse"], color=colors)
    ax.set_yticks(range(len(d)), labels, fontsize=8)
    ax.invert_yaxis()
    for i, (r, c) in enumerate(zip(d["rmse"], d["rmse_cut"])):
        ax.text(r + 0.0005, i, f"{c:+.0f}%", va="center", fontsize=7.5)
    ax.set_xlabel("二选 RMSE")
    ax.set_title("(a) 文本构造方式对替代模式恢复的影响", fontsize=9.5)

    ax = axes[1]
    plain = df.loc[df["spec"] == "plain", "rmse"].iloc[0]
    # token 覆盖率示意: 拼接截断 vs 分块
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(os.path.join(PROJ, "models", "st_minilm"))
    concat_lens, chunk_cov = [], []
    for _, r in books.iterrows():
        full = " ".join(str(r[f"user_review_{k}"]) for k in range(1, 6))
        n_tok = len(st.tokenizer(full)["input_ids"])
        concat_lens.append(min(n_tok, 256) / n_tok * 100)
        per = [len(st.tokenizer(str(r[f"user_review_{k}"]))["input_ids"])
               for k in range(1, 6)]
        chunk_cov.append(sum(min(p, 256) for p in per) / n_tok * 100)
    ax.bar([0, 1], [np.mean(concat_lens), np.mean(chunk_cov)],
           color=[C["base"], C["ours"]], width=0.5)
    for i, v in enumerate([np.mean(concat_lens), np.mean(chunk_cov)]):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["论文口径:\n5条评论拼接(截断256)", "本文M2:\n分块嵌入后聚合"])
    ax.set_ylabel("进入嵌入的有效 token 占比")
    ax.set_title("(b) 截断损失: M2 恢复了被丢弃的文本", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_m2.png"))
    plt.close()


def fig4(df):
    """M1 消融: 降噪前后 + PC 信度。"""
    pairs = [("reviews_ST", "reviews_ST_denoised"),
             ("description_ST", "description_ST_denoised"),
             ("image_resnet50", "image_resnet50_denoised")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    d = df.set_index("spec")
    labels, vals, colors = [], [], []
    for a, b in pairs:
        if a in d.index and b in d.index:
            labels += [DISPLAY[a], DISPLAY[b]]
            vals += [d.loc[a, "rmse"], d.loc[b, "rmse"]]
            colors += [C["base"], C["ours"]]
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.0004, i, f"{v:.4f}", va="center", fontsize=7)
    ax.set_xlabel("二选 RMSE")
    ax.set_title("(a) M1 多折降噪前后对比", fontsize=9.5)

    ax = axes[1]
    specs = ["reviews_ST", "reviews_ST_denoised", "reviews_ST_chunkmean",
             "reviews_ST_chunkattn", "description_ST", "description_ST_denoised",
             "image_resnet50", "image_resnet50_denoised"]
    rel_dir = os.path.join(CACHE, "rel")
    xs, bottoms, hs, cs = [], [], [], []
    xpos = 0
    xticks = []
    for sp in specs:
        f = os.path.join(rel_dir, sp + ".json")
        if not os.path.exists(f):
            continue
        rel = json.load(open(f))["rel_pc"]
        for p, r in enumerate(rel):
            xs.append(xpos)
            bottoms.append(0)
            hs.append(max(r, 0))
            cs.append(C["ours"] if sp.endswith("denoised") else C["base"])
            xpos += 1
        xticks.append((xpos - 3, DISPLAY[sp], 0))
        xpos += 1.2
    ax.bar(xs, hs, color=cs, width=0.85)
    for x, lab, _ in xticks:
        ax.text(x, 0.78, lab, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("PC1-5 split-half 信度 (截断到0)")
    ax.set_title("(b) 各规格主成分的信度 (红=降噪后)", fontsize=9.5)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig4_m1.png"))
    plt.close()


def fig5(df):
    """最近替代示例: 第一选 = Dopamine Detox (对齐论文表2 Panel A)。"""
    emp = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    pids = sorted(books["product_id"])
    j = pids.index(45)  # Dopamine Detox
    short = {3: "PleaseTellMe(M)", 4: "Housemaid(M)", 8: "Ritual(M)", 9: "Inmate(M)",
             21: "Serpent(F)", 23: "Ashes(F)", 29: "CourtRavens(F)", 45: "Dopamine(S)",
             46: "DontBelieve(S)", 47: "ArtLettingGo(S)"}
    labels = [short[p] for p in pids]
    show = ["plain", "attributes", "reviews_ST", "reviews_ST_chunkattn",
            "reviews_ST_denoised"]
    mats = [("实验数据", emp[j])]
    for sp in show:
        f = os.path.join(RES, f"diversion_{sp}.npy")
        if os.path.exists(f):
            mats.append((DISPLAY[sp], np.load(f)[j]))
    fig, axes = plt.subplots(1, len(mats), figsize=(2.6 * len(mats), 3.9),
                             sharey=True)
    row0 = emp[j].copy()
    row0[j] = -1.0
    order = np.argsort(-row0)[:9]
    for ax, (name, row) in zip(axes, mats):
        ax.barh(range(9), row[order], color=C["base"])
        ax.set_yticks(range(9), [labels[o] for o in order], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(name, fontsize=8)
        ax.tick_params(axis="x", labelsize=6.5)
    fig.suptitle("第一选 = Dopamine Detox(自助类) 时的二选概率排序 (S=自助 M=悬疑 F=奇幻)",
                 fontsize=9.5, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_example.png"))
    plt.close()


def fig6(df):
    d = df.set_index("spec").reindex([s for s in ORDER if s in set(df["spec"])])
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["top1"], 0.38, label="top-1 命中率", color=C["base"])
    ax.bar(x + 0.2, d["top3"], 0.38, label="top-3 命中率", color=C["ours"])
    for i, (a, b) in enumerate(zip(d["top1"], d["top3"])):
        ax.text(i - 0.2, a + 0.01, f"{a:.0%}", ha="center", fontsize=6.5)
        ax.text(i + 0.2, b + 0.01, f"{b:.0%}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [DISPLAY[s] for s in d.index], rotation=38, ha="right",
                  fontsize=7.5)
    ax.set_ylabel("最近替代命中比例 (10本书)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("最近替代品识别命中率 (本文新增指标)", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig6_topk.png"))
    plt.close()


def fig7(df):
    """AIC 与反事实表现的相关: 复现论文 0.78 的口径层级问题。"""
    from scipy.stats import pearsonr

    d = df.dropna(subset=["aic", "rmse"]).copy()
    d["k"] = pd.to_numeric(d.get("best_k"), errors="coerce").fillna(0)
    est = d[d["k"] > 0]

    def grp(spec):
        if "denoised" in spec:
            return "M1 降噪"
        if "chunk" in spec:
            return "M2 分块"
        return "已估随机系数"

    est = est.assign(g=est["spec"].map(grp))
    k0 = d[d["k"] == 0]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(k0["aic"], k0["rmse"], s=46, c="#B4B2A9", edgecolors="#5F5E5A",
               linewidths=0.6, label="K=0（退回 plain，无随机系数）", zorder=2)
    colors = {"已估随机系数": "#1D9E75", "M2 分块": "#D85A30", "M1 降噪": "#BA7517"}
    for g, sub in est.groupby("g"):
        ax.scatter(sub["aic"], sub["rmse"], s=64, c=colors[g], edgecolors="white",
                   linewidths=0.7, label=g, zorder=3)

    marks = {"reviews_ST_chunkmean": (7, 1), "reviews_ST_denoised": (5, 5),
             "description_ST": (-7, 4), "reviews_ST": (5, 5)}
    for sp, off in marks.items():
        r = d[d["spec"] == sp].iloc[0]
        ha = "right" if off[0] < 0 else "left"
        ax.annotate(DISPLAY.get(sp, sp).replace("★M2", "").replace("★M1", ""),
                    (r["aic"], r["rmse"]), fontsize=7, xytext=off,
                    textcoords="offset points", ha=ha, color="#2C2C2A")

    r_all = pearsonr(d["aic"], d["rmse"])[0]
    r_est = pearsonr(est["aic"], est["rmse"])[0] if len(est) > 2 else float("nan")
    txt = (f"论文（40 类目之间）：r = 0.78\n"
           f"本文（全部 {len(d)} 规格）：r = {r_all:+.2f}\n"
           f"本文（{len(est)} 个含随机系数）：r = {r_est:+.2f}")
    ax.text(0.02, 0.34, txt, transform=ax.transAxes, fontsize=7.6, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F1EFE8", ec="#B4B2A9", lw=0.6),
            zorder=5)

    ax.set_xlabel("一选 AIC（越小越好）")
    ax.set_ylabel("二选 diversion RMSE（越小越好）")
    ax.set_title("AIC 能代表反事实表现吗？——单类目内部的检验", fontsize=10.5)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7_calibration.png"))
    plt.close()


def fig8():
    """裂缝→组件→验证: 同一候选池上 AIC 选型 vs CV 选型。"""
    f = os.path.join(RES, "cv_selection.json")
    if not os.path.exists(f):
        print("[fig8] 缺 cv_selection.json, 跳过")
        return
    d = json.load(open(f, encoding="utf-8"))
    pool = d.get("pool") or []
    if len(pool) < 3:
        print("[fig8] pool 不足, 跳过")
        return
    from scipy.stats import pearsonr

    aic = np.array([r["aic_full"] for r in pool])
    rmse = np.array([r["rmse_full"] for r in pool])
    cv = np.array([r["cv_ll"] for r in pool])
    POOL_DISP = {"plain": "Plain Logit", "attributes": "属性(pages)",
                 "reviews_ST_K1": "评论拼接 K=1", "reviews_ST_K3": "评论拼接 K=3",
                 "reviews_ST_P6_K4": "评论拼接 K=4(P6)", "reviews_ST_chunkmean_K3": "分块均值 K=3",
                 "reviews_ST_chunkattn_K4": "注意力池化 K=4", "reviews_ST_denoised_K2": "降噪 K=2(AIC冠军)",
                 "description_ST_K1": "简介拼接 K=1", "description_ST_denoised_K2": "简介降噪 K=2"}
    lab = [POOL_DISP.get(r["label"], r["label"]) for r in pool]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for ax, x, xlab, ttl in (
        (axes[0], aic, "一选 AIC（越小越好）", "(a) 现状: AIC 选型"),
        (axes[1], -cv, "5折 CV 负对数似然（越小越好）", "(b) 修复: CV 似然选型"),
    ):
        ax.scatter(x, rmse, s=60, c="#378ADD", edgecolors="white", linewidths=0.7)
        for xi, yi, li in zip(x, rmse, lab):
            ax.annotate(li, (xi, yi), fontsize=6.8, xytext=(4, 4),
                        textcoords="offset points", color="#2C2C2A")
        ax.set_xlabel(xlab)
        ax.set_ylabel("二选 diversion RMSE")
        ax.set_title(ttl, fontsize=10.5)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].text(0.03, 0.03, f"r = {pearsonr(aic, rmse)[0]:+.2f}",
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FCEBEB", ec="#A32D2D", lw=0.6))
    axes[1].text(0.03, 0.03, f"r = {pearsonr(-cv, rmse)[0]:+.2f}",
                 transform=axes[1].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#EAF3DE", ec="#3B6D11", lw=0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig8_cvfix.png"))
    plt.close()


def fig9():
    """冷启动陷阱: (a) few-shot 曲线 (b) LOO 方法对比条形图。数据 align_v2.json。"""
    f = os.path.join(RES, "align_v2.json")
    if not os.path.exists(f):
        print("[fig9] 缺 align_v2.json, 跳过")
        return
    d = json.load(open(f, encoding="utf-8"))
    fs = d.get("fewshot", {})
    res = d.get("results", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) few-shot 曲线
    ms = sorted(int(k) for k in fs if np.isfinite(fs[k].get("rmse", np.nan)))
    if ms:
        al = [fs[str(m)]["rmse"] for m in ms]
        ax1.plot(ms, al, "o-", color="#c0392b", label="对齐投影(个体级 InfoNCE)")
    if "frozen" in res:
        ax1.axhline(res["frozen"]["rmse"], color="#27ae60", ls="--",
                    label=f"冻结嵌入 {res['frozen']['rmse']:.3f}")
    if "uniform" in res:
        ax1.axhline(res["uniform"]["rmse"], color="#7f8c8d", ls=":",
                    label=f"均匀分配 {res['uniform']['rmse']:.3f}")
    ax1.set_xlabel("训练可见书本数 m")
    ax1.set_ylabel("LOO diversion 列 RMSE")
    ax1.set_title("(a) Few-shot 训练规模曲线")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    # (b) 方法对比
    DISP = {"frozen": "冻结嵌入", "uniform": "均匀分配", "zca": "ZCA 白化",
            "diag_metric": "对角度量学习", "linear_probing": "线性投影+InfoNCE"}
    names = ["frozen", "zca", "diag_metric", "uniform", "linear_probing"]
    names = [n for n in names if n in res]
    vals = [res[n]["rmse"] for n in names]
    kens = [res[n].get("kendall", np.nan) for n in names]
    colors = ["#27ae60" if n == "frozen" else "#95a5a6" for n in names]
    bars = ax2.barh(range(len(names)), vals, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([DISP.get(n, n) for n in names], fontsize=9)
    ax2.invert_yaxis()
    for i, (v, k) in enumerate(zip(vals, kens)):
        ax2.text(v + 0.002, i, f"{v:.3f}" + (f"  (τ={k:.2f})" if np.isfinite(k) else ""),
                 va="center", fontsize=8)
    ax2.set_xlabel("LOO diversion 列 RMSE (越低越好)")
    ax2.set_title("(b) 冷启动: 学习型对齐 vs 冻结嵌入")
    ax2.set_xlim(0, max(vals) * 1.35)
    ax2.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig9_coldstart.png"))
    plt.close()


def main():
    df = load_all()
    cols = ["spec", "disp", "best_vars", "aic", "d_aic", "rmse", "rmse_cut",
            "mae", "top1", "top3", "seconds"]
    summ = df[cols].copy()
    summ.to_json(os.path.join(RES, "summary.json"), orient="records",
                 indent=2, force_ascii=False)
    print(summ.to_string(index=False))

    fig1(df)
    fig2(df)
    fig3(df)
    fig4(df)
    fig5(df)
    fig6(df)
    fig7(df)
    fig8()
    fig9()
    print(f"[analysis] 9 张图 -> {FIG}")


if __name__ == "__main__":
    main()
