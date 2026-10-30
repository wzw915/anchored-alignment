# -*- coding: utf-8 -*-
"""合并模拟 (模仿 Compiani et al., Appendix E) —— "缺失实验"复刻

设计 (照抄 Appendix E): 固定焦点书 f, 与其余 9 本逐一模拟横向合并;
其余 8 本价格固定 $5, 边际成本为 0; 两场景:
  (a) 合并前: f 与 k 分别由独立出版商在 Bertrand-Nash 下定价 (交替最优)
  (b) 合并后: 同一出版商对 (f,k) 解联合一阶条件
报告: 两本书的平均涨价幅度; 政策规则: 涨价 >= 5% 即应被审查.
我们的评估升级: 用真实 diversion 矩阵作为"真值", 各方法矩阵的预测
涨价偏离真值多少、5% 门槛下的合并审查决策错几单.

与 Appendix E 的一处刻意偏差: 我们的需求系统含外部选项 (权重 0.3),
以避免 Compiani 指出的"全体等比例涨价导致多重均衡"问题.

输入: results/div_pred_{frozen,fullft,anchored}.npy + cache/diversion_empirical.npy
输出: results/merger_sim.json + figures/figA1_merger.png
"""
import os, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "cache")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
P_FIX = 5.0            # 其余书固定价 (Appendix E: $5)
BETA = 0.5             # 价格敏感度 (每 $1)
OUTSIDE = 0.3          # 外部选项权重 (设计偏差, 见 docstring)
SIGMA0 = 0.6
GRID = np.linspace(0.6, 3.0, 121)   # 相对 $5 的价格网格 (零成本下最优价可能很高)
THRESH = 5.0           # 审查门槛 (%)


def demand(q0, D, p, p0, beta):
    dp = p - p0
    e = np.exp(-beta * dp)
    q = np.zeros(len(q0))
    for i in range(len(q0)):
        w = np.append(D[i] * e, OUTSIDE)
        w[i] = (SIGMA0 / (1 - SIGMA0)) * e[i]
        q += q0[i] * w[:len(q0)] / w.sum()
    return q


def best_response(q0, D, p, j, p0, beta, c=0.0):
    """产品 j 的最优价 (其余价格固定)."""
    best_p, best_pi = p[j], None
    for g in GRID:
        pj = p0[j] * g
        pp = p.copy(); pp[j] = pj
        q = demand(q0, D, pp, p0, beta)
        pi = (pp[j] - c) * q[j]
        if best_pi is None or pi > best_pi:
            best_pi, best_p = pi, pj
    return best_p


def nash_pair(q0, D, a, b, p0, beta, rounds=40):
    """(a) 独立出版商 Bertrand-Nash: a/b 交替最优反应."""
    p = p0.copy()
    for _ in range(rounds):
        pa = best_response(q0, D, p, a, p0, beta)
        if abs(pa - p[a]) < 1e-6:
            break
        p[a] = pa
        pb = best_response(q0, D, p, b, p0, beta)
        if abs(pb - p[b]) < 1e-6:
            break
        p[b] = pb
    return p


def joint_pair(q0, D, a, b, p_start, p0, beta, sweeps=15, c=0.0):
    """(b) 合并后联合定价: 最大化 (a+b) 合计收入, 对 (a,b) 坐标上升."""
    p = p_start.copy()

    def total(pi_idx, pj_val):
        pp = p.copy(); pp[pi_idx] = pj_val
        q = demand(q0, D, pp, p0, beta)
        return float(((pp[[a, b]] - c) * q[[a, b]]).sum())

    for _ in range(sweeps):
        changed = False
        for j in (a, b):
            best_p, best_pi = p[j], None
            for g in GRID:
                pj = p0[j] * g
                pi_v = total(j, pj)
                if best_pi is None or pi_v > best_pi:
                    best_pi, best_p = pi_v, pj
            if abs(best_p - p[j]) > 1e-6:
                p[j] = best_p; changed = True
        if not changed:
            break
    return p


def all_pairs(out_path):
    """全部 C(J,2) 书对扫描：给出含真阴性的完整混淆矩阵。

    原实验只合并"焦点书 × 其余 9 本"，而真实涨价全部超过 5% 门槛——
    一个恒定预测"涨价 20%"的傻瓜模型也能 0/9 全对，实验只测了假阴性。
    这里放开到全部书对：低替代（diversion 小）的对真实涨价低于门槛，
    构成真阴性，决策错误才可解释。输出 results/merger_sim_allpairs.json。
    """
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    titles = list(books["title"])
    J = len(asins)
    S_true = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    S_true = S_true / S_true.sum(1, keepdims=True)
    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    fc = first[first.choice == 1]["asin"].value_counts(normalize=True)
    q0 = np.array([fc.get(a, 0.0) for a in asins]); q0 = q0 / q0.sum()

    mats = {"truth": S_true}
    for m in ["frozen", "fullft", "anchored"]:
        f = os.path.join(RES, f"div_pred_{m}.npy")
        if os.path.exists(f):
            mats[m] = np.load(f)

    p0 = np.full(J, P_FIX)
    pairs = [(a, b) for a in range(J) for b in range(a + 1, J)]
    print(f"[all-pairs] {len(pairs)} 个书对 × {len(mats)} 个矩阵", flush=True)

    rows = []
    for n, (a, b) in enumerate(pairs):
        rec = {"pair": f"{a}-{b}",
               "name": f"{titles[a][:14]} + {titles[b][:14]}"}
        for name, D in mats.items():
            pn = nash_pair(q0, D, a, b, p0, BETA)
            pj = joint_pair(q0, D, a, b, pn, p0, BETA)
            inc = 100 * ((pj[a] + pj[b]) / 2 - (pn[a] + pn[b]) / 2) \
                / ((pn[a] + pn[b]) / 2)
            rec[name] = round(float(inc), 2)
        rows.append(rec)
        if (n + 1) % 10 == 0:
            print(f"      {n+1}/{len(pairs)} 对", flush=True)

    truth = np.array([r["truth"] for r in rows])
    out = dict(mode="all_pairs", n_pairs=len(rows), p_fix=P_FIX, beta=BETA,
               outside=OUTSIDE, thresh=THRESH, pairs=rows, methods={})
    for m in ["frozen", "fullft", "anchored"]:
        pred = np.array([r[m] for r in rows])
        pos_t, pos_p = truth >= THRESH, pred >= THRESH
        tp = int(np.sum(pos_t & pos_p)); fp = int(np.sum(~pos_t & pos_p))
        fn = int(np.sum(pos_t & ~pos_p)); tn = int(np.sum(~pos_t & ~pos_p))
        out["methods"][m] = dict(
            mae_pp=round(float(np.mean(np.abs(pred - truth))), 2),
            TP=tp, FP=fp, FN=fn, TN=tn,
            n_pos_true=int(pos_t.sum()), n_pos_pred=int(pos_p.sum()),
            accuracy=round((tp + tn) / len(rows), 3))
    json.dump(out, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"写入 {out_path}")
    print(json.dumps(out["methods"], indent=1))
    return out


def main():
    if "--all-pairs" in sys.argv:
        p = os.path.join(RES, "merger_sim_allpairs.json")
        all_pairs(p)
        return 0
    books = pd.read_csv(os.path.join(CACHE, "books_clean.csv"))
    asins = list(books["asin"])
    titles = list(books["title"])
    J = len(asins)
    S_true = np.load(os.path.join(CACHE, "diversion_empirical.npy"))
    S_true = S_true / S_true.sum(1, keepdims=True)

    first = pd.read_csv(os.path.join(CACHE, "first_choice.csv"))
    fc = first[first.choice == 1]["asin"].value_counts(normalize=True)
    q0 = np.array([fc.get(a, 0.0) for a in asins]); q0 = q0 / q0.sum()

    mats = {"真实 diversion (真值)": S_true}
    for m in ["frozen", "fullft", "anchored"]:
        f = os.path.join(RES, f"div_pred_{m}.npy")
        if os.path.exists(f):
            mats[m] = np.load(f)
    disp = {"frozen": "冻结嵌入", "fullft": "完全微调", "anchored": "锚定对齐",
            "真实 diversion (真值)": "真实 (真值)"}

    # 焦点书: 第一选份额最高的书 (对应 Appendix E 的 Dopamine Detox)
    focal = int(np.argmax(q0))
    p0 = np.full(J, P_FIX)

    rows = []
    for k in range(J):
        if k == focal:
            continue
        rec = {"pair": f"{titles[focal][:14]} + {titles[k][:14]}", "k": k}
        for name, D in mats.items():
            pn = nash_pair(q0, D, focal, k, p0, BETA)       # 合并前 Nash
            pj = joint_pair(q0, D, focal, k, pn, p0, BETA)  # 合并后联合
            inc = 100 * ((pj[focal] + pj[k]) / 2 - (pn[focal] + pn[k]) / 2) \
                / ((pn[focal] + pn[k]) / 2)
            rec[name] = round(float(inc), 2)
        rows.append(rec)

    truth = np.array([r["真实 diversion (真值)"] for r in rows])
    out = dict(focal=titles[focal], p_fix=P_FIX, beta=BETA, outside=OUTSIDE,
               thresh=THRESH, pairs=rows, methods={})
    for m in ["frozen", "fullft", "anchored"]:
        pred = np.array([r[m] for r in rows])
        dec_t = truth >= THRESH
        dec_p = pred >= THRESH
        out["methods"][m] = dict(
            mae_pp=round(float(np.mean(np.abs(pred - truth))), 2),
            decision_errors=int(np.sum(dec_t != dec_p)),
            n_mergers=len(rows),
            errors_detail=[{"pair": rows[i]["pair"],
                            "true": float(truth[i]), "pred": float(pred[i])}
                           for i in range(len(rows)) if dec_t[i] != dec_p[i]])

    with open(os.path.join(RES, "merger_sim.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 图: 每个合并对的预测涨价 (分组条形) + 真值散点 + 5% 门槛线
    labels = [r["pair"] for r in rows]
    x = np.arange(len(rows))
    w = 0.2
    colors = {"frozen": "#2563eb", "fullft": "#9ca3af", "anchored": "#dc2626"}
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for i, m in enumerate(["frozen", "fullft", "anchored"]):
        pred = np.array([r[m] for r in rows])
        ax.bar(x + (i - 1) * w, pred, w, label=disp[m], color=colors[m], zorder=3)
    ax.scatter(x, truth, marker="D", color="black", s=42, zorder=5,
               label="真实 diversion (真值)")
    ax.axhline(THRESH, color="#ef4444", ls="--", lw=1.2, zorder=2)
    ax.annotate(f"审查门槛 {THRESH:.0f}%", (len(rows) - 0.5, THRESH + 0.6),
                fontsize=9, color="#ef4444", ha="right")
    ax.set_xticks(x, labels, fontsize=7.2, rotation=28, ha="right")
    ax.set_ylabel("两本合并书的预测平均涨价 (%)", fontsize=10)
    ax.set_title(f"合并模拟（模仿 Compiani et al., Appendix E）：焦点书《{titles[focal]}》\n"
                 "其余书固定 $5、零边际成本；黑色菱形 = 真实 diversion 矩阵的预测", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.savefig(os.path.join(FIG, "figA1_merger.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "figA1_merger.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(out["methods"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
