# -*- coding: utf-8 -*-
"""合成对照实验：个体级对齐能否恢复一个"已知的"需求子空间。

用途：论文用它排除"实现错误"这一解释。逻辑是——如果数据生成过程完全满足
个体级对齐的隐含假设（语义信息里就含有决定替代模式的那几个维度，且可由
线性映射取出），那么同一套 InfoNCE 算法就应当把这几维恢复出来。若在真实
嵌入上它反而变差，问题就出在真实数据的几何结构，而不是代码写错。

生成过程
  b_j ~ N(0, I_{d_b})                        每个产品的需求潜变量（真值）
  s_jk = softmax_k(b_j · b_k / tau_b)        替代模式真值（行归一）
  a_j  = b_j                                 语义 = 需求（假设成立的最强情形）
  z_j  = [a_j, eps_j] ∈ R^{d_z}              嵌入：前 d_b 维是信号，其余是噪声
  消费者 i：先按 s_j 抽第一选 j，再按 s_j 抽第二选 k（不含 j 自身）

对比三种表示
  frozen   原始 z_j（信号淹没在噪声里）
  zca      零参数白化
  aligned  个体级 InfoNCE 学出的线性投影 W（与主实验同构：正样本=第一选→第二选，
           负样本=该消费者未选的产品，温度 tau_nce）

评估
  R²        用线性探针从表示预测 b（5 折交叉验证），衡量需求子空间是否被恢复
  RMSE      diversion 列预测误差（留一产品）
  tau       Kendall 秩相关

输出 results/synth_control.json

用法：python src/synth_control.py [--reps 5]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")

J = 40            # 产品数
D_B = 2           # 需求维度（真值只由这两维决定）
D_Z = 384         # 嵌入维度（与 all-MiniLM-L6-v2 一致）
TAU_B = 0.3       # 替代模式温度
N_CONS = 5000     # 消费者数
N_NEG = 3         # 每样本负例数
TAU_NCE = 0.1     # InfoNCE 温度
EPOCHS = 400
LR = 1e-2
SEED = 2026
PROBE_K = 8       # 主报告用的探针维数
# 敏感性扫描只在有效范围内取值：5 折 CV 的训练集为 0.8*J=32 行，
# 探针维数超过 n_train/4 后 OLS 重新落入欠定区，扫描就失去意义。
PROBE_GRID = (2, 4, 6, 8)


def gen_data(rng):
    """返回 b（需求真值）、S（替代模式真值）、z（嵌入）、样本三元组。"""
    b = rng.normal(size=(J, D_B))
    logits = b @ b.T / TAU_B
    np.fill_diagonal(logits, -1e9)
    S = np.exp(logits - logits.max(1, keepdims=True))
    S /= S.sum(1, keepdims=True)

    # 语义 = 需求：信号放在前 D_B 维，其余是独立噪声
    z = rng.normal(size=(J, D_Z))
    z[:, :D_B] = b

    # 消费者：第一选按行分布抽 j，第二选按 s_j 抽 k（排除 j）
    j = rng.choice(J, size=N_CONS, p=np.full(J, 1.0 / J))
    k = np.array([rng.choice(J, p=S[jj] / S[jj].sum()) for jj in j])
    return b, S, z, j, k


def train_infonce(z, j, k, seed=SEED):
    """线性投影 + InfoNCE，正样本=(j,k)，负样本=其余产品中随机 3 个。"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Et = torch.tensor(z, dtype=torch.float32)
    W = torch.nn.Parameter(torch.eye(D_Z))
    opt = torch.optim.Adam([W], lr=LR)

    jt = torch.tensor(j, dtype=torch.long)
    kt = torch.tensor(k, dtype=torch.long)
    # 负例：该消费者没选的那些产品里随机取 3 个
    alli = np.arange(J)
    neg = np.empty((len(j), N_NEG), dtype=np.int64)
    for i, (jj, kk) in enumerate(zip(j, k)):
        cand = alli[(alli != jj) & (alli != kk)]
        neg[i] = rng.choice(cand, size=N_NEG, replace=False)
    nt = torch.tensor(neg, dtype=torch.long)

    for _ in range(EPOCHS):
        Zp = F.normalize(Et @ W, dim=1)
        Sm = Zp @ Zp.T
        pos = Sm[jt, kt].unsqueeze(1)
        ngs = Sm[jt.unsqueeze(1).expand(-1, N_NEG), nt]
        loss = -F.log_softmax(torch.cat([pos, ngs], 1) / TAU_NCE, 1)[:, 0].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (Et @ W).numpy().copy()


def zca(z, eps=1e-3):
    C = np.cov(z.T) + eps * np.eye(z.shape[1])
    ev, evec = np.linalg.eigh(C)
    W = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
    return z @ W


def subspace_r2(Z, b, k=8):
    """线性探针 R²：先在标准化后的表示上做 PCA 降到 k 维，再逐个维度回归。

    为什么必须先降维：J=40 个产品、d_z=384 维，无正则 OLS 在 p >> n 时
    用最小范数解会把系数摊到全部噪声维上，即使真值就在前两维也会给出
    负 R²（实测 frozen 仅 +0.05/-0.12）。PCA 探针把有效的比较限制在
    表示的主方向上，frozen 的信号被 382 个噪声维淹没、aligned 的信号
    被放大，二者的差异才可读。
    """
    Zs = StandardScaler().fit_transform(Z)
    kk = int(min(k, Zs.shape[1], Zs.shape[0] // 5))
    P = PCA(n_components=kk, random_state=0).fit_transform(Zs)
    scores = [cross_val_score(LinearRegression(), P, b[:, d],
                              cv=5, scoring="r2").mean()
              for d in range(b.shape[1])]
    return float(np.mean(scores))


def diversion_metrics(Z, S):
    """留一产品：用余弦相似度预测 diversion 列。"""
    Zn = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-10, None)
    C = Zn @ Zn.T
    rmse, taus = [], []
    for j0 in range(J):
        col = C[:, j0].copy()
        pred = np.exp(col / TAU_B)
        pred[j0] = 0
        pred = pred / pred.sum()
        keep = [i for i in range(J) if i != j0]
        rmse.append(np.sqrt(np.mean((pred[keep] - S[keep, j0]) ** 2)))
        taus.append(kendalltau(pred[keep], S[keep, j0]).statistic)
    return float(np.mean(rmse)), float(np.mean(taus))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    runs = []
    for rep in range(a.reps):
        rng = np.random.default_rng(SEED + rep)
        b, S, z, j, k = gen_data(rng)
        Zal = train_infonce(z, j, k, seed=SEED + rep)
        Zzca = zca(z)

        row = dict(rep=rep)
        for name, Z in (("frozen", z), ("zca", Zzca), ("aligned", Zal)):
            r2 = subspace_r2(Z, b, k=PROBE_K)
            rm, tu = diversion_metrics(Z, S)
            # k 敏感性 + 主成分方差占比，用来说明探针维数不改变排序结论
            r2_grid = {k: subspace_r2(Z, b, k=k) for k in PROBE_GRID}
            evr = PCA(n_components=3, random_state=0).fit(
                StandardScaler().fit_transform(Z)).explained_variance_ratio_
            row[name] = dict(r2=r2, r2_grid=r2_grid,
                             evr3=[float(x) for x in evr], rmse=rm, tau=tu)
            print(f"  rep{rep} {name:8s} R2={r2:+.4f}  RMSE={rm:.4f}  tau={tu:+.4f}"
                  f"  evr3={np.round(evr, 3)}", flush=True)
        runs.append(row)

    agg = {}
    for name in ("frozen", "zca", "aligned"):
        agg[name] = dict(
            r2=float(np.mean([r[name]["r2"] for r in runs])),
            r2_sd=float(np.std([r[name]["r2"] for r in runs])),
            rmse=float(np.mean([r[name]["rmse"] for r in runs])),
            tau=float(np.mean([r[name]["tau"] for r in runs])),
            r2_grid={k: float(np.mean([r[name]["r2_grid"][k] for r in runs]))
                     for k in PROBE_GRID},
        )

    out = dict(setting=dict(J=J, d_b=D_B, d_z=D_Z, tau_b=TAU_B, n_consumers=N_CONS,
                            n_neg=N_NEG, tau_nce=TAU_NCE, epochs=EPOCHS,
                            reps=a.reps, seed=SEED,
                            probe="PCA(%d)+5-fold linear" % PROBE_K,
                            probe_grid=list(PROBE_GRID),
                            note="语义=需求的合成设定：数据生成过程满足个体级对齐的假设"),
               aggregate=agg, runs=runs)
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "synth_control.json"), "w"),
              ensure_ascii=False, indent=1)

    print("\n=== 汇总（%d 次重复）===" % a.reps)
    for name in ("frozen", "zca", "aligned"):
        v = agg[name]
        print(f"{name:8s} R2={v['r2']:+.4f}±{v['r2_sd']:.4f}  "
              f"RMSE={v['rmse']:.4f}  tau={v['tau']:+.4f}")
    print("\n读出：合成设定下 aligned 的 R2 应显著高于 frozen；")
    print("若在真实嵌入上 aligned 反而更差，则失败源于数据几何而非实现错误。")
    print("写入 results/synth_control.json")


if __name__ == "__main__":
    main()
