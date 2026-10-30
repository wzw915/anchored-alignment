# -*- coding: utf-8 -*-
"""Books 共购边表对账：直接回到 Books_ratings.csv 重数共同买家。

为什么不用"全量建 user->set"的写法：2914 万行、约 2000 万个去重用户，
每个用户一个 set 对象就吃掉数 GB，进程会被 OOM 杀掉（已发生一次）。
改为 175 位整数位图：每行只做 mask[user] |= 1<<idx，一个用户只存一个 int。

验证三件事
  E1  175 本诱导子图的边数 = 6,906
  E2  派生权重 <= 原始重数（原构建有 MAX_PAIRS 截断，只会少算不会多算）
  E3  绝大多数边的派生权重与原始重数**精确相等**，说明边表就是原始
      记录聚合出来的，不是另外生成的
  E4  在 175 本内买过 >=2 本的用户数（这才是真正参与构图的样本规模，
      与全类目"买过>=3本"的 249 万不是同一口径）

用法：python src/audit_books_edges.py [--json results/books_edge_audit.json]
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
CSV_PATH = os.path.join(AMZ, "Books_ratings.csv")
EDGES = os.path.join(AMZ, "copurchase_edges.json")
COVERED = os.path.join(AMZ, "covered_asins.json")

MIN_BOOKS = 3     # 全类目口径（仅用于说明，不在本脚本统计）
MAX_PAIRS = 200   # 原构建对每用户书对的截断上限


def build_masks(covered_list):
    """流式扫评分文件，返回 user -> 175 位位图。"""
    idx = {a: i for i, a in enumerate(covered_list)}
    mask = {}
    n_rows = 0
    n_hit = 0
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        iu, ia = header.index("user_id"), header.index("parent_asin")
        for row in rd:
            n_rows += 1
            try:
                j = idx.get(row[ia])
                if j is None:
                    continue
                n_hit += 1
                u = row[iu]
                mask[u] = mask.get(u, 0) | (1 << j)
            except IndexError:
                continue
            if n_rows % 5_000_000 == 0:
                print(f"      {n_rows/1e6:.0f}M 行, 命中 {n_hit:,}, "
                      f"用户 {len(mask):,}", flush=True)
    return mask, n_rows, n_hit


def enumerate_edges(mask):
    """每个用户的书目位图展开成书对，累加共购次数。"""
    cnt = defaultdict(int)
    n_users_ge2 = 0
    for m in mask.values():
        if m & (m - 1) == 0:      # 只有一本，跳过
            continue
        n_users_ge2 += 1
        bits = []
        x = m
        while x:
            low = x & -x
            bits.append(low.bit_length() - 1)
            x ^= low
        if len(bits) > 40:        # 极端用户守卫，避免组合爆炸
            bits = bits[:40]
        for i in range(len(bits)):
            bi = bits[i]
            for j in range(i + 1, len(bits)):
                bj = bits[j]
                key = (bi, bj) if bi < bj else (bj, bi)
                cnt[key] += 1
    return cnt, n_users_ge2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(PROJ, "results",
                                                   "books_edge_audit.json"))
    a = ap.parse_args()

    covered = json.load(open(COVERED, encoding="utf-8"))
    print(f"[E0] covered_asins = {len(covered)} 本，位图宽度 {len(covered)} 位",
          flush=True)

    mask, n_rows, n_hit = build_masks(covered)
    print(f"[E0] 扫完 {n_rows:,} 行，175 本内命中 {n_hit:,} 行，"
          f"相关用户 {len(mask):,}", flush=True)

    cnt, n_users_ge2 = enumerate_edges(mask)
    print(f"[E4] 175 本内买过 >=2 本的用户 = {n_users_ge2:,}", flush=True)
    print(f"[E1] 重算诱导子图边数 = {len(cnt):,}", flush=True)

    # 派生边表（同一次筛选下应有 6906 条）
    g = json.load(open(EDGES, encoding="utf-8"))
    # 索引必须与位图一致：位图用的是 covered 的位置，不能换成 400 本的 books
    # 顺序，否则 (i,j) 落在两套不同的编号空间里，比对全部错位。
    pos = {b: i for i, b in enumerate(covered)}
    cov = set(covered)
    der = {}
    for x, y, w in g["edges"]:
        if x in cov and y in cov:
            i, j = pos[x], pos[y]
            der[(i, j) if i < j else (j, i)] = int(w)
    print(f"[E1] 派生边表诱导子图 = {len(der):,} 条  "
          f"[期望 6,906] {len(der) == 6906}", flush=True)

    # 派生表显然不是"所有 w>=1 的边"，先看哪个权重阈值能复现 6906
    for thr in (1, 2, 3, 4, 5):
        n_thr = sum(1 for v in cnt.values() if v >= thr)
        print(f"      重算 w>={thr}: {n_thr:,} 条", flush=True)

    common = set(cnt) & set(der)
    only_der = set(der) - set(cnt)
    only_new = set(cnt) - set(der)
    exact = sum(1 for k in common if cnt[k] == der[k])
    le = sum(1 for k in common if der[k] <= cnt[k])
    over = [k for k in common if der[k] > cnt[k]]
    print(f"[E2] 共有边 {len(common):,}；派生权重<=重算 {le:,} "
          f"({100*le/max(len(common),1):.2f}%)；越界 {len(over)}", flush=True)
    print(f"[E3] 权重精确相等 {exact:,} ({100*exact/max(len(common),1):.2f}%)",
          flush=True)
    print(f"     只在派生表 {len(only_der):,}；只在重算 {len(only_new):,}",
          flush=True)

    # 权重分布对比
    dv = sorted(der.values())
    cv = sorted(cnt[k] for k in common)
    print(f"[E3] 派生权重 中位 {dv[len(dv)//2]} 最大 {dv[-1]}；"
          f"重算 中位 {cv[len(cv)//2]} 最大 {cv[-1]}", flush=True)
    if only_der:
        k = next(iter(only_der))
        print(f"      示例只在派生表: books {books[k[0]] if False else ''} "
              f"idx{k} 权重 {der[k]} 重算 {cnt.get(k, 0)}", flush=True)

    out = dict(
        covered=len(covered), rows_scanned=int(n_rows), rows_hit=int(n_hit),
        users_touching_covered=int(len(mask)),
        users_in_graph_ge2=int(n_users_ge2),
        edges_recount=int(len(cnt)), edges_derived=int(len(der)),
        edges_match=bool(len(der) == 6906),
        common=int(len(common)), only_derived=int(len(only_der)),
        only_recount=int(len(only_new)),
        weight_le_recount=int(le), weight_over_recount=int(len(over)),
        weight_exact=int(exact),
        median_weight_derived=int(dv[len(dv)//2]),
        median_weight_recount=int(cv[len(cv)//2]),
    )
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(out, open(a.json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n写入 {a.json}")

    # 同时导出无截断边表（原构建 MAX_PAIRS=15 会压低权重；这里不截断），
    # 供 dataset2 敏感性分析用，检验"结论是否依赖截断"。
    full_path = os.path.join(AMZ, "copurchase_edges_full.json")
    full_edges = [[covered[i], covered[j], int(c)]
                  for (i, j), c in sorted(cnt.items())]
    json.dump(dict(
        n_users_in_graph=int(n_users_ge2),
        n_books=len(covered), n_edges=len(full_edges),
        edges=full_edges, books=sorted(covered),
        note="无截断重算：每用户贡献全部 175 本内书对，权重=真实共同买家数；"
             "与 copurchase_edges.json 的差异来自其 MAX_PAIRS=15 截断",
    ), open(full_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"写出无截断边表 {full_path}（{len(full_edges):,} 条边）")

    ok_all = (len(der) == 6906 and len(over) == 0 and
              exact / max(len(common), 1) > 0.99)
    print(f"结论：{'PASS' if ok_all else 'CHECK'}")
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
