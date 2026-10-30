# -*- coding: utf-8 -*-
"""纯 Python 复刻 amazon_build.py，验证 copurchase_edges.json 可由原始 CSV 再生。

为什么不用 pandas 版直接跑：本机环境下 pandas 分块读取 1.7 GB CSV 会被
SIGTERM 杀掉（同样的现象见 audit_provenance 旧 A1 版，日志停在 25M 行）；
纯 csv 模块则稳定。本脚本逐条对齐 pandas 的语义：

  - chunksize=2_000_000 行；
  - chunk 内 groupby("user_id", sort=True)：组按 user_id 字典序排列，
    组内保持文件行序（pandas 的 groupby 保序性质）；
  - books = list(dict.fromkeys(...))：组内按首次出现顺序去重（dropna 后）；
  - MIN_BOOKS = 3 才计入；每用户枚举书对，最多 MAX_PAIRS = 15 对（先 i 外层
    后 j 内层，凑满即断，与原实现的嵌套 break 一致）；
  - 边权 = 截断后的共购用户数；度数取前 TOP_N=400 本；边过滤到 400 本内。

若输出与 data/amazon/copurchase_edges.json 逐字段一致，即证明该边表
确实由"原始评分文件 + 该脚本"生成，不存在另外的数据来源。
"""
import csv
import json
import os
import sys
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
CSV_PATH = os.path.join(AMZ, "Books_ratings.csv")
REF = os.path.join(AMZ, "copurchase_edges.json")

MIN_BOOKS = 3
MAX_PAIRS = 15
TOP_N = 400
CHUNK = 2_000_000


def read_chunks():
    """按 2M 行切块产出 [(user, [asin 原序, 已去空])]。"""
    rows = []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        iu, ia = header.index("user_id"), header.index("parent_asin")
        for row in rd:
            if len(row) <= max(iu, ia):
                continue
            u, a = row[iu], row[ia]
            if a:                      # 对应 dropna
                rows.append((u, a))
            if len(rows) == CHUNK:
                yield rows
                rows = []
    if rows:
        yield rows


def chunk_groups(rows):
    """复刻 groupby(user_id, sort=True)：按 user 排序，组内保行序。"""
    rows = sorted(rows, key=lambda r: r[0])
    cur, books = None, []
    for u, a in rows:
        if u != cur:
            if cur is not None:
                yield cur, books
            cur, books = u, []
        books.append(a)
    if cur is not None:
        yield cur, books


def main():
    edge_cnt = defaultdict(int)
    n_users = 0
    n_rows = 0
    for rows in read_chunks():
        n_rows += len(rows)
        for u, bs in chunk_groups(rows):
            books = list(dict.fromkeys(bs))
            if len(books) < MIN_BOOKS:
                continue
            n_users += 1
            cnt = 0
            for i in range(len(books)):
                for j in range(i + 1, len(books)):
                    a, b = sorted((books[i], books[j]))
                    edge_cnt[(a, b)] += 1
                    cnt += 1
                    if cnt >= MAX_PAIRS:
                        break
                if cnt >= MAX_PAIRS:
                    break
        print(f"      {n_rows/1e6:.0f}M 行, users={n_users:,}, "
              f"edges={len(edge_cnt):,}", flush=True)

    degree = defaultdict(int)
    for (a, b), c in edge_cnt.items():
        degree[a] += c
        degree[b] += c
    top = sorted(degree, key=degree.get, reverse=True)[:TOP_N]
    top_set = set(top)
    edges = [[a, b, c] for (a, b), c in edge_cnt.items()
             if a in top_set and b in top_set]

    new = dict(n_users_multi=int(n_users), n_books=len(top),
               n_edges=len(edges), edges=edges, books=top)

    ref = json.load(open(REF, encoding="utf-8"))
    same = dict(
        meta=all(ref[k] == new[k] for k in ("n_users_multi", "n_books", "n_edges")),
        books=ref["books"] == new["books"],
        edges=ref["edges"] == new["edges"],
    )
    print(f"\n[A1复刻] users>=3books: 重算 {n_users:,} vs 参考 {ref['n_users_multi']:,}"
          f"  {'PASS' if ref['n_users_multi'] == n_users else 'DIFF'}")
    print(f"[A1复刻] 400 本名单相同: {same['books']}")
    print(f"[A2复刻] 边表逐条相同: {same['edges']} "
          f"({len(edges):,} 条 vs {len(ref['edges']):,})")
    ok = all(same.values())
    print(f"\n结论：{'PASS — 边表完全可由原始 CSV 再生' if ok else 'DIFF — 见上'}")

    out_p = os.path.join(PROJ, "results", "amazon_build_replay.json")
    json.dump(dict(ok=ok, **same,
                   n_users_replay=int(n_users),
                   n_users_ref=int(ref["n_users_multi"]),
                   n_edges_replay=int(len(edges)),
                   n_edges_ref=int(len(ref["edges"]))),
              open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"写入 {out_p}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
