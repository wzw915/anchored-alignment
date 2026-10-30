# -*- coding: utf-8 -*-
"""构建 Amazon Books 共购替代图 (用户级共购作为替代真值代理)

输入: data/amazon/Books_ratings.csv (user_id,parent_asin,rating,timestamp)
输出: data/amazon/copurchase_edges.json (加权边表), book_stats.json
策略: 流式读取; 只保留购书 >=3 本的用户; 每用户最多取前 15 对组合(防超级用户主导);
      边权 = 共购用户数; 取度数最高的 ~400 本书。
"""
import os, json, sys
import pandas as pd
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
MIN_BOOKS = 3
MAX_PAIRS = 15
TOP_N = 400

CHUNK = 2_000_000

edge_cnt = defaultdict(int)
book_users = defaultdict(set)
n_users = 0
for chunk in pd.read_csv(os.path.join(AMZ, "Books_ratings.csv"),
                         usecols=["user_id", "parent_asin"], chunksize=CHUNK):
    for u, g in chunk.groupby("user_id")["parent_asin"]:
        books = list(dict.fromkeys(g.dropna().astype(str)))
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

degree = defaultdict(int)
for (a, b), c in edge_cnt.items():
    degree[a] += c
    degree[b] += c
top = sorted(degree, key=degree.get, reverse=True)[:TOP_N]
top_set = set(top)
edges = [[a, b, c] for (a, b), c in edge_cnt.items() if a in top_set and b in top_set]

with open(os.path.join(AMZ, "copurchase_edges.json"), "w") as f:
    json.dump(dict(n_users_multi=int(n_users), n_books=len(top),
                   n_edges=len(edges),
                   edges=edges, books=top), f)
print(f"users>=3books: {n_users}, books: {len(top)}, edges: {len(edges)}")
