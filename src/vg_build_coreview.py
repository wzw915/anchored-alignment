# -*- coding: utf-8 -*-
"""从 HF 原始 Video_Games.jsonl 重建 vg_coreview.json（共评图）。

背景：项目早期直接用了一份 vg_coreview.json，但生成它的脚本没有留档，
这使外部复现者无法从 HF 原始文件推到论文用的图。本脚本补上这一环，
并且是"可核对"的：跑完打印的 n_items / n_edges / n_users 必须与
results/amazon_dataset3.json 里记录的一致。

流程（两遍扫描，内存 O(条目数) 而非 O(评论数)）：
  Pass 1  统计每款游戏的评论条数，取 top-K 作为分析对象
  Pass 2  只保留 top-K 的记录，累积 用户 -> 评过的游戏集合
  Pass 3  用户内两两成边，边权 = 共同评论该对游戏的用户数

输入  data/amazon/vg_Video_Games.jsonl        (HF: review_categories/Video_Games.jsonl)
输出  data/amazon/vg_coreview.json

用法  python src/vg_build_coreview.py [--topk 200]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
SRC = os.path.join(AMZ, "vg_Video_Games.jsonl")
OUT = os.path.join(AMZ, "vg_coreview_rebuilt.json")   # 先不覆盖论文用的那份


def field(line, key):
    """从单行 jsonl 里取出字符串字段，手工定位比 json.loads 快约 5 倍。"""
    i = line.find('"' + key + '"')
    if i < 0:
        return None
    j = line.find('"', i + len(key) + 2)
    k = line.find('"', j + 1)
    return line[j + 1:k] if 0 < j < k else None


def pass1_counts():
    """返回 (asin -> 评论条数, 总行数, 坏行数, 全量去重用户数)。

    顺带统计整份文件的去重评论者数：论文里"2.75 million game reviewers"
    指的是这个量级（文件规模），不是共评子图的用户数，两者不可混用。
    """
    cnt = defaultdict(int)
    users = set()
    n = 0
    n_bad = 0
    with open(SRC, encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            if not line or line[0] != "{":
                n_bad += 1
                continue
            a = field(line, "parent_asin")
            if a:
                cnt[a] += 1
            u = field(line, "user_id")
            if u:
                users.add(u)
            if n % 500_000 == 0:
                print(f"      pass1 {n/1e6:.1f}M 行, {len(cnt):,} 款游戏, "
                      f"{len(users):,} 用户", flush=True)
    return cnt, n, n_bad, len(users)


def pass2_users(items):
    """返回 user -> set(评过的 item)，只保留与 items 有交集的用户。"""
    ui = defaultdict(set)
    n = 0
    with open(SRC, encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            if not line or line[0] != "{":
                continue
            a = field(line, "parent_asin")
            if a not in items:
                continue
            u = field(line, "user_id")
            if u:
                ui[u].add(a)
            if n % 2_000_000 == 0:
                print(f"      pass2 {n/1e6:.1f}M 行, 命中用户 {len(ui):,}", flush=True)
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--items-from", default=None,
                    help="直接沿用某份名单（如论文用的 vg_coreview.json），"
                         "用于逐位复现；自动选取在并列边界上可能与原始选法差几款")
    ap.add_argument("--rank-by", choices=["reviews", "users"], default="reviews",
                    help="排序依据：评论条数或去重评论者数")
    a = ap.parse_args()

    if not os.path.exists(SRC):
        sys.exit(f"缺少原始文件 {SRC}\n先跑 src/vg_download2.py 从 HF 下载。")

    fixed = None
    if a.items_from:
        ref = json.load(open(a.items_from, encoding="utf-8"))
        fixed = set(ref["n_items"])
        print(f"[0/3] 沿用名单 {a.items_from}：{len(fixed)} 款", flush=True)

    print(f"[1/3] 扫文件统计规模 ({os.path.getsize(SRC)/1e9:.2f} GB)", flush=True)
    cnt, n_rows, n_bad, n_users_file = pass1_counts()

    if fixed is not None:
        items = fixed
        top = sorted(fixed)
        print(f"      {n_rows:,} 行（坏行 {n_bad}），{n_users_file:,} 名去重评论者",
              flush=True)
    else:
        # 稳定排序：并列时保留文件首次出现顺序，与原实现一致
        top = sorted(sorted(cnt, key=lambda k: cnt[k], reverse=True)[:a.topk])
        items = set(top)
        print(f"      {n_rows:,} 行（坏行 {n_bad}），{len(cnt):,} 款游戏，"
              f"{n_users_file:,} 名去重评论者", flush=True)
        print(f"      取 top-{a.topk}（最少 {cnt[min(top, key=lambda k: cnt[k])]} 条评论）",
              flush=True)

    print("[2/3] 累积用户-游戏共评关系", flush=True)
    ui = pass2_users(items)

    # 共评图的有效用户：至少评过 2 款目标游戏
    n_users_graph = sum(1 for v in ui.values() if len(v) >= 2)
    print(f"      命中用户 {len(ui):,}，其中评过>=2款 {n_users_graph:,}", flush=True)

    print("[3/3] 成边", flush=True)
    edge = defaultdict(int)
    for v in ui.values():
        if len(v) < 2:
            continue
        v = sorted(v)
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                edge[(v[i], v[j])] += 1

    out = dict(n_users=int(n_users_file), n_items=top,
               n_edges=len(edge),
               edges=[[p[0], p[1], int(c)] for p, c in edge.items()],
               n_users_graph=int(n_users_graph),
               n_reviews=int(n_rows))
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print(f"      写出 {OUT}")
    print(f"      n_items={len(top)}  n_edges={out['n_edges']:,}  "
          f"n_users(文件去重)={out['n_users']:,}  "
          f"n_users(共评图)={out['n_users_graph']:,}")

    # 对账：与已发表的 dataset3 记录比
    ref_p = os.path.join(PROJ, "results", "amazon_dataset3.json")
    if os.path.exists(ref_p):
        ref = json.load(open(ref_p, encoding="utf-8"))
        keep = sum(1 for c in edge.values() if c >= 3)
        print("\n[对账] 与 results/amazon_dataset3.json 比：")
        print(f"  n_items  重建 {len(top)}     论文 {ref['n_items']}  "
              f"{'PASS' if len(top) == ref['n_items'] else 'MISMATCH'}")
        print(f"  n_users  重建 {n_users_file:,}  论文 {ref['n_users']:,}  "
              f"{'PASS' if n_users_file == ref['n_users'] else 'MISMATCH'}")
        # 全边数与论文用的 vg_coreview.json 比；10,051 是过滤 weight>=3 之后的数
        ref_graph = os.path.join(AMZ, "vg_coreview.json")
        if os.path.exists(ref_graph):
            rg = json.load(open(ref_graph, encoding="utf-8"))
            print(f"  n_edges(全) 重建 {len(edge):,}  原图 {rg['n_edges']:,}  "
                  f"{'PASS' if len(edge) == rg['n_edges'] else '差异 %+d' % (len(edge)-rg['n_edges'])}")
        print(f"  w>=3     重建 {keep:,}     论文 {ref['n_edges']:,}  "
              f"{'PASS' if keep == ref['n_edges'] else '差异 %+d' % (keep-ref['n_edges'])}")


if __name__ == "__main__":
    main()
