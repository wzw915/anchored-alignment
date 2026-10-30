# -*- coding: utf-8 -*-
"""从 HF 原始 Books.jsonl（reviews）生成 Books_ratings.csv。

背景：项目早期先把 20 GB 的 Books.jsonl 转成 4 列 CSV 再算共购图，但转换
脚本没有留档，外部复现者因此无法从 HF 原始文件推到 data/amazon/Books_ratings.csv。
本脚本补上这一环，并支持断点与抽样，便于分次跑完。

产出列：user_id, parent_asin, rating, timestamp  （与 HF 原始字段同名）
规模参考：完整转换约 2,914 万行 / 1.69 GB（时间跨度 1996-06 至 2023-09）。

用法
  python src/amazon_books_reviews.py                 # 全量，边下边转，不落 20GB 中间文件
  python src/amazon_books_reviews.py --limit 200000  # 只取前 20 万行做冒烟测试
  python src/amazon_books_reviews.py --range 0 5e9   # 只处理前 5 GB 字节区间
"""
import argparse
import csv
import json
import os
import sys
import urllib.request

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
OUT = os.path.join(AMZ, "Books_ratings.csv")
MIRROR = "https://hf-mirror.com"
URL = (MIRROR + "/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
       "raw/review_categories/Books.jsonl")

FIELDS = ["user_id", "parent_asin", "rating", "timestamp"]


def open_stream(start=None, end=None):
    """打开 HF 文件流，可指定字节区间（HF 支持 Range）。"""
    req = urllib.request.Request(URL)
    if start is not None:
        req.add_header("Range", f"bytes={int(start)}-"
                                + (f"{int(end)}" if end else ""))
    return urllib.request.urlopen(req, timeout=120)


def remote_size():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0) or
                   r.headers.get("X-Linked-Size", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="最多写多少行，0=不限")
    ap.add_argument("--range", nargs=2, type=float, default=None,
                    help="字节区间 start end，如 0 5e9")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖已存在的输出文件（默认拒绝，避免误毁正式数据）")
    a = ap.parse_args()

    size = remote_size()
    print(f"[books] 远端 Books.jsonl = {size/1e9:.2f} GB")

    start = end = None
    if a.range:
        start, end = int(a.range[0]), int(a.range[1])
        print(f"[books] 处理字节区间 {start:,} - {end:,}")

    if os.path.exists(a.out):
        if not a.force:
            sys.exit(f"[books] {a.out} 已存在。要用抽样模式请加 --out 指定新路径；"
                     f"确实要覆盖请加 --force。")
        backup = a.out + ".bak"
        print(f"[books] 备份 {os.path.basename(a.out)} -> {os.path.basename(backup)}")
        os.replace(a.out, backup)

    n = 0
    skipped = 0
    resp = open_stream(start, end)
    fo = open(a.out, "w", encoding="utf-8", newline="")
    w = csv.writer(fo)
    w.writerow(FIELDS)
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace")
            if not line.startswith("{"):
                skipped += 1          # 分片边界上的半行
                continue
            try:
                d = json.loads(line)
            except Exception:
                skipped += 1
                continue
            u = d.get("user_id")
            pa = d.get("parent_asin")
            if not u or not pa:
                skipped += 1
                continue
            w.writerow([u, pa, d.get("rating", ""), d.get("timestamp", "")])
            n += 1
            if n % 5_000_000 == 0:
                fo.flush()
                print(f"      {n/1e6:.0f}M 行", flush=True)
            if a.limit and n >= a.limit:
                print(f"[books] 达到 --limit {a.limit}")
                break
    finally:
        fo.close()

    gb = os.path.getsize(a.out) / 1e9
    print(f"[books] 写出 {a.out}: {n:,} 行 / {gb:.2f} GB（跳过坏行 {skipped}）")


if __name__ == "__main__":
    main()
