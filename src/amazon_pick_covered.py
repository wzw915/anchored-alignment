# -*- coding: utf-8 -*-
"""从 400 本共购候选书里挑出"元数据里有可读文本"的那批 → covered_asins.json。

背景：共购图在 400 本书上建，但 meta_Books.jsonl 只完整覆盖了其中一部分
（本地只有 5.05 GB 前缀，官方文件 14.70 GB）。只有能在 meta 里取到
书名/简介/特征的书，才能进入图文嵌入这一步，这批书就是论文里的 n = 175。
和 vg_build_coreview.py 一样，这个脚本原本也缺失，导致 175 这个数字在
外部无法复现，现在补上。

输入  data/amazon/copurchase_edges.json        （400 本候选）
      data/amazon/meta_books_head.jsonl        （2.0 GB 前缀）
      data/amazon/meta_books_full.jsonl        （5.05 GB 续传段）
输出  data/amazon/covered_asins_rebuilt.json
      cache/amazon_texts_rebuilt.json          （每本的拼接文本，供嵌入复用）

用法  python src/amazon_pick_covered.py
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
CACHE = os.path.join(PROJ, "cache")
OUT = os.path.join(AMZ, "covered_asins_rebuilt.json")
OUT_TXT = os.path.join(CACHE, "amazon_texts_rebuilt.json")

META_FILES = ["meta_books_head.jsonl", "meta_books_full.jsonl"]


def build_text(d):
    """与 amazon_dataset2.load_texts 完全一致的三段拼接。"""
    parts = [d.get("title") or ""]
    desc = d.get("description")
    if isinstance(desc, list):
        parts.append(" ".join(x for x in desc if isinstance(x, str)))
    elif isinstance(desc, str):
        parts.append(desc)
    feats = d.get("features")
    if isinstance(feats, list):
        parts.append(" ".join(x for x in feats if isinstance(x, str)))
    return " [SEP] ".join(p for p in parts if p.strip())[:4000]


def main():
    g = json.load(open(os.path.join(AMZ, "copurchase_edges.json")))
    want = set(g["books"])
    print(f"[pick] 候选 {len(want)} 本（共购图 top-400）", flush=True)

    texts = {}
    for fn in META_FILES:
        p = os.path.join(AMZ, fn)
        if not os.path.exists(p):
            print(f"[pick] 缺少 {fn}，跳过（先跑下载脚本）", flush=True)
            continue
        print(f"[pick] 扫描 {fn} ({os.path.getsize(p)/1e9:.2f} GB)", flush=True)
        n = 0
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                n += 1
                if not line or line[0] != "{":
                    continue
                i = line.find('"parent_asin"')
                if i < 0:
                    continue
                j = line.find('"', i + 14)
                k = line.find('"', j + 1)
                a = line[j + 1:k] if 0 < j < k else None
                if a not in want or a in texts:
                    continue
                try:
                    texts[a] = build_text(json.loads(line))
                except Exception:
                    continue
                if n % 500_000 == 0:
                    print(f"      {n/1e6:.1f}M 行, 命中 {len(texts)}", flush=True)
        print(f"      扫完 {n:,} 行，累计命中 {len(texts)}", flush=True)

    covered = sorted(texts)
    json.dump(covered, open(OUT, "w"), ensure_ascii=False)
    json.dump(texts, open(OUT_TXT, "w", encoding="utf-8"), ensure_ascii=False)

    print(f"\n[pick] 有文本的书 = {len(covered)}  [对照 results/amazon_dataset2.json "
          f"的 n_items]")
    ref = json.load(open(os.path.join(PROJ, "results", "amazon_dataset2.json"),
                        encoding="utf-8"))
    print(f"[对账] 重建 {len(covered)}  vs  论文 {ref['n_items']}  "
          f"{'PASS' if len(covered) == ref['n_items'] else 'MISMATCH'}")
    print(f"[pick] 写出 {OUT}\n[pick] 写出 {OUT_TXT}")
    if len(covered) != ref["n_items"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
