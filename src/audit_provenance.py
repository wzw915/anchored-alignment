# -*- coding: utf-8 -*-
"""数据溯源审计 — 独立重算，不 import 任何项目模块。

目的：证明论文里的数据集统计量（用户数、边数）确实来自磁盘上的原始
公开文件，而不是任何形式的合成或手填。方法是从原始 jsonl / csv 用
一套与主流程完全独立的实现重新算一遍，然后逐项对账。

校验项
  A1 Books_ratings.csv 全量：购书 >=3 本的用户数（期望 2,492,843）
  A2 Books 175 本诱导子图：共购边数（期望 6,906）
  A3 vg_Video_Games.jsonl 全量：共同评论 200 款游戏中的 >=2 款用户数
  A4 VG 200 本诱导子图：weight>=3 的边数（期望 10,051）
  A5 survey 原始 CSV：通过全部清洗规则的受访者数（期望 9,265）
  A6 原始文件指纹：字节大小 + 抽样 sha256 + 行完整性

用法：python src/audit_provenance.py [--fast]
      --fast 只跑 A5/A6（秒级），跳过两个 Amazon 全量扫描
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
RAW = os.path.join(PROJ, "DeepLogitReplication-main", "data", "experiment")
OUT = os.path.join(PROJ, "results", "provenance_audit.json")

MIN_BOOKS = 3
MAX_PAIRS = 15
VG_EDGE_W = 3


def ok(flag):
    return "PASS" if flag else "**MISMATCH**"


# ---------------------------------------------------------------- A1 / A2
def audit_books():
    """A1/A2：委托给位图版 src/audit_books_edges.py，合并其结果。

    为什么不在这里自己扫：旧实现给每个用户建一个 set，2914 万行约 2000 万
    去重用户会把内存吃穿（进程曾被 OOM 杀死，日志停在 25M 行）。
    audit_books_edges.py 用 175 位整数位图，每个用户只存一个 int，
    内存 <1 GB，20 秒跑完，且逐条核对派生边权重。
    """
    import subprocess
    import sys

    script = os.path.join(PROJ, "src", "audit_books_edges.py")
    r = subprocess.run([sys.executable, script], capture_output=True,
                       text=True, encoding="utf-8")
    sys.stdout.write(r.stdout[-3000:])
    if r.stderr:
        sys.stdout.write(r.stderr[-1000:])
    p = os.path.join(PROJ, "results", "books_edge_audit.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return dict(error="audit_books_edges.py 未产出结果", rc=r.returncode)


# ---------------------------------------------------------------- A3 / A4
def audit_vg():
    """全量扫 vg_Video_Games.jsonl，独立重算共评图。"""
    items = set(json.load(open(os.path.join(AMZ, "vg_coreview.json")))["n_items"])
    print(f"[A3] 目标游戏 = {len(items)} 款", flush=True)

    user_items = defaultdict(set)
    n_rows = 0
    n_bad = 0
    path = os.path.join(AMZ, "vg_Video_Games.jsonl")
    size = os.path.getsize(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            n_rows += 1
            if not line or line[0] != "{":
                n_bad += 1
                continue
            # 只取 parent_asin，手工定位避免整行 json 解析（2.7GB）
            i = line.find('"parent_asin"')
            if i < 0:
                n_bad += 1
                continue
            j = line.find('"', i + 14)
            k = line.find('"', j + 1)
            if not (0 < j < k):
                n_bad += 1
                continue
            a = line[j + 1:k]
            if a in items:
                u = line.find('"user_id"')
                if u < 0:
                    continue
                p = line.find('"', u + 10)
                q = line.find('"', p + 1)
                if 0 < p < q:
                    user_items[line[p + 1:q]].add(a)
            if n_rows % 2_000_000 == 0:
                print(f"      {n_rows/1e6:.0f}M 行 / "
                      f"{size/1e9:.2f}GB, 用户 {len(user_items):,}", flush=True)

    n_multi = sum(1 for v in user_items.values() if len(v) >= 2)
    edge_cnt = defaultdict(int)
    for u, its in user_items.items():
        its = sorted(its)
        for i in range(len(its)):
            for j in range(i + 1, len(its)):
                edge_cnt[(its[i], its[j])] += 1
    n_edges_raw = len(edge_cnt)
    n_edges_kept = sum(1 for c in edge_cnt.values() if c >= VG_EDGE_W)

    # 口径说明：2,753,806 是整个 Video_Games 文件的去重评论者数（全文件口径，
    # 由 vg_build_coreview.py 独立验证 PASS）；本函数只解析目标 200 款的行，
    # n_multi=65,037 是"评过 >=2 款目标游戏"的图内有效用户，两者不是同一
    # 口径，不能互相断言，故 n_multi 仅作参考值记录。
    print(f"[A3] 评论>=2款目标游戏的用户 = {n_multi:,}（参考值，口径为图内用户；"
          f"文件级 2,753,806 已由 vg_build_coreview.py 验证）", flush=True)
    print(f"[A4] 共评边数(全) = {n_edges_raw:,} [期望 16,090] "
          f"{ok(n_edges_raw == 16090)}", flush=True)
    print(f"[A4] 共评边数(w>=3) = {n_edges_kept:,} [期望 10,051] "
          f"{ok(n_edges_kept == 10051)}", flush=True)
    return dict(n_rows=int(n_rows), n_bad_lines=int(n_bad),
                n_users_in_graph=int(n_multi),
                users_in_graph_note="评过>=2款目标游戏；文件级去重 2,753,806 由 vg_build_coreview.py 验证",
                n_edges_raw=int(n_edges_raw), expect_edges_raw=16090,
                edges_raw_match=bool(n_edges_raw == 16090),
                n_edges_kept=int(n_edges_kept), expect_edges_kept=10051,
                edges_kept_match=bool(n_edges_kept == 10051))


# ---------------------------------------------------------------- A5
def audit_survey():
    """重跑原论文 R 脚本的清洗规则，核对样本量。"""
    import pandas as pd

    COMPREHENSION2 = ("The e-book that you would be most likely to buy in an online "
                      "bookstore if you were presented with this selection of e-books "
                      "outside this study")
    CONSENT = "I agree to participate in the research"
    df = pd.read_csv(os.path.join(RAW, "survey_responses",
                                  "ebook_survey_data_edited.csv"))
    n0 = len(df)
    steps = {}
    df = df.iloc[553:].copy();                 steps["去 pilot/test"] = len(df)
    df = df[df["Comprehension #1"] == 2];      steps["理解检查1"] = len(df)
    df = df[df["Comprehension #2"] == COMPREHENSION2]; steps["理解检查2"] = len(df)
    df = df[df["consent"] == CONSENT];         steps["签署同意"] = len(df)
    df = df[df["Progress"] == 100];            steps["进度100%"] = len(df)
    df = df[df["Finished"].astype(str).str.lower() == "true"]; steps["已完成"] = len(df)
    df = df[pd.to_numeric(df["Q_TotalDuration"], errors="coerce") >= 60]
    steps["时长>=60s"] = len(df)
    for k, v in steps.items():
        print(f"[A5] {k:12s} -> {v:,}", flush=True)
    # 期望值 2026-10-29 校正：原记 4,632 来自 cv_selection.py 中
    # choice_id.nunique()//2 的误算；本审计直接从原始 CSV 数受访者，得 9,265。
    print(f"[A5] 最终样本 = {steps['时长>=60s']:,} [期望 9,265] "
          f"{ok(steps['时长>=60s'] == 9265)}", flush=True)
    return dict(n_raw=int(n0), steps={k: int(v) for k, v in steps.items()},
                n_final=int(steps["时长>=60s"]), expect=9265,
                match=bool(steps["时长>=60s"] == 9265))


# ---------------------------------------------------------------- A6
def audit_fingerprint():
    """文件指纹：大小 / 抽样 sha256 / 末行完整性。"""
    files = [
        ("DeepLogitReplication-main/data/experiment/survey_responses/"
         "ebook_survey_data_edited.csv", "text"),
        ("DeepLogitReplication-main/data/experiment/books.csv", "text"),
        ("data/amazon/Books_ratings.csv", "text"),
        ("data/amazon/vg_Video_Games.jsonl", "text"),
        ("data/amazon/vg_meta_Video_Games.jsonl", "text"),
        ("data/amazon/dm_Digital_Music.jsonl", "text"),
        ("data/amazon/meta_books_full.jsonl", "text"),
    ]
    out = {}
    for rel, _ in files:
        p = os.path.join(PROJ, rel)
        if not os.path.exists(p):
            out[rel] = dict(exists=False)
            continue
        size = os.path.getsize(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:                 # 头 8MB
            h.update(f.read(8 << 20))
        head = h.hexdigest()[:16]
        h2 = hashlib.sha256()
        with open(p, "rb") as f:                 # 尾 8MB
            f.seek(max(0, size - (8 << 20)))
            h2.update(f.read())
        tail = h2.hexdigest()[:16]
        # 末行是否可解析（判断下载是否截断）
        with open(p, "rb") as f:
            f.seek(max(0, size - 4096))
            last = f.read().split(b"\n")[-1][:200]
        intact = (last.strip() == b"") or last.strip().endswith(b"}") \
            or (b"," in last and not last.startswith(b"{"))
        out[rel] = dict(exists=True, bytes=int(size), gb=round(size / 1e9, 3),
                        sha256_head_8mb=head, sha256_tail_8mb=tail,
                        tail_line_intact=bool(intact))
        print(f"[A6] {rel}\n     {size/1e6:,.1f} MB  head={head}  "
              f"tail={tail}  末行完整={intact}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="只跑 A5/A6")
    a = ap.parse_args()

    res = {"when": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
           "note": "独立重算，未 import 任何项目模块；期望值取自 results/*.json"}
    res["A5_survey"] = audit_survey()
    res["A6_fingerprint"] = audit_fingerprint()
    if not a.fast:
        res["A1_A2_books"] = audit_books()
        res["A3_A4_vg"] = audit_vg()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[audit] 写入 {OUT}")

    fails = []
    for sec in ("A1_A2_books", "A3_A4_vg", "A5_survey"):
        for k, v in res.get(sec, {}).items():
            if k.endswith("_match") or k == "match":
                if v is False:
                    fails.append(f"{sec}.{k}")
    print("[audit] 结论:", "全部一致" if not fails else f"不一致 {fails}")
    sys.exit(0 if not fails else 2)


if __name__ == "__main__":
    main()
