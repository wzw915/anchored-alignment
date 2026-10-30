"""构建书籍选择实验数据集（对齐论文 0_experiment_build_dataset.R 与 load_data_long）。

步骤:
  1. 清洗 Qualtrics 原始导出: 去 pilot、通过理解性检查、进度完整、时长>=60s -> 9265 人
  2. product_id -> asin 字典 (通过标题匹配 books.csv)
  3. 长格式选择数据: 每个任务 10/9 个备选, 含 price/position/choice
  4. 经验 diversion 矩阵: S[j,k] = P(第二选 = k | 第一选 = j)
  5. 结构化属性: year/pages 标准化, genre 哑变量

输出: cache/first_choice.csv, cache/second_choice.csv, cache/diversion_empirical.npy,
      cache/books_clean.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJ, "DeepLogitReplication-main", "data", "experiment")
CACHE = os.path.join(PROJ, "cache")
os.makedirs(CACHE, exist_ok=True)

COMPREHENSION2_ANS = ("The e-book that you would be most likely to buy in an online "
                      "bookstore if you were presented with this selection of e-books "
                      "outside this study")
CONSENT = "I agree to participate in the research"


def norm_title(s: str) -> str:
    """统一撇号与空白，供标题匹配。"""
    return re.sub(r"\s+", " ", str(s).replace("\u2019", "'").replace("`", "'")).strip()


def main():
    # ---------- 1. 清洗 ----------
    df = pd.read_csv(os.path.join(RAW, "survey_responses", "ebook_survey_data_edited.csv"))
    n0 = len(df)
    df = df.iloc[553:].copy()                       # 去 pilot/test（对应 R 的 554:nrow）
    n_pilot = len(df)
    df = df[df["Comprehension #1"] == 2]
    n_c1 = len(df)
    df = df[df["Comprehension #2"] == COMPREHENSION2_ANS]
    n_c2 = len(df)
    df = df[df["consent"] == CONSENT]
    df = df[df["Progress"] == 100]
    df = df[df["Finished"].astype(str).str.lower() == "true"]
    df = df[pd.to_numeric(df["Q_TotalDuration"], errors="coerce") >= 60].copy()
    df["id_var"] = np.arange(1, len(df) + 1)
    print(f"[build] 原始 {n0} -> 去pilot {n_pilot} -> 理解1 {n_c1} -> 理解2 {n_c2} "
          f"-> 全部筛选 {len(df)} (论文: 9265)")

    # ---------- 2. product_id -> asin 字典 ----------
    books = pd.read_csv(os.path.join(RAW, "books", "books.csv"))
    books["title_norm"] = books["title"].map(norm_title)
    title2asin = dict(zip(books["title_norm"], books["asin"]))

    first_row = df.iloc[0]
    pid_list = [int(x) for x in str(first_row["order_array_id_1"]).split(";")]
    title_list = [norm_title(t) for t in str(first_row["order_array_titles_1"]).split(";")]
    pid2asin = {}
    for pid, t in zip(pid_list, title_list):
        if t not in title2asin:
            raise KeyError(f"标题无法匹配: {t!r}")
        pid2asin[pid] = title2asin[t]
    assert len(pid2asin) == 10, "应为 10 本书"
    print(f"[build] product_id 字典: {len(pid2asin)} 本 (样例: {pid2asin[pid_list[0]]} <- {pid_list[0]})")

    # ---------- 3. 长格式 ----------
    def long_task(task: int) -> pd.DataFrame:
        sub = df[[f"order_array_id_{task}", f"order_array_prices_{task}",
                  f"selected_id_{task}", f"id_var"]].copy()
        rows = []
        for _, r in sub.iterrows():
            pids = [int(x) for x in str(r[f"order_array_id_{task}"]).split(";")]
            prices = [float(x) for x in str(r[f"order_array_prices_{task}"]).split(";")]
            sel = int(r[f"selected_id_{task}"])
            for pos, (pid, pr) in enumerate(zip(pids, prices), start=1):
                rows.append((r["id_var"], task, pid, pr, pos, int(pid == sel)))
        out = pd.DataFrame(rows, columns=["respondent_id", "choice_number", "product_id",
                                          "price", "position", "choice"])
        out["choice_id"] = out["respondent_id"] * 2 + (task - 1)
        return out.sort_values(["respondent_id", "product_id"]).reset_index(drop=True)

    first, second = long_task(1), long_task(2)
    print(f"[build] 一选 {len(first)//10} 人 x 10 备选 = {len(first)} 行; "
          f"二选 {len(second)//9} 人 x 9 备选 = {len(second)} 行")
    assert (first.groupby("choice_id")["choice"].sum() == 1).all()
    assert (second.groupby("choice_id")["choice"].sum() == 1).all()

    # ---------- 4. 经验 diversion 矩阵 ----------
    pids_sorted = sorted(pid2asin)
    idx = {p: i for i, p in enumerate(pids_sorted)}
    J = len(pids_sorted)
    cnt = np.zeros((J, J))
    fc = first[first["choice"] == 1].set_index("respondent_id")["product_id"]
    sc = second[second["choice"] == 1].set_index("respondent_id")["product_id"]
    both = fc.index.intersection(sc.index)
    for r in both:
        cnt[idx[fc[r]], idx[sc[r]]] += 1
    div = cnt / cnt.sum(axis=1, keepdims=True)
    assert np.allclose(div.sum(axis=1), 1)
    print(f"[build] diversion 矩阵 {div.shape}, 行和=1, 总样本 {int(cnt.sum())}")

    # ---------- 5. 属性与输出 ----------
    asin2 = {a: a for a in books["asin"]}
    for d, name in ((first, "first_choice"), (second, "second_choice")):
        d["asin"] = d["product_id"].map(pid2asin)
        d = d.merge(books[["asin", "genre", "number_of_pages", "publication_year"]], on="asin")
        d["year"] = (d["publication_year"] - d["publication_year"].mean()) / d["publication_year"].std()
        d["pages"] = (d["number_of_pages"] - d["number_of_pages"].mean()) / d["number_of_pages"].std()
        d["genre_mystery"] = (d["genre"] == "Mystery, Thriller & Suspense").astype(int)
        d["genre_scifi"] = (d["genre"] == "Science Fiction & Fantasy").astype(int)
        for p in pids_sorted:
            if p != 47:                              # 基准: product_id 47 (The Art of Letting Go)
                d[f"product_id_{p}"] = (d["product_id"] == p).astype(int)
        d.to_csv(os.path.join(CACHE, f"{name}.csv"), index=False)

    np.save(os.path.join(CACHE, "diversion_empirical.npy"), div)
    meta = pd.DataFrame({"product_id": pids_sorted,
                         "asin": [pid2asin[p] for p in pids_sorted]})
    meta = meta.merge(books[["asin", "title", "author", "genre", "number_of_pages",
                             "publication_year", "description", "user_review_1",
                             "user_review_2", "user_review_3", "user_review_4",
                             "user_review_5"]], on="asin")
    meta.to_csv(os.path.join(CACHE, "books_clean.csv"), index=False)
    print("[build] 完成 -> cache/", os.listdir(CACHE))


if __name__ == "__main__":
    sys.exit(main())
