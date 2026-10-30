# 数据来源与完整性声明

本文件回答三个问题：每个数字来自哪个文件、文件从哪来、别人能不能从公开源重算出来。
所有条目都可以用 `src/audit_provenance.py` 复核，不需要相信本文档的任何一句话。

最后更新：2026-10-29（审计脚本产出 `results/provenance_audit.json`）

---

## 0. 结论先行

| 数据集 | 性质 | 从公开源可复现 | 关键校验 |
|---|---|---|---|
| 1 书籍选择实验（survey） | 第三方公开数据，作者复现包 | 是，逐位一致 | 清洗漏斗 11,328 → 9,265，与 R 脚本一致 |
| 2 Amazon Books 共购 | 本地上传到 HF 的公开数据派生 | 是，名单逐字一致 | 175 本名单完全相同、文本 175/175 逐字相同、诱导子图 6,906 边复现 |
| 3 Amazon Video Games 共评 | 同上 | 是，字节级完整 | 原始文件与 HF 服务端 Content-Length 完全相等；去重评论者 2,753,806 精确复现 |

**没有任何一个数字是构造的。** 三个数据集全部由磁盘上的公开原始文件经脚本计算得到，
脚本、中间产物、审计输出都在仓库内。下面逐项列出。

---

## 1. 数据集 1：书籍选择实验（主研究）

### 来源
第三方复现包，非本项目生成。原始包内文件原样使用，未做任何修改：

| 文件 | 字节 | sha256(前 4 MB) | 说明 |
|---|---|---|---|
| `DeepLogitReplication-main/data/experiment/survey_responses/ebook_survey_data_edited.csv` | 18,245,813 | `5d929b875a76…` | Qualtrics 导出，11,328 行 |
| `DeepLogitReplication-main/data/experiment/books/books.csv` | 39,955 | `8d4a61c14a72…` | 10 本书的书名/简介/5 条评论/页数/年份/类型 |
| `DeepLogitReplication-main/data/experiment/books/book_covers/*.jpg` | 10 个文件 | — | 10 张封面，与 `books.csv` 的 asin 一一对应 |

对应论文：Compiani, Morozov & Seiler (2026), *Demand Estimation with Text and Image Data*。

### 清洗链路（`src/build_data.py`）
```
11,328 原始行
  → 去掉 pilot/test 段（iloc[553:]）        10,775
  → 理解性检查 1 通过                        9,860
  → 理解性检查 2 通过                        9,344
  → 签署同意 + 进度 100% + Finished          9,265
  → 时长 >= 60 秒                            9,265
```
审计脚本独立重跑同一套规则，七步人数与上表完全一致（`results/provenance_audit.json` 的 `A5_survey`）。

**样本量口径**：9,265 名受访者，每人完成 2 个选择任务（第一选 10 个备选、第二选 9 个），
故长表为 92,650 + 83,385 行。早期 `results/cv_selection.json` 曾把该值记为 4,632，
原因是 `choice_id.nunique() // 2` 的整数除法错误（`choice_id = respondent_id * 2 + task - 1`，
而 `first` 只含 task=1，其唯一值数就等于受访者数）。该字段已于 2026-10-29 修正为 9,265，
折分与全部估计结果不受影响，论文正文引用的始终是 9,265。

---

## 2. 数据集 2：Amazon Books 共购

数据源：HuggingFace `McAuley-Lab/Amazon-Reviews-2023`（镜像 `hf-mirror.com`），
与论文的 40 类目泛化实验同源，可独立下载校验。

### 2.1 原始文件与完整性

| 本地文件 | 本地字节 | HF 官方字节 | 覆盖 | 用途 |
|---|---|---|---|---|
| `data/amazon/Books_ratings.csv` | 1,689,808,538 | （由 20,121,186,727 的 `review_categories/Books.jsonl` 转换） | 全量 | 共购图 |
| `data/amazon/meta_books_full.jsonl` | 5,050,452,475 | 14,697,581,022 | **34.4% 前缀** | 书名/简介提取 |
| `data/amazon/meta_books_head.jsonl` | 2,147,483,649 | 同上 | 前 2 GB 段 | 同上（与 full 重叠） |

`Books_ratings.csv` 是 29,139,329 行、时间跨度 1996-06-17 至 2023-09-14 的完整类目点评，
抽样统计到 570 万去重用户、252 万去重书目，确认是全量而非片段。

**必须如实说明的两点**：

1. `meta_Books.jsonl` 官方 14.70 GB，本地只有连续前缀 5.05 GB（34.4%）。
   `meta_books_head.jsonl` 与 `meta_books_full.jsonl` 的前 4 MB sha256 相同，
   确认后者是在前者基础上续传追加的结果，两者是同一文件的 2 GB 段与 5 GB 段，不可相加。
2. `data/amazon/parts/part_*.bin`（8 个，合计 5.47 GB）是当时 Range 下载的原始分片残留，
   **未参与任何计算**，可以安全删除。

前缀截断对结果是安全的：共购图在 400 本书上建，其中能在 meta 前缀里取到文本的恰好 175 本，
这 175 本全部命中（见 2.3 的复现校验）。但反过来说，**用完整 meta 可以得到超过 175 本**，
所以 175 这个数依赖本地前缀长度，这是本研究的一个数据边界，不应被读成"只有 175 本有文本"。

### 2.2 派生链路

```
Books_ratings.csv (29,139,329 行, 1996-2023)
  └─ src/amazon_build.py            → copurchase_edges.json
       阈值：每用户购书 >= 3 本；每用户最多取 15 对；取度数 top-400
       产出：400 本书 / 37,542 条边 / 2,492,843 名多书用户
  └─ src/amazon_pick_covered.py     → covered_asins.json
       在 meta 前缀里能取到文本的 400 本 → 175 本
  └─ 175 本诱导子图                 → 6,906 条边（= results/amazon_dataset2.json 的 n_edges）
```

`Books_ratings.csv` 从 HF 原始 jsonl 生成的脚本为 `src/amazon_books_reviews.py`
（边下边转，不落 20 GB 中间文件；默认拒绝覆盖已有文件，避免误毁正式数据）。

### 2.3 复现校验（已完成，可重跑）

| 校验项 | 重建 | 论文记录 | 结果 |
|---|---|---|---|
| 多书用户数（>=3 本） | 2,492,843 | 2,492,843 | 一致 |
| 175 本名单 | 175 本 | 175 本 | **集合完全相同** |
| 175 本文本 | — | — | **175/175 逐字相同** |
| 175 本诱导子图边数 | 6,906 | 6,906 | 一致 |
| 边权抽样回原始 CSV 重数 | 见审计输出 | — | 派生权重 ≤ 原始共同买家数（截断只会少算） |

`covered_asins.json` 的复现命令：`python src/amazon_pick_covered.py`
（扫 `meta_books_head.jsonl` 58.9 万行 + `meta_books_full.jsonl` 146.7 万行，命中 175）。

---

## 3. 数据集 3：Amazon Video Games 共评

### 3.1 原始文件与完整性（本数据集最硬的一条）

| 本地文件 | 本地字节 | HF 官方字节 | 判定 |
|---|---|---|---|
| `data/amazon/vg_Video_Games.jsonl` | **2,682,948,518** | **2,682,948,518** | 字节级完整 |
| `data/amazon/vg_meta_Video_Games.jsonl` | **437,375,785** | **437,375,785** | 字节级完整 |

两个文件与 HF 服务端返回的 `Content-Length` 完全相等，行级解析坏行数为 0，
即整份 462 万条评论一条不少。这是三个数据集中最干净的一个。

`vg_meta_Video_Games.jsonl` 的第一行是下载分片拼接处的半行碎片，
所有读取脚本用 `if line[0] != "{": continue` 跳过，不影响任何结果（已计入"坏行数"审计）。

### 3.2 派生链路

```
src/vg_download2.py                  → vg_Video_Games.jsonl / vg_meta_Video_Games.jsonl
src/vg_build_coreview.py             → vg_coreview.json
      Pass1  统计每款游戏评论条数 → 取 top-200
      Pass2  只保留这 200 款，累积 用户 → 评过的游戏集合
      Pass3  用户内两两成边，边权 = 共同评论该对游戏的用户数
      产出：200 款 / 16,090 条边 / 2,753,806 名去重评论者
src/amazon_dataset3.py  (MIN_EDGES=3) → results/amazon_dataset3.json
      过滤 weight < 3 → 10,051 条边 → LOO 10 折不适用，改为留一项出
```

### 3.3 复现校验（已完成，可重跑）

| 校验项 | 重建 | 论文记录 | 结果 |
|---|---|---|---|
| 去重评论者数（全文件） | 2,753,806 | 2,753,806 | **精确一致** |
| 200 款名单（沿用原始名单） | 200 | 200 | 一致 |
| 共评边数（沿用原始名单） | 见 `src/vg_rebuild_fixed.log` | 16,090 | 见日志 |
| 共评边权 | — | — | 15859 条共有边**权重零差异** |
| 自动选 top-200 | 200 款，与原始差 2 款 | — | 并列边界取舍差异，见下 |

**关于自动选取的 2 款差异**：`vg_build_coreview.py` 不带 `--items-from` 时，
按评论条数降序自动取 top-200，在边界（2031 条评论附近）与其他游戏并列，
tie-break 与原始选法相差 2 款（`B00I9TNNE4`/`B0CBRT4BWW` vs `B00G2EVF3E`/`B00HX1UXD8`），
导致全边数 16,186 而非 16,090。**15859 条共有边的权重完全相同**，说明边权计算逻辑无误，
差异只来自这 2 款项。要逐位复现论文数字，请加 `--items-from data/amazon/vg_coreview.json`。

### 3.4 用户数口径（易被误读，务必区分）

| 数字 | 含义 | 出处 |
|---|---|---|
| 2,753,806 | Video Games 整份文件的去重评论者数 | `vg_coreview.json` 的 `n_users` |
| 65,037 | 其中评过 >= 2 款目标游戏的用户（真正参与构图） | `vg_coreview.json` 的 `n_users_graph` |
| 2,492,843 | Books 全类目中买过 >= 3 本书的用户 | `copurchase_edges.json` 的 `n_users_multi` |
| 待补 | Books 在 175 本内买过 >= 2 本的用户（与 65,037 同口径） | `audit_provenance.py` 的 `A1.n_users_in_graph` |

论文正文说"diversion proxies come from 2.49 million book buyers and 2.75 million game reviewers"，
其中两个数**不是同一口径**（前者是构图用户，后者是文件规模）。审稿人若追问，
应当用 65,037 与 Books 的同口径数字对比，而不是 2.49 M vs 2.75 M。

---

## 4. 已弃用的数据

`dm_Digital_Music.jsonl`（78.8 MB）、`meta_Digital_Music.jsonl`（67.1 MB）、`music_coreview.json`
是早期尝试的第三类目。弃用原因：Digital Music 类目共评图仅 172 条边、平均度 3，
统计上不足以支撑 LOO 评估（对比 Video Games 10,051 条边、平均度 161）。
文件保留供追溯，**不参与任何论文结论**。

---

## 5. 一键复核

```bash
PY="python"   # 需 pandas / numpy / scipy / torch / sentence-transformers

# 快速审计（秒级）：原始 survey 清洗漏斗 + 全部文件指纹
$PY src/audit_provenance.py --fast

# 完整审计（约 20 分钟）：另加 Books 全量用户统计、VG 全量重算、边权抽样回数
$PY src/audit_provenance.py

# 从 HF 原始文件重建两个图，并对账论文记录
$PY src/vg_build_coreview.py --items-from data/amazon/vg_coreview.json
$PY src/amazon_pick_covered.py
$PY src/amazon_build.py
```

审计输出写入 `results/provenance_audit.json`，逐项带 `*_match` 布尔值，
任一不符即以退出码 2 结束，可直接接进 CI。

---

## 6. 许可与引用

- Amazon Reviews 2023：McAuley Lab, UC San Diego。使用须遵守其仓库条款，本仓库不再分发原始 jsonl。
- 书籍选择实验数据：来自上述论文的官方复现包，随包许可使用。
- 本项目脚本：仅供课程与研究复现，不含任何再分发的原始评论内容。
