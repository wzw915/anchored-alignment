# 复现指南

从零开始的全流程：装环境 → 拉数据 → 建图 → 嵌入 → 估计 → 作图。
每一步都给出**预期输出**，跑完可以和论文数字逐项对照；对不上就是环境或数据版本问题。

数据来源与完整性见 [DATA_SOURCES.md](DATA_SOURCES.md)。

---

## 0. 环境

```bash
python 3.11+          # 本项目在 3.13 上验证
pip install -r requirements.txt
```

核心依赖：`numpy scipy pandas pyarrow torch` + `sentence-transformers`（文本嵌入）
+ `timm`（封面图像嵌入）+ `xlogit`（mixed logit 估计）+ `matplotlib`。

模型权重（约 470 MB，首次运行自动从 HF 镜像下载到 `models/`）：
`models/st_minilm`（all-MiniLM-L6-v2）、`models/resnet50`、`models/inception_v3`。

---

## 1. 数据获取

### 1.1 主数据集（18 MB，必需，秒级）

书籍选择实验数据来自原论文复现包，放在 `DeepLogitReplication-main/data/experiment/`。
若目录为空，从原论文的 replication package 取回，必须包含：

```
survey_responses/ebook_survey_data_edited.csv   18.2 MB   11,328 行
books/books.csv                                 40 KB     10 本书的属性
books/book_covers/*.jpg                         10 张封面
```

### 1.2 外部验证数据（可选，17 GB，慢）

不做第 5 节的跨类目验证可以完全跳过这一步。

```bash
# Video Games：两个文件，字节级完整，约 3.1 GB
python src/vg_download2.py

# Books 点评转 CSV：流式读 HF 原始 20 GB jsonl，边下边转，不占额外磁盘
python src/amazon_books_reviews.py

# Books 元数据：14.7 GB，慢且易断；只需要覆盖到 175 本书的前 5 GB 前缀即可
# （本研究用的就是 5.05 GB 前缀，详见 DATA_SOURCES.md 第 2.1 节的说明）
```

`vg_download2.py` 带僵死检测（`--speed-limit`）与断点续传（先写 `.new` 再追加），
中断后重跑会接着下。不要用 `curl -o` 直接覆盖目标文件，会把已下部分清零。

---

## 2. 主数据集：建表与嵌入（约 6 分钟）

```bash
python src/build_data.py        # 清洗 → 长表 → 经验 diversion 矩阵
python src/embeddings.py        # 15 个嵌入规格 + split-half 信度
```

**预期输出**

| 步骤 | 指标 | 值 |
|---|---|---|
| build_data | 受访者 | 9,265（自 11,328 起，七步漏斗） |
| build_data | 第一选长表 | 92,650 行（9,265 × 10 备选） |
| build_data | 第二选长表 | 83,385 行（9,265 × 9 备选） |
| build_data | diversion 矩阵 | 10 × 10，行和 = 1 |
| embeddings | 规格数 | 15（title/description/reviews/image × 各变体） |

清洗漏斗的七步人数若与 `results/provenance_audit.json` 的 `A5_survey.steps` 有任何一步不同，
先查 `ebook_survey_data_edited.csv` 是否被改动过（sha256 前 4 MB 应为 `5d929b875a76…`）。

---

## 3. 估计与选型（约 40 分钟）

```bash
python src/est_logit.py --spec reviews_ST_chunkmean   # 单个规格
python src/run_experiments.py                          # 全部 17 个规格
python src/cv_selection.py                             # 5 折交叉验证选型
python src/replication_checks.py --phase ident         # 识别性检查
python src/replication_checks.py --phase draws         # Halton 抽样数敏感性
```

**预期输出**：`results/model_*.json`（每规格 AIC / RMSE / diversion 预测）、
`results/cv_selection.json`、`results/replication_checks.json`。

关键校验：`reviews_ST_chunkmean` 的二选 RMSE 应为 **0.0667**（论文最优规格）。

---

## 4. 锚定对齐消融（约 2 小时，CPU）

这是论文核心实验，双协议 × 7 方法的留一项出评估。

```bash
python src/ablation_loo.py     # 单位球协议
python src/ablation_loo2.py    # 中心化协议
```

**预期输出**（`results/ablation_loo.json`，10 折均值 ± 折间标准差）

| 方法 | 单位球 RMSE | 单位球 τ | 中心化 RMSE | 中心化 τ |
|---|---|---|---|---|
| frozen | 0.0808 | 0.472 | 0.0725 | 0.478 |
| fullft | 0.1225 | 0.433 | 0.1218 | 0.139 |
| unbounded | 0.0788 | 0.656 | **0.1399** | 0.511 |
| anchored 0.1 | 0.0798 | 0.517 | **0.0704** | 0.489 |
| anchored 0.3 | 0.0777 | 0.611 | 0.0831 | 0.517 |
| anchored 0.5 | 0.0774 | **0.656** | 0.1055 | 0.528 |
| anchored 0.7 | 0.0781 | 0.656 | 0.1217 | 0.544 |

两个必须复现的现象：无界残差在单位球下与 anchored-0.5 打平（τ 都是 0.656），
但在中心化协议下塌到 0.1399；anchored-0.1 在中心化下反而最优（0.0704）。
这就是论文"监督层级是发动机、范数球是保险"的经验依据。

---

## 5. 跨类目验证（可选，约 2.5 小时，需要第 1.2 节的数据）

```bash
python src/amazon_build.py                    # Books 共购图（扫 2,914 万行）
python src/amazon_pick_covered.py             # 400 本 → 有文本的 175 本
python src/vg_build_coreview.py --items-from data/amazon/vg_coreview.json
python src/amazon_dataset2.py                 # Books LOO，175 折
python src/amazon_dataset3.py                 # Video Games LOO，200 折
python src/fig_external.py                    # 图 4
```

**预期输出**

| 数据集 | 方法 | RMSE | Kendall τ | Top-3 |
|---|---|---|---|---|
| Books（175 本，6,906 边） | frozen | 0.0136 | 0.160 | 0.43 |
| | anchored 0.5 | 0.0134 | **0.208** | 0.42 |
| | unbounded | 0.0136 | 0.258 | **0.21** |
| Video Games（200 款，10,051 边） | frozen | 0.0091 | 0.199 | 0.33 |
| | anchored 0.5 | 0.0088 | **0.257** | 0.32 |
| | unbounded | 0.0088 | 0.276 | 0.31 |

数据构建环节的对账（防止用错版本的原始文件）：

| 检查 | 期望 |
|---|---|
| Books 多书用户数（>=3 本） | 2,492,843 |
| Books 175 本名单 | 与 `covered_asins.json` 集合完全相同 |
| Books 175 本诱导子图边数 | 6,906 |
| VG 去重评论者数 | 2,753,806 |
| VG 共评边数（weight >= 3） | 10,051 |

`amazon_dataset2.py` / `amazon_dataset3.py` 的置换检验对 175/200 折用 20,000 次蒙特卡洛
（全枚举 2^175 不可行）；主研究是 10 折，用 2^10 = 1,024 次全枚举，两者方法不同但都写成符号翻转。

---

## 6. 反事实模拟与出图（约 10 分钟）

```bash
python src/pricing_sim.py        # 定价：给 10 本书重新定价
python src/merger_sim.py         # 合并：9 组两两合并
python src/tsne_fig3.py          # 表示漂移 t-SNE
python src/paper_figs.py         # 主图
python src/graphical_abstract.py # 图形摘要
```

这些脚本**不含新的数据**：它们读第 3–5 节的估计结果与 diversion 矩阵做数值模拟。
模拟参数（价格网格、合并组合、利润函数）全部写在脚本顶部的常量里，
论文引用模拟结果时应在文中标注"基于估计参数的反事实模拟"。

---

## 7. 一键复核数据真实性

```bash
python src/audit_provenance.py --fast   # 秒级：survey 清洗漏斗 + 全部文件指纹
python src/audit_books_edges.py         # 约 20 秒：位图法重数共购边（内存 <1 GB）
python src/amazon_build_pure.py         # 约 90 秒：纯 Python 复刻 amazon_build.py，逐条对比 37,542 条边
python src/synth_control.py --reps 5    # 30 秒：合成对照，验证个体级算法本身没有写错
python src/audit_provenance.py          # 全量：委托位图法 + VG 全量重算 + 指纹
```

注意：不要在本机直接跑 pandas 版的 `amazon_build.py` 来做验证——pandas 分块
读取 1.7 GB CSV 会被 SIGTERM 杀掉（纯 csv 模块则稳定，见 `amazon_build_pure.py`，
它逐条对齐了 pandas chunksize=2M / groupby(sort=True) / dict.fromkeys 的语义）。

输出 `results/provenance_audit.json`、`results/books_edge_audit.json`、
`results/synth_control.json`，逐项带 `*_match` / `ok` 布尔值，任何一项不符退出码为 2。
这些脚本不 import 任何项目模块中与估计相关的部分，用独立实现重算，
是判断"数据是否真实、代码是否自洽"的入口。

三者的分工要说清楚，否则容易被误读：

- `audit_provenance.py` 证明**原始文件是真的**（体积、抽样 sha256、末行完整）且
  survey 的清洗漏斗逐级可复现；
- `audit_books_edges.py` 证明**派生边表是从原始记录聚合出来的**——它把 175 本
  书编码成整数位图，流式扫完 2914 万行，重新数出共同买家，再与
  `copurchase_edges.json` 的权重逐条比对。结论：6,906 条边全部是重算子集、
  权重越界为 0；权重系统性偏低来自原构建 `MAX_PAIRS=15` 的截断（已在论文
  4.7 节披露），无截断版 `copurchase_edges_full.json` 上重跑的结论不变。
- `amazon_build_pure.py` 是更强的证据：**用纯 Python 逐条复刻原构建脚本，
  输出的 2,492,843 个多书用户、400 本名单、37,542 条边与现有文件完全一致**，
  即边表可由"原始 CSV + 脚本"完全再生。
- `synth_control.py` 证明**算法本身没有实现错误**，与数据真实性无关。它生成一个
  已知需求子空间，看同一套 InfoNCE 能否把它找回来。论文引用的数字口径是
  "R² = 0.75（五次重复，SD 0.07），冻结表示 −0.86"；这个 R² 用 PCA 降维后的
  线性探针测量，因为 40 个产品配 384 维时，未降维的 OLS 即使真值就在前两维
  也只会得到负 R²。换探针维数（2/4/6/8）不改变排序结论。

---

## 8. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `vg_meta` 第一行 json 解析失败 | 分片拼接处的半行碎片 | 读取时 `if line[0] != "{": continue`，不影响结果 |
| Books 下载 430 MB ↔ 7 MB 反复振荡 | `curl -o` 整文件覆盖 | 先下到 `.new` 再 `>>` 追加，见 `vg_download2.py` |
| 置换检验跑不动 | 折数 > 16 时 2^n 全枚举爆炸 | 改蒙特卡洛 20,000 次 |
| 嵌入全为零向量 | HF 直连不可用导致权重下载为空文件 | 用 `src/hf_fetch.py` 从 `hf-mirror.com` 重下 |
| `cv_selection.json` 样本量显示 4,632 | 早期整数除法 bug，已修 | 应为 9,265，见 DATA_SOURCES.md 第 1 节 |

---

## 9. 论文产出链

```bash
# Word 版（课程提交，TNR 12pt、三线表、无封面）
python src/build_paper_docx.py deliverable/英文小论文_v2.md deliverable/Anchored_Alignment_v3.2.docx

# LaTeX 投稿版（elsarticle，DSS 格式）
python src/md_to_elsarticle.py        # 生成 deliverable/paper_elsarticle.tex
pdflatex paper_elsarticle.tex && pdflatex paper_elsarticle.tex
```

`paper_elsarticle.tex` 要点：引用 [n] 全部映射为 `\cite{refN}`（37 条 thebibliography，
编号 = 正文首现顺序）；四个编号公式为 equation 环境；表格 booktabs 三线表；
图片优先嵌入同名 PDF 矢量图。作者与单位信息为占位符，投稿前填写。
