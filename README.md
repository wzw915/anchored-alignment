# Anchored Alignment: Demand Substitution from Text and Image Embeddings

Companion repository for the manuscript *"Zero-Shot Estimation of Product
Substitution from Multimodal Embeddings via Anchored Alignment"*.

Individual-level fine-tuning of pretrained embeddings destroys the global
geometry that product-level demand tasks consume: on a book choice experiment,
full fine-tuning raises diversion-matrix RMSE by 52% (p = 0.006) even though it
sees the same data. **Anchored alignment** fixes this by learning a bounded
residual on top of frozen anchors, `e_j = c_j + Δ_j` with `‖Δ_j‖ ≤ δ_max`:

| Method | RMSE | Kendall τ | Top-3 | Profit loss (pricing) |
|---|---|---|---|---|
| Frozen embeddings | 0.0808 | 0.472 | — | 0.21% |
| Full fine-tuning | 0.1225 | 0.523 | — | **25.45%** |
| **Anchored alignment (δ_max = 0.5)** | **0.0774** | **0.656** | no loss | **0.17%** |

The ranking gain replicates on two Amazon behavioral datasets (books
co-purchase, +31%; video-games co-review, +29%; p < 0.0001), and in merger
screening the fine-tuned matrix waves through all nine mergers that the true
matrix flags for review.

## Repository layout

```
├── src/                  # all pipeline code (paths derived from __file__)
├── results/              # JSON/NPY outputs of every stage (regenerable)
├── figures/              # paper figures (vector PDF + 300 dpi PNG)
├── REPRODUCE.md          # step-by-step reproduction guide with expected outputs
├── DATA_SOURCES.md       # provenance, byte counts and SHA-256 of every input
└── requirements.txt
```

## Quick start

```bash
python 3.11+   # verified on 3.13
pip install -r requirements.txt

# 1. obtain the survey data (18 MB) — see REPRODUCE.md §1
# 2. build tables and embeddings (~6 min)
# 3. estimation, ablations, external validation, pricing/merger sims
```

Full instructions, expected outputs and per-figure checkpoints are in
[REPRODUCE.md](REPRODUCE.md). Raw external corpora (Amazon Reviews 2023,
≈17 GB) are downloaded on demand; every downstream step is reproducible from
byte-fingerprinted inputs (see [DATA_SOURCES.md](DATA_SOURCES.md)).

## Data

| Dataset | Source | Size |
|---|---|---|
| Book choice experiment (main) | [Official replication package](https://github.com/ilyamorozov/DeepLogitReplication) of Compiani, Morozov & Seiler (2026), *Demand Estimation with Text and Image Data* | 9,265 respondents |
| Amazon Books co-purchase | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (`raw/meta_Books.jsonl`, `raw/review_Books.jsonl`) | 29.1 M reviews, 2.49 M multi-book buyers |
| Amazon Video Games co-review | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (`raw/meta_Video_Games.jsonl`, `raw/review_Video_Games.jsonl`) | 4.6 M reviews, 2.75 M reviewers |

Download commands for every raw file (with a China-accessible mirror fallback)
are in [REPRODUCE.md](REPRODUCE.md); byte counts and SHA-256 fingerprints of
each source file are in [DATA_SOURCES.md](DATA_SOURCES.md).

No synthetic data is used anywhere in the pipeline. The anonymized,
aggregated survey matrix is included under `results/`; raw survey responses
containing respondent-level information are available from the authors on
request.

## License

Code is released under the [MIT License](LICENSE).
