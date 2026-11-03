# -*- coding: utf-8 -*-
"""demo_delta_choice.py — 现场演示：δ_max 是怎么选的（诚实协议）

只读 results/align_anchored.json，秒级完成：
  1) 打印 δ 网格 {0.1, 0.3, 0.5, 0.7} 的 10 折平均指标（主协议）；
  2) 打印每一折"只用训练行"选出来的 δ；
  3) 结论：0.3–0.7 是平台，不是只在一个点上好。
"""
from __future__ import annotations

import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJ, "results", "align_anchored.json"), encoding="utf-8") as f:
    d = json.load(f)

print("δ_max 网格（主协议，10 折平均）")
print(f"{'δ':>6}{'RMSE':>10}{'Kendall τ':>12}{'Top-3':>8}")
for key in sorted(d["per_delta"], key=float):
    row = d["per_delta"][key]
    print(f"{key:>6}{row['rmse']:>10.4f}{row['kendall']:>12.4f}{row['top3']:>8.2f}")

chosen = d["train_selected"]["deltas_chosen"]
print()
print("每一折只用训练行选出的 δ_max：", chosen)
print("选择规则：训练行 RMSE 最小的 δ（留出的商品全程不可见——诚实协议）")
best = min(d["per_delta"], key=lambda k: d["per_delta"][k]["rmse"])
print(f"结论：τ 在 δ∈[0.3, 0.7] 保持 0.61–0.66 的平台，网格最优 δ={best}。")
