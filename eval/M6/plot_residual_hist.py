"""
残差分布直方图对比图
各 SNR 等级叠绘，标注 Residual_STD / Mean / Skewness
输出: residual_hist_compare.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── 路径 ──────────────────────────────────────────────────────────────────────
CASES_DIR = "/home/linux1917366562/MREPT_code/eval/M6/mae_compare_nib/cases"
OUTPUT    = "/home/linux1917366562/MREPT_code/eval/M6/residual_hist_compare.png"

# ── 读取数据 ──────────────────────────────────────────────────────────────────
cases = sorted(os.listdir(CASES_DIR))          # M6_SNR010 … M6_SNR150
records = []
for case in cases:
    path = os.path.join(CASES_DIR, case, "case_report.json")
    with open(path) as f:
        d = json.load(f)
    rd = d["residual_distribution"]
    snr_str = case.replace("M6_SNR", "")        # "010" → int 10
    records.append({
        "snr"    : int(snr_str),
        "label"  : f"SNR {int(snr_str)}",
        "edges"  : np.array(rd["Histogram_Edges"]),
        "counts" : np.array(rd["Histogram_Counts"], dtype=float),
        "mean"   : rd["Residual_Mean"],
        "std"    : rd["Residual_STD"],
        "skew"   : rd["Residual_Skewness"],
    })

# ── Bin 中心 & 归一化为概率密度 ───────────────────────────────────────────────
for r in records:
    edges   = r["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths  = np.diff(edges)
    r["centers"] = centers
    # 归一化为概率密度 (pdf)，面积 = 1
    r["pdf"] = r["counts"] / (r["counts"].sum() * widths)

# ── 配色：低 SNR = 深蓝，高 SNR = 深红 ───────────────────────────────────────
n = len(records)
cmap  = cm.get_cmap("RdYlBu_r", n)
colors = [cmap(i / (n - 1)) for i in range(n)]

# ── 绘图 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                          gridspec_kw={"width_ratios": [2.5, 1]})

ax = axes[0]

for i, r in enumerate(records):
    # 阶梯折线（step-post）
    ax.step(r["centers"], r["pdf"],
            where="mid",
            color=colors[i],
            linewidth=1.8,
            alpha=0.85,
            label=f"{r['label']}  (σ={r['std']:.2f}, μ={r['mean']:+.2f})")

# 参考正态曲线（SNR010 & SNR150）
from scipy.stats import norm
x_fit = np.linspace(-6, 6, 300)
for idx in [0, -1]:
    r = records[idx]
    y_fit = norm.pdf(x_fit, r["mean"], r["std"])
    ax.plot(x_fit, y_fit, "--", color=colors[idx], linewidth=1.0, alpha=0.5)

ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel("Residual (S/m)", fontsize=12)
ax.set_ylabel("Probability Density", fontsize=12)
ax.set_title("Residual Distribution by SNR Level", fontsize=13, fontweight="bold")
ax.set_xlim(-6, 6)
ax.legend(fontsize=8.5, framealpha=0.7, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.4)

# ── 右侧：统计量折线 ──────────────────────────────────────────────────────────
ax2 = axes[1]
snrs = [r["snr"]  for r in records]
stds = [r["std"]  for r in records]
mus  = [r["mean"] for r in records]
skew = [r["skew"] for r in records]

ax2_twin = ax2.twinx()

l1, = ax2.plot(snrs, stds, "o-", color="#d62728", linewidth=2, label="Residual STD")
l2, = ax2.plot(snrs, mus,  "s--", color="#1f77b4", linewidth=2, label="Residual Mean")
l3, = ax2_twin.plot(snrs, skew, "^:", color="#2ca02c", linewidth=2, label="Skewness")

ax2.set_xlabel("SNR", fontsize=11)
ax2.set_ylabel("STD / Mean (S/m)", fontsize=11)
ax2_twin.set_ylabel("Skewness", fontsize=11, color="#2ca02c")
ax2_twin.tick_params(axis="y", labelcolor="#2ca02c")
ax2.set_title("Summary Statistics vs SNR", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.4)

lines  = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize=9, loc="lower right")

# SNR x 轴用实际值（非均匀间距）
ax2.set_xticks(snrs)
ax2.set_xticklabels(snrs, rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"✓ 已保存: {OUTPUT}")
plt.show()
