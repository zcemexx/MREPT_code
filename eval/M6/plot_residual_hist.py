"""
残差分布直方图 — SNR 10 / 50 / 150 并列 subplot
输出: residual_hist_compare.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── 路径 ──────────────────────────────────────────────────────────────────────
CASES_DIR = "/home/linux1917366562/MREPT_code/eval/M6/mae_compare_nib/cases"
OUTPUT    = "/home/linux1917366562/MREPT_code/eval/M6/residual_hist_compare.png"

# ── 读取数据 ──────────────────────────────────────────────────────────────────
SELECTED_SNRS = [10, 50, 150]
records = []
for snr in SELECTED_SNRS:
    case = f"M6_SNR{snr:03d}"
    path = os.path.join(CASES_DIR, case, "case_report.json")
    with open(path) as f:
        d = json.load(f)
    rd = d["residual_distribution"]
    records.append({
        "snr"    : snr,
        "edges"  : np.array(rd["Histogram_Edges"]),
        "counts" : np.array(rd["Histogram_Counts"], dtype=float),
        "mean"   : rd["Residual_Mean"],
        "std"    : rd["Residual_STD"],
        "skew"   : rd["Residual_Skewness"],
    })

# ── Bin 中心 & 归一化为概率密度 ───────────────────────────────────────────────
for r in records:
    edges        = r["edges"]
    centers      = 0.5 * (edges[:-1] + edges[1:])
    widths       = np.diff(edges)
    r["centers"] = centers
    r["pdf"]     = r["counts"] / (r["counts"].sum() * widths)

# ── 固定 3 色 ────────────────────────────────────────────────────────────────
colors = ["#1f77b4", "#ff7f0e", "#d62728"]   # SNR10=蓝 SNR50=橙 SNR150=红

# ── 并列 subplot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
x_fit = np.linspace(-7, 7, 400)

for ax, r, color in zip(axes, records, colors):
    # 柱状图（bar）
    ax.bar(r["centers"], r["pdf"],
           width=np.diff(r["edges"]),
           color=color, alpha=0.55, edgecolor=color, linewidth=0.6,
           label="Histogram")

    # 拟合正态曲线
    y_fit = norm.pdf(x_fit, r["mean"], r["std"])
    ax.plot(x_fit, y_fit, "-", color=color, linewidth=2.0,
            label=f"Normal fit\nμ={r['mean']:+.3f}\nσ={r['std']:.3f}\nskew={r['skew']:+.3f}")

    # 零线
    ax.axvline(0, color="gray", linewidth=0.9, linestyle="--", alpha=0.7)

    ax.set_title(f"SNR {r['snr']}", fontsize=13, fontweight="bold", color=color)
    ax.set_xlabel("Residual (S/m)", fontsize=11)
    ax.set_xlim(-7, 7)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, framealpha=0.75, loc="upper left")

axes[0].set_ylabel("Probability Density", fontsize=11)

fig.suptitle("Residual Distribution at Selected SNR Levels", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"✓ 已保存: {OUTPUT}")
plt.show()
