"""
生成 SNR–MAE 曲线 和 SNR–Acc 曲线
数据来源: eval/M6/mae_compare_nib/cases/M6_SNR*/case_report.json
"""

import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 读取数据 ──────────────────────────────────────────────────────────────────
CASES_DIR = os.path.join(os.path.dirname(__file__), "mae_compare_nib", "cases")

records = []
for case_name in sorted(os.listdir(CASES_DIR)):
    json_path = os.path.join(CASES_DIR, case_name, "case_report.json")
    if not os.path.isfile(json_path):
        continue
    m = re.search(r"SNR(\d+)", case_name)
    if not m:
        continue
    snr = int(m.group(1))
    with open(json_path) as f:
        data = json.load(f)
    tm = data.get("tissue_metrics", {})
    records.append({
        "snr": snr,
        "Global_MAE":  tm.get("Global", {}).get("MAE"),
        "WM_MAE":      tm.get("WM",     {}).get("MAE"),
        "GM_MAE":      tm.get("GM",     {}).get("MAE"),
        "CSF_MAE":     tm.get("CSF",    {}).get("MAE"),
        "Global_Acc1": tm.get("Global", {}).get("Acc_1"),
        "Global_Acc3": tm.get("Global", {}).get("Acc_3"),
        "Global_Acc5": tm.get("Global", {}).get("Acc_5"),
    })

records.sort(key=lambda r: r["snr"])
snrs = np.array([r["snr"] for r in records])

def col(key):
    return np.array([r[key] for r in records], dtype=float)

# ── 风格 ──────────────────────────────────────────────────────────────────────
COLORS = {
    "Global": "#2c7bb6",
    "WM":     "#d7191c",
    "GM":     "#1a9641",
    "CSF":    "#fdae61",
    "Acc1":   "#2c7bb6",
    "Acc3":   "#d7191c",
    "Acc5":   "#1a9641",
}
MARKERS = {"Global": "o", "WM": "s", "GM": "^", "CSF": "D",
           "Acc1": "o", "Acc3": "s", "Acc5": "^"}

def apply_style(ax):
    ax.set_xscale("log")
    ax.set_xticks(snrs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("SNR", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

# ── 图1: SNR–MAE ──────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
for tissue in ["Global", "WM", "GM", "CSF"]:
    ax1.plot(snrs, col(f"{tissue}_MAE"),
             color=COLORS[tissue], marker=MARKERS[tissue],
             linewidth=1.8, markersize=6, label=tissue)
ax1.set_ylabel("MAE (S/m)", fontsize=12)
ax1.set_title("SNR vs MAE (Global / WM / GM / CSF)", fontsize=13)
apply_style(ax1)
fig1.tight_layout()
out1 = os.path.join(os.path.dirname(__file__), "snr_mae_curve.png")
fig1.savefig(out1, dpi=150)
print(f"Saved: {out1}")

# ── 图2: SNR–Acc ──────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
for label, key, c, mk in [
    ("Acc@1 S/m",  "Global_Acc1", COLORS["Acc1"], MARKERS["Acc1"]),
    ("Acc@3 S/m",  "Global_Acc3", COLORS["Acc3"], MARKERS["Acc3"]),
    ("Acc@5 S/m",  "Global_Acc5", COLORS["Acc5"], MARKERS["Acc5"]),
]:
    ax2.plot(snrs, col(key),
             color=c, marker=mk, linewidth=1.8, markersize=6, label=label)
ax2.set_ylabel("Accuracy (fraction within threshold)", fontsize=12)
ax2.set_title("SNR vs Accuracy (Global, Acc@1/3/5 S/m)", fontsize=13)
ax2.set_ylim(0, 1.0)
apply_style(ax2)
fig2.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), "snr_acc_curve.png")
fig2.savefig(out2, dpi=150)
print(f"Saved: {out2}")
