from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


DEFAULT_METRIC_ROOT = Path("/Users/apple/Documents/mresult/metric")
DEFAULT_OUTPUT_DIR = Path("/Users/apple/Documents/mresult")

METHOD_ORDER = ["Fixed", "CNN", "Oracle"]
REGION_ORDER = ["Global", "GM", "WM", "CSF"]
SNR_ORDER = [10, 20, 30, 40, 50, 75, 100, 150]
DISPLAY_SNRS = [10, 50, 150]
FIG1_SNR = 50

FIG1_METHOD_PALETTE = {
    "Fixed": "#4d4d4d",
    "CNN": "#c47a00",
    "Oracle": "#7b5ea7",
}
ACC_PALETTE = {
    "Acc1": "#e41a1c",
    "Acc3": "#377eb8",
    "Acc5": "#4daf4a",
}
ACC_LABELS = {
    "Acc1": r"$\pm$1",
    "Acc3": r"$\pm$3",
    "Acc5": r"$\pm$5",
}

EXPECTED_VIZ_KEYS = [
    "Difference_Maps/SNR_010/Diff_Cond",
    "Difference_Maps/SNR_010/Diff_Radius",
    "Difference_Maps/SNR_050/Diff_Cond",
    "Difference_Maps/SNR_050/Diff_Radius",
    "Difference_Maps/SNR_150/Diff_Cond",
    "Difference_Maps/SNR_150/Diff_Radius",
    "Simulation/GT_Cond",
    "Simulation/SNR_010/CNN_Cond",
    "Simulation/SNR_010/CNN_Pred_Radius",
    "Simulation/SNR_010/Fixed_Cond",
    "Simulation/SNR_010/Oracle_Cond",
    "Simulation/SNR_050/CNN_Cond",
    "Simulation/SNR_050/CNN_Pred_Radius",
    "Simulation/SNR_050/Fixed_Cond",
    "Simulation/SNR_050/Oracle_Cond",
    "Simulation/SNR_150/CNN_Cond",
    "Simulation/SNR_150/CNN_Pred_Radius",
    "Simulation/SNR_150/Fixed_Cond",
    "Simulation/SNR_150/Oracle_Cond",
]


def configure_style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_quantitative_data(metric_root: Path = DEFAULT_METRIC_ROOT) -> pd.DataFrame:
    csv_path = _require_file(metric_root / "quantitative_results.csv")
    df = pd.read_csv(csv_path)
    if len(df) != 1632:
        raise ValueError(f"Expected 1632 quantitative rows, found {len(df)} at {csv_path}")
    rename_map = {
        "Case_ID": "Case",
        "Acc_1": "Acc1",
        "Acc_3": "Acc3",
        "Acc_5": "Acc5",
    }
    df = df.rename(columns=rename_map)
    for column in ["SNR", "MAE", "RMSE", "SSIM", "Acc1", "Acc3", "Acc5"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Case"] = df["Case"].astype(str)
    df["Method"] = pd.Categorical(df["Method"], categories=METHOD_ORDER, ordered=True)
    df["Region"] = pd.Categorical(df["Region"], categories=REGION_ORDER, ordered=True)
    return df.sort_values(["Case", "SNR", "Method", "Region"]).reset_index(drop=True)


def load_visualization_arrays(metric_root: Path = DEFAULT_METRIC_ROOT) -> dict[str, np.ndarray]:
    npz_path = _require_file(metric_root / "visualization_slices.npz")
    arrays = np.load(npz_path)
    keys = sorted(arrays.files)
    if keys != sorted(EXPECTED_VIZ_KEYS):
        missing = sorted(set(EXPECTED_VIZ_KEYS) - set(keys))
        extra = sorted(set(keys) - set(EXPECTED_VIZ_KEYS))
        raise ValueError(f"Unexpected visualization keys. Missing={missing}, extra={extra}")
    loaded = {key: np.asarray(arrays[key], dtype=np.float32) for key in arrays.files}
    shapes = {array.shape for array in loaded.values()}
    if len(shapes) != 1:
        raise ValueError(f"Expected one common array shape, found {shapes}")
    return loaded


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)
    plt.close(fig)
    return saved_paths


def _finite_values(chunks: Iterable[np.ndarray]) -> np.ndarray:
    finite_chunks = []
    for chunk in chunks:
        array = np.asarray(chunk)
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_chunks.append(finite)
    if not finite_chunks:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    return np.concatenate(finite_chunks).astype(np.float32, copy=False)


def percentile_clim(chunks: Iterable[np.ndarray], low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    merged = _finite_values(chunks)
    lo = float(np.percentile(merged, low))
    hi = float(np.percentile(merged, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1e-6
    return lo, hi


def nonnegative_clim(chunks: Iterable[np.ndarray], high: float = 99.0) -> tuple[float, float]:
    merged = _finite_values(chunks)
    hi = float(np.percentile(merged, high))
    if not np.isfinite(hi) or hi <= 0:
        hi = 1e-6
    return 0.0, hi


def _format_snr_key(snr: int) -> str:
    return f"SNR_{int(snr):03d}"


def _sim_key(snr: int, name: str) -> str:
    return f"Simulation/{_format_snr_key(snr)}/{name}"


def _diff_key(snr: int, name: str) -> str:
    return f"Difference_Maps/{_format_snr_key(snr)}/{name}"


def make_fig1(df: pd.DataFrame) -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.38, wspace=0.28)

    df_global = df[df["Region"] == "Global"].copy()

    ax1 = fig.add_subplot(grid[0, 0])
    sns.lineplot(
        data=df_global,
        x="SNR",
        y="MAE",
        hue="Method",
        hue_order=METHOD_ORDER,
        palette=FIG1_METHOD_PALETTE,
        marker="o",
        errorbar="sd",
        ax=ax1,
    )
    ax1.set_title("A. Global MAE vs. SNR", fontweight="bold", loc="left")
    ax1.set_xlabel("SNR")
    ax1.set_ylabel("Mean Absolute Error (S/m)")
    legend = ax1.get_legend()
    if legend is not None:
        legend.remove()

    ax2 = fig.add_subplot(grid[0, 1])
    sns.lineplot(
        data=df_global,
        x="SNR",
        y="SSIM",
        hue="Method",
        hue_order=METHOD_ORDER,
        palette=FIG1_METHOD_PALETTE,
        marker="s",
        errorbar="sd",
        ax=ax2,
    )
    ax2.set_title("B. Global SSIM vs. SNR", fontweight="bold", loc="left")
    ax2.set_xlabel("SNR")
    ax2.set_ylabel("SSIM")
    legend = ax2.get_legend()
    if legend is not None:
        legend.remove()

    ax3 = fig.add_subplot(grid[0, 2])
    df_cnn = df_global[df_global["Method"] == "CNN"].copy()
    df_acc = df_cnn.melt(
        id_vars=["SNR", "Case"],
        value_vars=["Acc1", "Acc3", "Acc5"],
        var_name="Tolerance",
        value_name="Accuracy",
    )
    for metric_name in ["Acc1", "Acc3", "Acc5"]:
        sns.lineplot(
            data=df_acc[df_acc["Tolerance"] == metric_name],
            x="SNR",
            y="Accuracy",
            label=ACC_LABELS[metric_name],
            color=ACC_PALETTE[metric_name],
            marker="o",
            errorbar="sd",
            ax=ax3,
        )
    ax3.set_title("C. CNN Prediction Accuracy vs. SNR", fontweight="bold", loc="left")
    ax3.set_xlabel("SNR")
    ax3.set_ylabel("Accuracy Ratio")
    ax3.set_ylim(0.0, 1.02)
    ax3.legend(title="Tolerance", loc="lower right", frameon=True)

    ax4 = fig.add_subplot(grid[1, :])
    df_snr = df[df["SNR"] == FIG1_SNR].copy()
    sns.boxplot(
        data=df_snr,
        x="Region",
        y="MAE",
        hue="Method",
        hue_order=METHOD_ORDER,
        order=REGION_ORDER,
        palette=FIG1_METHOD_PALETTE,
        width=0.68,
        fliersize=0,
        boxprops={"alpha": 0.8},
        ax=ax4,
    )
    sns.stripplot(
        data=df_snr,
        x="Region",
        y="MAE",
        hue="Method",
        hue_order=METHOD_ORDER,
        order=REGION_ORDER,
        dodge=True,
        palette={method: "#202020" for method in METHOD_ORDER},
        alpha=0.45,
        size=3.5,
        jitter=0.08,
        linewidth=0,
        ax=ax4,
    )
    ax4.set_title(
        f"D. Region-Specific Error Distribution at SNR={FIG1_SNR}",
        fontweight="bold",
        loc="left",
    )
    ax4.set_xlabel("Tissue Region")
    ax4.set_ylabel("Mean Absolute Error (S/m)")
    handles, labels = ax4.get_legend_handles_labels()
    dedup: list[tuple[object, str]] = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label in METHOD_ORDER and label not in seen:
            dedup.append((handle, label))
            seen.add(label)
    if dedup:
        ax4.legend(
            [item[0] for item in dedup],
            [item[1] for item in dedup],
            title="Reconstruction Method",
            loc="upper right",
            frameon=False,
        )

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(SNR_ORDER)
        ax.grid(alpha=0.2, linewidth=0.6)
    ax4.grid(axis="y", alpha=0.2, linewidth=0.6)
    return fig


def make_fig2(arrays: dict[str, np.ndarray]) -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(13.5, 15))
    fig.patch.set_facecolor("black")
    grid = fig.add_gridspec(6, 5, left=0.06, right=0.97, top=0.95, bottom=0.12, wspace=0.0, hspace=0.0)

    gt = arrays["Simulation/GT_Cond"]
    recon_arrays = [gt]
    error_arrays = []
    radius_arrays = []
    for snr in DISPLAY_SNRS:
        for method in METHOD_ORDER:
            recon = arrays[_sim_key(snr, f"{method}_Cond")]
            recon_arrays.append(recon)
            error_arrays.append(np.abs(recon - gt))
        predicted_radius = arrays[_sim_key(snr, "CNN_Pred_Radius")]
        final_label = predicted_radius - arrays[_diff_key(snr, "Diff_Radius")]
        radius_arrays.extend([predicted_radius, final_label])
    recon_clim = percentile_clim(recon_arrays, low=1.0, high=99.0)
    error_clim = nonnegative_clim(error_arrays, high=99.0)
    radius_clim = percentile_clim(radius_arrays, low=1.0, high=99.0)

    for snr_idx, snr in enumerate(DISPLAY_SNRS):
        recon_row = snr_idx * 2
        detail_row = recon_row + 1
        predicted_radius = arrays[_sim_key(snr, "CNN_Pred_Radius")]
        final_label = predicted_radius - arrays[_diff_key(snr, "Diff_Radius")]
        for col_idx, title in enumerate(["Fixed", "CNN", "Oracle", "GT", "Radius Maps"]):
            ax_recon = fig.add_subplot(grid[recon_row, col_idx])
            ax_detail = fig.add_subplot(grid[detail_row, col_idx])
            ax_recon.set_facecolor("black")
            ax_detail.set_facecolor("black")
            ax_recon.axis("off")
            ax_detail.axis("off")

            if col_idx == 0:
                ax_recon.text(
                    -0.13,
                    0.5,
                    f"SNR {snr}\nconductivity map",
                    transform=ax_recon.transAxes,
                    va="center",
                    ha="right",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )
                ax_detail.text(
                    -0.13,
                    0.5,
                    f"SNR {snr}\nMAE map / radius",
                    transform=ax_detail.transAxes,
                    va="center",
                    ha="right",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )
            if recon_row == 0:
                ax_recon.set_title(title, fontsize=13, fontweight="bold", pad=10, color="white")

            if col_idx < 3:
                method = METHOD_ORDER[col_idx]
                recon = arrays[_sim_key(snr, f"{method}_Cond")]
                error = np.abs(recon - gt)
                ax_recon.imshow(recon, cmap="jet", vmin=recon_clim[0], vmax=recon_clim[1])
                ax_detail.imshow(error, cmap="hot", vmin=error_clim[0], vmax=error_clim[1])
            elif col_idx == 3:
                ax_recon.imshow(gt, cmap="jet", vmin=recon_clim[0], vmax=recon_clim[1])
                ax_detail.imshow(gt, cmap="jet", vmin=recon_clim[0], vmax=recon_clim[1])
            else:
                ax_recon.imshow(final_label, cmap="viridis", vmin=radius_clim[0], vmax=radius_clim[1])
                ax_detail.imshow(predicted_radius, cmap="viridis", vmin=radius_clim[0], vmax=radius_clim[1])

    _add_horizontal_colorbar(fig, "jet", recon_clim, [0.08, 0.06, 0.22, 0.015], "Conductivity (S/m)", text_color="white", facecolor="black")
    _add_horizontal_colorbar(fig, "hot", error_clim, [0.39, 0.06, 0.22, 0.015], "MAE map (S/m)", text_color="white", facecolor="black")
    _add_horizontal_colorbar(fig, "viridis", radius_clim, [0.70, 0.06, 0.22, 0.015], "Radius Label / Kernel", text_color="white", facecolor="black")
    return fig


def make_fig3(arrays: dict[str, np.ndarray]) -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(10, 11))
    grid = fig.add_gridspec(4, 4, left=0.08, right=0.95, top=0.95, bottom=0.10, wspace=0.02, hspace=0.02)

    gt = arrays["Simulation/GT_Cond"]
    recon_arrays = [gt]
    error_arrays = []
    for snr in DISPLAY_SNRS:
        for method in METHOD_ORDER:
            recon = arrays[_sim_key(snr, f"{method}_Cond")]
            recon_arrays.append(recon)
            if snr == 50:
                error_arrays.append(np.abs(recon - gt))
    recon_clim = percentile_clim(recon_arrays, low=1.0, high=99.0)
    error_clim = nonnegative_clim(error_arrays, high=99.0)

    row_configs = [
        (10, False, "SNR 10\nRecon"),
        (50, False, "SNR 50\nRecon"),
        (50, True, "SNR 50\nError"),
        (150, False, "SNR 150\nRecon"),
    ]
    column_titles = ["Fixed", "CNN", "Oracle", "GT"]

    for row_idx, (snr, is_error, row_label) in enumerate(row_configs):
        for col_idx, column_title in enumerate(column_titles):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(column_title, fontsize=13, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.text(
                    -0.12,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=11,
                    fontweight="bold",
                )

            if is_error and col_idx == 3:
                continue

            if col_idx < 3:
                method = METHOD_ORDER[col_idx]
                recon = arrays[_sim_key(snr, f"{method}_Cond")]
                if is_error:
                    image = np.abs(recon - gt)
                    ax.imshow(image, cmap="turbo", vmin=error_clim[0], vmax=error_clim[1])
                else:
                    ax.imshow(recon, cmap="magma", vmin=recon_clim[0], vmax=recon_clim[1])
            else:
                ax.imshow(gt, cmap="magma", vmin=recon_clim[0], vmax=recon_clim[1])

    _add_horizontal_colorbar(fig, "magma", recon_clim, [0.16, 0.045, 0.28, 0.017], "Conductivity (S/m)")
    _add_horizontal_colorbar(fig, "turbo", error_clim, [0.58, 0.045, 0.28, 0.017], "|Error| (S/m)")
    return fig


def make_radius_mae_comparison_figure(arrays: dict[str, np.ndarray]) -> plt.Figure:
    configure_style()
    fig = plt.figure(figsize=(9.5, 13))
    grid = fig.add_gridspec(3, 2, left=0.08, right=0.93, top=0.95, bottom=0.10, wspace=0.04, hspace=0.06)

    radius_diff_arrays = []
    mae_arrays = []
    for snr in DISPLAY_SNRS:
        radius_diff_arrays.append(-(arrays[_diff_key(snr, "Diff_Radius")]))
        mae_arrays.append(np.abs(arrays[_diff_key(snr, "Diff_Cond")]))
    radius_abs_max = max(float(np.nanmax(np.abs(arr))) for arr in radius_diff_arrays)
    radius_clim = (-radius_abs_max, radius_abs_max if radius_abs_max > 0 else 1.0)
    mae_clim = nonnegative_clim(mae_arrays, high=99.0)

    for row_idx, snr in enumerate(DISPLAY_SNRS):
        ax_radius = fig.add_subplot(grid[row_idx, 0])
        ax_mae = fig.add_subplot(grid[row_idx, 1])
        ax_radius.axis("off")
        ax_mae.axis("off")

        if row_idx == 0:
            ax_radius.set_title("Oracle - Pred Radius", fontsize=13, fontweight="bold", pad=10)
            ax_mae.set_title("MAE Map", fontsize=13, fontweight="bold", pad=10)

        ax_radius.text(
            -0.10,
            0.5,
            f"SNR {snr}",
            transform=ax_radius.transAxes,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

        radius_diff = -(arrays[_diff_key(snr, "Diff_Radius")])
        mae_map = np.abs(arrays[_diff_key(snr, "Diff_Cond")])
        ax_radius.imshow(radius_diff, cmap="coolwarm", vmin=radius_clim[0], vmax=radius_clim[1])
        ax_mae.imshow(mae_map, cmap="hot", vmin=mae_clim[0], vmax=mae_clim[1])

    _add_horizontal_colorbar(fig, "coolwarm", radius_clim, [0.13, 0.05, 0.28, 0.017], "Oracle - Pred Radius")
    _add_horizontal_colorbar(fig, "hot", mae_clim, [0.57, 0.05, 0.28, 0.017], "MAE map (S/m)")
    return fig


def _add_horizontal_colorbar(
    fig: plt.Figure,
    cmap: str,
    clim: tuple[float, float],
    rect: Sequence[float],
    label: str,
    text_color: str = "black",
    facecolor: str | None = None,
) -> None:
    cax = fig.add_axes(rect)
    if facecolor is not None:
        cax.set_facecolor(facecolor)
    sm = ScalarMappable(norm=Normalize(vmin=clim[0], vmax=clim[1]), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=10, color=text_color)
    cbar.ax.tick_params(labelsize=9, color=text_color, labelcolor=text_color)
    cbar.outline.set_edgecolor(text_color)
