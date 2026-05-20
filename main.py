# -*- coding: utf-8 -*-
"""Ponon crystal_Subsystem test

副车架子系统声子晶体减振试验数据处理与出图主程序。

功能：
1. 批量读取 yuanshishuju/5.8 与 yuanshishuju/5.12 下的 .mat 工况数据；
2. 根据不同日期批次的数据结构选择输入基准和输出测点；
3. 使用 Welch PSD 计算输入归一化 PSD 比值 R_j(f)=S_yy,j(f)/S_xx(f)；
4. 多测点在线性域等权平均，得到 R_avg(f)；
5. 输出 R_avg(f) 对比曲线、频段均值 L_B 和相对 no-load 减振率 DR_B；
6. 导出 PNG 图片和 Excel 数据。
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch


# =========================
# User configuration
# =========================

FS = 48000.0
TARGET_DF = 1.0 / 20.0
PLOT_FREQ_RANGE = (70.0, 140.0)
BANDS = [
    ("80-100 Hz", 80.0, 100.0),
    ("90-110 Hz", 90.0, 110.0),
    ("70-140 Hz", 70.0, 140.0),
]

SAVE_FIGURES = True
SAVE_EXCEL = True
SHOW_FIGURES = False
FIGURE_DPI = 600

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "yuanshishuju"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
XLSX_DIR = RESULTS_DIR / "xlsx"


@dataclass(frozen=True)
class DateConfig:
    date_tag: str
    input_index: int
    input_text: str
    sensor_indices: List[int]
    sensor_names: List[str]
    point_text: str
    order: List[str]
    labels: Dict[str, str]
    colors: Dict[str, str]


DATE_CONFIGS: Dict[str, DateConfig] = {
    "5.8": DateConfig(
        date_tag="5.8",
        input_index=0,
        input_text="Data第1行",
        sensor_indices=[3, 4, 6, 7],
        sensor_names=["左前", "左后", "右前", "右后"],
        point_text="四点",
        order=["no-load", "DVA", "plan9", "plan6", "plan7", "plan8"],
        labels={
            "no-load": "no-load",
            "DVA": "DVA",
            "plan9": "8-95.5g-0.844kg",
            "plan6": "8-110.5g-0.964kg",
            "plan7": "8-125.5g-1.084kg",
            "plan8": "8-140.5g-1.204kg",
        },
        colors={
            "no-load": "#000000",
            "DVA": "#ff0000",
            "plan9": "#17becf",
            "plan6": "#8c564b",
            "plan7": "#1f77b4",
            "plan8": "#2ca02c",
        },
    ),
    "5.12": DateConfig(
        date_tag="5.12",
        input_index=1,
        input_text="Data第2行",
        sensor_indices=[4, 5, 6, 7, 8, 9],
        sensor_names=["左前", "左后", "左中", "右前", "右后", "右中"],
        point_text="六点",
        order=["no-load", "DVA", "plan1", "plan2", "plan3", "plan4", "plan5"],
        labels={
            "no-load": "no-load",
            "DVA": "DVA",
            "plan1": "10-60g-0.72kg",
            "plan2": "10-90g-1.02kg",
            "plan3": "10-105g-1.17kg",
            "plan4": "12-90g-1.224kg",
            "plan5": "12-105g-1.404kg",
        },
        colors={
            "no-load": "#000000",
            "DVA": "#ff0000",
            "plan1": "#1f77b4",
            "plan2": "#2ca02c",
            "plan3": "#9467bd",
            "plan4": "#ff7f0e",
            "plan5": "#17becf",
        },
    ),
}

BAND_COLORS = {
    "80-100 Hz": "#4c78a8",
    "90-110 Hz": "#f58518",
    "70-140 Hz": "#54a24b",
}


# =========================
# Basic utilities
# =========================


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def ensure_directories() -> None:
    for folder in [DATA_ROOT, FIGURES_DIR, XLSX_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def condition_key_from_file(path: Path) -> str:
    stem = path.stem.lower().replace("-", "_")
    if "no_load" in stem or "noload" in stem:
        return "no-load"
    if stem.startswith("dva"):
        return "DVA"
    if stem.startswith("1kg"):
        return "1kg"
    if stem.startswith("2kg"):
        return "2kg"
    match = re.match(r"^(plan\d+)", stem)
    if match:
        return match.group(1)
    return path.stem


def collect_condition_files(date_tag: str, config: DateConfig) -> Dict[str, Path]:
    data_dir = DATA_ROOT / date_tag
    if not data_dir.exists():
        print(f"[warning] 数据目录不存在，跳过: {data_dir}")
        return {}

    files: Dict[str, Path] = {}
    for mat_path in sorted(data_dir.glob("*.mat")):
        key = condition_key_from_file(mat_path)
        if key in config.order and key not in files:
            files[key] = mat_path
        elif key in files:
            print(f"[warning] 工况 {key} 有重复文件，仅使用: {files[key].name}")
    return files


# =========================
# MAT reading
# =========================


def load_mat_data(path: Path) -> np.ndarray:
    """Load shdf.Data from MATLAB .mat file.

    First try scipy.io.loadmat for standard MATLAB files. If that fails,
    try a simple HDF5/v7.3 reader.
    """
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "shdf" not in mat:
            raise ValueError("缺少 shdf 结构体")
        shdf = mat["shdf"]
        if not hasattr(shdf, "Data"):
            raise ValueError("shdf 中缺少 Data 字段")
        data = np.asarray(shdf.Data, dtype=float)
    except NotImplementedError:
        data = load_mat_data_hdf5(path)

    if data.ndim != 2:
        raise ValueError(f"Data 维度异常: {data.shape}")
    return data


def load_mat_data_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "shdf" not in h5:
            raise ValueError("HDF5 文件中缺少 shdf")
        shdf = h5["shdf"]
        if "Data" not in shdf:
            raise ValueError("HDF5 shdf 中缺少 Data")
        data = np.array(shdf["Data"], dtype=float)

    # MATLAB v7.3 often stores arrays transposed relative to scipy.loadmat.
    # This project expects Data as M x N; if rows are much larger than columns,
    # transpose as a conservative correction.
    if data.ndim == 2 and data.shape[0] > data.shape[1]:
        data = data.T
    return data


# =========================
# PSD and metrics
# =========================


def compute_psd(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float).ravel()
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if signal.size < 8:
        raise ValueError("有效数据点过少，无法计算 PSD")

    nfft_target = int(round(FS / TARGET_DF))
    nperseg = signal.size
    nfft = max(nfft_target, nperseg)
    freq, psd = welch(
        signal,
        fs=FS,
        window="hann",
        nperseg=nperseg,
        noverlap=0,
        nfft=nfft,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return freq, psd


def compute_ravg_curve(mat_path: Path, config: DateConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data = load_mat_data(mat_path)
    needed_index = max([config.input_index] + config.sensor_indices)
    if needed_index >= data.shape[0]:
        raise ValueError(f"Data 行数不足: 当前 {data.shape[0]} 行，需要至少 {needed_index + 1} 行")

    freq, psd_in = compute_psd(data[config.input_index, :])
    tiny = np.finfo(float).tiny
    ratio_curves = []
    sensor_ratio_map: Dict[str, np.ndarray] = {}

    for sensor_index, sensor_name in zip(config.sensor_indices, config.sensor_names):
        freq_out, psd_out = compute_psd(data[sensor_index, :])
        if freq_out.shape != freq.shape or not np.allclose(freq_out, freq):
            raise ValueError("输入与输出 PSD 频率轴不一致")
        ratio = psd_out / np.maximum(psd_in, tiny)
        ratio_curves.append(ratio)
        sensor_ratio_map[sensor_name] = ratio

    r_avg = np.mean(np.vstack(ratio_curves), axis=0)
    return freq, r_avg, sensor_ratio_map


def band_metric(freq: np.ndarray, r_avg: np.ndarray, f_low: float, f_high: float) -> Tuple[float, float]:
    mask = (freq >= f_low) & (freq <= f_high)
    if mask.sum() < 2:
        raise ValueError(f"{f_low:.0f}-{f_high:.0f} Hz 频段内频率点过少")

    valid = r_avg[mask]
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size == 0:
        raise ValueError("频段内没有有效 PSD 比值")

    r_band = float(np.mean(valid))
    l_band = -10.0 * np.log10(max(r_band, np.finfo(float).tiny))
    return l_band, r_band


def compute_date_results(date_tag: str, config: DateConfig):
    files = collect_condition_files(date_tag, config)
    if not files:
        return None

    missing = [key for key in config.order if key not in files]
    if missing:
        print(f"[warning] {date_tag} 缺少工况，跳过: {', '.join(missing)}")

    condition_keys = [key for key in config.order if key in files]
    if "no-load" not in condition_keys:
        print(f"[warning] {date_tag} 缺少 no-load，无法计算相对减振率")
        return None

    freq_ref = None
    ravg_curves: Dict[str, np.ndarray] = {}
    band_l_values = {band_name: [] for band_name, _, _ in BANDS}
    band_r_values = {band_name: [] for band_name, _, _ in BANDS}

    for key in condition_keys:
        try:
            freq, r_avg, _sensor_ratio_map = compute_ravg_curve(files[key], config)
        except Exception as exc:
            print(f"[warning] {date_tag} {files[key].name} 处理失败: {exc}")
            continue

        if freq_ref is None:
            freq_ref = freq
        elif freq.shape != freq_ref.shape or not np.allclose(freq, freq_ref):
            print(f"[warning] {date_tag} {key} 频率轴不一致，跳过")
            continue

        ravg_curves[key] = r_avg
        for band_name, f_low, f_high in BANDS:
            l_band, r_band = band_metric(freq, r_avg, f_low, f_high)
            band_l_values[band_name].append(l_band)
            band_r_values[band_name].append(r_band)

        print(f"{date_tag} {key:8s} " + ", ".join(f"{b}: L={band_l_values[b][-1]:.2f} dB" for b, _, _ in BANDS))

    condition_keys = [key for key in condition_keys if key in ravg_curves]
    if not condition_keys:
        return None

    baseline_index = condition_keys.index("no-load")
    band_dr_values = {band_name: [] for band_name, _, _ in BANDS}
    band_delta_values = {band_name: [] for band_name, _, _ in BANDS}

    for band_name, _, _ in BANDS:
        baseline_l = band_l_values[band_name][baseline_index]
        for l_band in band_l_values[band_name]:
            delta_l = l_band - baseline_l
            dr = (1.0 - 10.0 ** (-delta_l / 10.0)) * 100.0
            band_delta_values[band_name].append(delta_l)
            band_dr_values[band_name].append(dr)

    return {
        "date_tag": date_tag,
        "config": config,
        "freq": freq_ref,
        "condition_keys": condition_keys,
        "ravg_curves": ravg_curves,
        "band_l_values": band_l_values,
        "band_r_values": band_r_values,
        "band_delta_values": band_delta_values,
        "band_dr_values": band_dr_values,
    }


# =========================
# Plotting and export
# =========================


def condition_labels(config: DateConfig, condition_keys: Iterable[str]) -> List[str]:
    return [config.labels.get(key, key) for key in condition_keys]


def plot_ravg_curves(result: dict) -> Path:
    date_tag = result["date_tag"]
    config: DateConfig = result["config"]
    freq = result["freq"]
    condition_keys = result["condition_keys"]
    ravg_curves = result["ravg_curves"]

    mask = (freq >= PLOT_FREQ_RANGE[0]) & (freq <= PLOT_FREQ_RANGE[1])
    fig, ax = plt.subplots(figsize=(12.0, 6.2))

    for key in condition_keys:
        r_avg = ravg_curves[key]
        r_db = 10.0 * np.log10(np.maximum(r_avg, np.finfo(float).tiny))
        ax.plot(
            freq[mask],
            r_db[mask],
            label=config.labels.get(key, key),
            color=config.colors.get(key),
            linewidth=1.6,
        )

    ax.set_title(f"{date_tag}数据：{config.point_text}平均输入归一化PSD比值")
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("10log10(R_avg) / dB")
    ax.set_xlim(PLOT_FREQ_RANGE)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    out_dir = FIGURES_DIR / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_tag}_平均输入归一化PSD比值曲线.png"
    if SAVE_FIGURES:
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return out_path


def grouped_bar(ax, x, values_by_band, width, ylabel, title, labels, annotate_fmt, include_baseline=True):
    offsets = np.linspace(-width, width, len(BANDS))
    for offset, (band_name, _, _) in zip(offsets, BANDS):
        values = np.asarray(values_by_band[band_name], dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.9,
            label=band_name,
            color=BAND_COLORS[band_name],
            edgecolor="black",
            linewidth=0.7,
        )
        span = max(float(np.nanmax(values) - np.nanmin(values)), 1.0)
        pad = max(span * 0.02, 0.12)
        for idx, (bar, value) in enumerate(zip(bars, values)):
            if not include_baseline and idx == 0:
                continue
            va = "bottom" if value >= 0 else "top"
            y_text = value + pad if value >= 0 else value - pad
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                annotate_fmt.format(value),
                ha="center",
                va=va,
                fontsize=8,
                rotation=90,
                color="black",
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True, ncol=3)


def plot_band_metrics(result: dict) -> Path:
    date_tag = result["date_tag"]
    config: DateConfig = result["config"]
    condition_keys = result["condition_keys"]
    labels = condition_labels(config, condition_keys)
    x = np.arange(len(condition_keys), dtype=float)
    width = 0.24

    fig, axes = plt.subplots(2, 1, figsize=(17.5, 11.0), sharex=False)
    grouped_bar(
        axes[0],
        x,
        result["band_l_values"],
        width,
        ylabel="频段均值 $L_B$ / dB",
        title=f"{date_tag}数据：按输入归一化PSD公式计算的{config.point_text}平均频段均值",
        labels=labels,
        annotate_fmt="{:.1f}",
        include_baseline=True,
    )
    grouped_bar(
        axes[1],
        x,
        result["band_dr_values"],
        width,
        ylabel="相对 no-load 减振率 / %",
        title=f"{date_tag}数据：相对no-load的{config.point_text}平均减振率",
        labels=labels,
        annotate_fmt="{:.1f}%",
        include_baseline=False,
    )
    axes[1].axhline(0, color="black", linewidth=1.0)

    formula = (
        r"$R_j(f)=S_{yy,j}(f)/S_{xx}(f)$；"
        r"$R_{avg}(f)=\sum w_jR_j(f)/\sum w_j$；"
        r"$L_B=-10\log_{10}(\frac{1}{N_B}\sum_{f_i\in B}R_{avg}(f_i))$；"
        r"$DR_B=[1-10^{-\Delta L_B/10}]\times100\%$。"
    )
    fig.text(
        0.5,
        0.025,
        formula,
        ha="center",
        va="bottom",
        fontsize=11,
        color="black",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )
    fig.suptitle(f"{date_tag}数据：输入基准 {config.input_text}，{config.point_text}等权平均", fontsize=18)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.15, hspace=0.42)

    out_dir = FIGURES_DIR / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_tag}_输入归一化PSD公式_频段均值与减振率.png"
    if SAVE_FIGURES:
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return out_path


def export_excel(result: dict) -> Path:
    date_tag = result["date_tag"]
    config: DateConfig = result["config"]
    freq = result["freq"]
    condition_keys = result["condition_keys"]
    ravg_curves = result["ravg_curves"]

    xlsx_dir = XLSX_DIR / date_tag
    xlsx_dir.mkdir(parents=True, exist_ok=True)
    out_path = xlsx_dir / f"{date_tag}_输入归一化PSD比值与频段指标.xlsx"

    curve_data = {"Frequency_Hz": freq}
    for key in condition_keys:
        label = config.labels.get(key, key)
        r_avg = ravg_curves[key]
        curve_data[f"{label}_Ravg"] = r_avg
        curve_data[f"{label}_10log10_Ravg_dB"] = 10.0 * np.log10(np.maximum(r_avg, np.finfo(float).tiny))

    band_rows = []
    for idx, key in enumerate(condition_keys):
        label = config.labels.get(key, key)
        row = {"Condition": key, "Label": label}
        for band_name, _, _ in BANDS:
            row[f"{band_name}_L_B_dB"] = result["band_l_values"][band_name][idx]
            row[f"{band_name}_Delta_L_B_dB"] = result["band_delta_values"][band_name][idx]
            row[f"{band_name}_DR_percent"] = result["band_dr_values"][band_name][idx]
        band_rows.append(row)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(curve_data).to_excel(writer, sheet_name="Ravg_curves", index=False)
        pd.DataFrame(band_rows).to_excel(writer, sheet_name="Band_metrics", index=False)

    return out_path


# =========================
# Main
# =========================


def process_date(date_tag: str, config: DateConfig) -> None:
    print(f"\n===== Processing {date_tag} =====")
    result = compute_date_results(date_tag, config)
    if result is None:
        return

    curve_path = plot_ravg_curves(result)
    metric_path = plot_band_metrics(result)
    print(f"[saved] {curve_path}")
    print(f"[saved] {metric_path}")

    if SAVE_EXCEL:
        excel_path = export_excel(result)
        print(f"[saved] {excel_path}")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    setup_plot_style()
    ensure_directories()

    for date_tag, config in DATE_CONFIGS.items():
        process_date(date_tag, config)


if __name__ == "__main__":
    main()
