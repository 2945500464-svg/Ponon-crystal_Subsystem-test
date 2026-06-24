from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES_DIR, figure_save_dpi, legend_fontsize
from .plot_style import resolve_plot_font_family
from .utils import get_condition_color, get_condition_plot_order, get_display_condition_name, normalize_condition_label, sanitize_filename

_DEFAULT_PLOT_FONT, _DEFAULT_PLOT_FONT_LIST = resolve_plot_font_family("Microsoft YaHei UI")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = _DEFAULT_PLOT_FONT_LIST
plt.rcParams["axes.unicode_minus"] = False

def resolve_condition_order(
    all_results: Dict[str, Dict[str, Any]],
    condition_order: Optional[List[str]] = None,
) -> List[str]:
    """优先使用 GUI 手动顺序，未覆盖的工况按默认规则追加。"""
    default_order = get_condition_plot_order(all_results)
    if not condition_order:
        return default_order

    selected_order: List[str] = []
    seen = set()
    for name in condition_order:
        if name in all_results and name not in seen:
            selected_order.append(name)
            seen.add(name)
    selected_order.extend(name for name in default_order if name not in seen)
    return selected_order


def plot_force_acceleration_frf_comparison(
    all_results: Dict[str, Dict[str, Any]],
    freq_min: float,
    freq_max: float,
    save_flag: bool,
    figures_dir: Path,
    condition_order: Optional[List[str]] = None,
) -> Tuple[Optional[plt.Figure], Optional[Path]]:
    """绘制所有工况输入点加速度/力 FRF 对比图。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    has_data = False

    for condition_name in resolve_condition_order(all_results, condition_order):
        result = all_results[condition_name]
        if "frf_freqs" not in result or "accel_force_frf_db" not in result:
            continue

        ax.plot(
            result["frf_freqs"],
            result["accel_force_frf_db"],
            linewidth=1.4,
            label=condition_name,
            color=get_condition_color(condition_name),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None, None

    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("输入点 PSD 比值 S_aa/S_ff / dB")
    ax.set_title("输入点加速度/力 PSD 比值对比")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=legend_fontsize)
    fig.tight_layout()

    saved_path: Optional[Path] = None
    if save_flag:
        saved_path = save_figure_png(fig, "PSD比值_输入点加速度_力.png", figures_dir)

    return fig, saved_path


def save_figure_png(fig: plt.Figure, filename: str, figures_dir: Path = FIGURES_DIR) -> Path:
    """按统一分辨率保存 PNG。"""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_path = figures_dir / sanitize_filename(filename)
    fig.savefig(saved_path, dpi=figure_save_dpi, bbox_inches="tight")
    return saved_path


def plot_normalized_psd_ratio_heatmap(
    all_results: Dict[str, Dict[str, Any]],
    baseline_label: str,
    condition_order: List[str],
    title: str,
    freq_min: float,
    freq_max: float,
) -> Optional[plt.Figure]:
    """绘制相对指定基准的输入归一化 PSD 比值改善热力图，只显示不保存。"""
    canonical_to_name: Dict[str, str] = {}
    for condition_name in all_results:
        canonical_to_name.setdefault(normalize_condition_label(condition_name), condition_name)

    baseline_name = canonical_to_name.get(baseline_label, baseline_label)
    baseline_result = all_results.get(baseline_name)
    if baseline_result is None or "normalized_psd_ratio_avg" not in baseline_result:
        print(f"  warning: 缺少基准工况 {baseline_label}，归一化 PSD 比值热力图未绘制")
        return None

    baseline_freqs = baseline_result["normalized_psd_freqs"]
    baseline_ratio = np.maximum(baseline_result["normalized_psd_ratio_avg"], np.finfo(float).tiny)
    rows: List[np.ndarray] = []
    row_labels: List[str] = []

    for label in condition_order:
        condition_name = label if label in all_results else canonical_to_name.get(normalize_condition_label(label), label)
        result = all_results.get(condition_name)
        if result is None or "normalized_psd_ratio_avg" not in result:
            continue

        ratio = result["normalized_psd_ratio_avg"]
        freqs = result["normalized_psd_freqs"]
        if len(freqs) != len(baseline_freqs) or not np.allclose(freqs, baseline_freqs):
            ratio = np.interp(baseline_freqs, freqs, ratio)

        rows.append(10.0 * np.log10(np.maximum(ratio, np.finfo(float).tiny) / baseline_ratio))
        row_labels.append(get_display_condition_name(condition_name))

    if not rows:
        return None

    matrix = np.vstack(rows)
    finite_values = matrix[np.isfinite(matrix)]
    vmax = float(np.nanpercentile(np.abs(finite_values), 98)) if finite_values.size else 1.0
    vmax = max(vmax, 1.0)

    fig_height = max(3.8, 0.48 * len(row_labels) + 2.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        extent=[baseline_freqs[0], baseline_freqs[-1], len(row_labels) - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlim(freq_min, freq_max)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("方案")
    ax.set_title(title)
    ax.axvline(90, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(110, color="black", linestyle="--", linewidth=0.9)
    ax.text(100, -0.35, "目标频段 90-110 Hz", ha="center", va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"})
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("PSD比值改善量 / dB")
    fig.tight_layout()
    return fig


def plot_force_psd_comparison(
    all_results: Dict[str, Dict[str, Any]],
    freq_min: float,
    freq_max: float,
    condition_order: Optional[List[str]] = None,
) -> Optional[plt.Figure]:
    """绘制所有工况输入力 PSD 对比图，只显示不保存。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    has_data = False

    for condition_name in resolve_condition_order(all_results, condition_order):
        result = all_results[condition_name]
        if "force_psd_freqs" not in result or "force_psd" not in result:
            continue

        ax.plot(
            result["force_psd_freqs"],
            result["force_psd"],
            linewidth=1.4,
            label=condition_name,
            color=get_condition_color(condition_name),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("输入力 PSD / N²·Hz⁻¹")
    ax.set_title("输入力 PSD 对比")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=legend_fontsize)
    fig.tight_layout()
    return fig


def plot_input_accel_psd_comparison(
    all_results: Dict[str, Dict[str, Any]],
    freq_min: float,
    freq_max: float,
    condition_order: Optional[List[str]] = None,
) -> Optional[plt.Figure]:
    """绘制所有工况激振器加速度 PSD 对比图，只显示不保存。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    has_data = False

    for condition_name in resolve_condition_order(all_results, condition_order):
        result = all_results[condition_name]
        if "input_accel_psd_freqs" not in result or "input_accel_psd" not in result:
            continue

        ax.plot(
            result["input_accel_psd_freqs"],
            result["input_accel_psd"],
            linewidth=1.4,
            label=get_display_condition_name(condition_name),
            color=get_condition_color(condition_name),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("激振器加速度 PSD / (m/s²)²·Hz⁻¹")
    ax.set_title("激振器加速度 PSD 对比")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=legend_fontsize)
    fig.tight_layout()
    return fig


def plot_acceleration_fft_comparison(
    all_results: Dict[str, Dict[str, Any]],
    sensor_name: str,
    freq_min: float,
    freq_max: float,
    condition_order: Optional[List[str]] = None,
) -> Optional[plt.Figure]:
    """绘制同一加速度测点在不同工况下的 1 Hz FFT 幅值对比图。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    has_data = False

    for condition_name in resolve_condition_order(all_results, condition_order):
        result = all_results[condition_name]
        fft_names = [str(name) for name in result.get("accel_fft_names", result.get("output_names", []))]
        fft_data = result.get("accel_fft_amplitude")
        fft_freqs = result.get("accel_fft_freqs")
        if fft_data is None or fft_freqs is None or sensor_name not in fft_names:
            continue

        row_index = fft_names.index(sensor_name)
        if row_index >= fft_data.shape[0]:
            continue

        ax.plot(
            fft_freqs,
            fft_data[row_index, :],
            linewidth=1.4,
            label=get_display_condition_name(condition_name),
            color=get_condition_color(condition_name),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("加速度 FFT 幅值 / (m/s²)")
    ax.set_title(f"{sensor_name} 1 Hz加速度FFT幅值对比")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=legend_fontsize)
    fig.tight_layout()
    return fig


def plot_transfer_loss_comparison(
    all_results: Dict[str, Dict[str, Any]],
    output_index: int,
    output_name: str,
    freq_min: float,
    freq_max: float,
    save_flag: bool,
    figures_dir: Path,
    condition_order: Optional[List[str]] = None,
) -> Tuple[Optional[plt.Figure], Optional[Path]]:
    """绘制同一输出通道在不同工况下的传递损失对比图。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    has_data = False

    for condition_name in resolve_condition_order(all_results, condition_order):
        result = all_results[condition_name]
        tl_data = result["transfer_loss_db"]
        if output_index >= tl_data.shape[0]:
            continue

        ax.plot(
            result["freqs"],
            tl_data[output_index, :],
            linewidth=1.4,
            label=get_display_condition_name(condition_name),
            color=get_condition_color(condition_name),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None, None

    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("PSD 传递比 / dB")
    ax.set_title(f"{output_name} PSD 传递比对比")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=legend_fontsize)
    fig.tight_layout()

    saved_path: Optional[Path] = None
    if save_flag:
        saved_path = save_figure_png(fig, f"TL_{sanitize_filename(output_name)}.png", figures_dir)

    return fig, saved_path


def plot_weighted_transfer_rate_comparison(
    all_results: Dict[str, Dict[str, Any]],
    sensor_names: List[str],
    weights: List[float],
    freq_min: float,
    freq_max: float,
    save_flag: bool,
    figures_dir: Path,
    condition_order: Optional[List[str]] = None,
) -> Tuple[Optional[plt.Figure], Optional[Path]]:
    """绘制指定测点等权/加权平均 PSD 传递率对比图。"""
    if len(sensor_names) != len(weights):
        raise ValueError("sensor_names 和 weights 长度必须一致")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    has_data = False
    tiny = np.finfo(float).tiny
    weight_array = np.asarray(weights, dtype=float)
    weight_array = weight_array / np.sum(weight_array)

    for condition_name in resolve_condition_order(all_results, condition_order):
        result = all_results[condition_name]
        output_names = [str(name) for name in result.get("output_names", [])]
        tl_data = result["transfer_loss_db"]

        selected_rows: List[np.ndarray] = []
        selected_weights: List[float] = []
        for sensor_name, weight in zip(sensor_names, weight_array):
            if sensor_name not in output_names:
                continue
            row_index = output_names.index(sensor_name)
            if row_index >= tl_data.shape[0]:
                continue
            selected_rows.append(10.0 ** (tl_data[row_index, :] / 10.0))
            selected_weights.append(float(weight))

        if not selected_rows:
            continue

        selected_weight_array = np.asarray(selected_weights, dtype=float)
        selected_weight_array = selected_weight_array / np.sum(selected_weight_array)
        weighted_ratio = np.average(np.vstack(selected_rows), axis=0, weights=selected_weight_array)
        weighted_db = 10.0 * np.log10(np.maximum(weighted_ratio, tiny))

        ax.plot(
            result["freqs"],
            weighted_db,
            linewidth=1.6,
            label=condition_name,
            color=get_condition_color(condition_name),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None, None

    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel("加权 PSD 传递率 / dB")
    ax.set_title("六测点加权 PSD 传递率对比（左前、左后、左中、右前、右后、右中）")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", frameon=True, fontsize=legend_fontsize)
    fig.tight_layout()

    saved_path: Optional[Path] = None
    if save_flag:
        saved_path = save_figure_png(fig, "六测点加权PSD传递率_左前左后左中右前右后右中.png", figures_dir)

    return fig, saved_path


def build_average_results_between_dates(
    all_results: Dict[str, Dict[str, Any]],
    first_date: str,
    second_date: str,
) -> Dict[str, Dict[str, Any]]:
    """将两个日期的同名工况在线性传递率域平均，再转换为 dB。"""
    def pair_key(condition_name: str) -> str:
        return re.sub(r"^5\.\d+\s+", "", str(condition_name)).strip()

    first_results = {
        pair_key(condition_name): result
        for condition_name, result in all_results.items()
        if condition_name.startswith(f"{first_date} ")
    }
    second_results = {
        pair_key(condition_name): result
        for condition_name, result in all_results.items()
        if condition_name.startswith(f"{second_date} ")
    }

    average_results: Dict[str, Dict[str, Any]] = {}
    tiny = np.finfo(float).tiny

    for label in sorted(set(first_results) & set(second_results)):
        first_result = first_results[label]
        second_result = second_results[label]

        first_freqs = first_result["freqs"]
        second_freqs = second_result["freqs"]
        first_output_names = [str(name) for name in first_result.get("output_names", [])]
        second_output_names = [str(name) for name in second_result.get("output_names", [])]

        averaged_rows: List[np.ndarray] = []
        averaged_names: List[str] = []
        for output_name in first_output_names:
            if output_name not in second_output_names:
                continue

            first_row = first_result["transfer_loss_db"][first_output_names.index(output_name), :]
            second_row = second_result["transfer_loss_db"][second_output_names.index(output_name), :]
            if len(second_freqs) != len(first_freqs) or not np.allclose(second_freqs, first_freqs):
                second_row = np.interp(first_freqs, second_freqs, second_row)

            first_ratio = 10.0 ** (first_row / 10.0)
            second_ratio = 10.0 ** (second_row / 10.0)
            averaged_ratio = 0.5 * (first_ratio + second_ratio)
            averaged_rows.append(10.0 * np.log10(np.maximum(averaged_ratio, tiny)))
            averaged_names.append(output_name)

        if not averaged_rows:
            continue

        average_results[f"平均 {label}"] = {
            "freqs": first_freqs,
            "transfer_loss_db": np.vstack(averaged_rows),
            "output_names": averaged_names,
            "channel_names": first_result.get("channel_names", averaged_names),
            "source_file": f"{first_date}+{second_date}",
        }

        first_norm_freqs = first_result.get("normalized_psd_freqs")
        first_norm_ratio = first_result.get("normalized_psd_ratio_avg")
        second_norm_freqs = second_result.get("normalized_psd_freqs")
        second_norm_ratio = second_result.get("normalized_psd_ratio_avg")
        if first_norm_freqs is not None and first_norm_ratio is not None and second_norm_freqs is not None and second_norm_ratio is not None:
            second_ratio_interp = second_norm_ratio
            if len(second_norm_freqs) != len(first_norm_freqs) or not np.allclose(second_norm_freqs, first_norm_freqs):
                second_ratio_interp = np.interp(first_norm_freqs, second_norm_freqs, second_norm_ratio)
            average_results[f"平均 {label}"]["normalized_psd_freqs"] = first_norm_freqs
            average_results[f"平均 {label}"]["normalized_psd_ratio_avg"] = 0.5 * (first_norm_ratio + second_ratio_interp)

        for freq_key, value_key in [
            ("input_accel_psd_freqs", "input_accel_psd"),
            ("force_psd_freqs", "force_psd"),
        ]:
            first_freq = first_result.get(freq_key)
            first_value = first_result.get(value_key)
            second_freq = second_result.get(freq_key)
            second_value = second_result.get(value_key)
            if first_freq is None or first_value is None or second_freq is None or second_value is None:
                continue
            second_value_interp = second_value
            if len(second_freq) != len(first_freq) or not np.allclose(second_freq, first_freq):
                second_value_interp = np.interp(first_freq, second_freq, second_value)
            average_results[f"平均 {label}"][freq_key] = first_freq
            average_results[f"平均 {label}"][value_key] = 0.5 * (first_value + second_value_interp)

        first_frf_freqs = first_result.get("frf_freqs")
        first_frf_db = first_result.get("accel_force_frf_db")
        second_frf_freqs = second_result.get("frf_freqs")
        second_frf_db = second_result.get("accel_force_frf_db")
        if first_frf_freqs is not None and first_frf_db is not None and second_frf_freqs is not None and second_frf_db is not None:
            second_frf_interp = second_frf_db
            if len(second_frf_freqs) != len(first_frf_freqs) or not np.allclose(second_frf_freqs, first_frf_freqs):
                second_frf_interp = np.interp(first_frf_freqs, second_frf_freqs, second_frf_db)
            first_frf_ratio = 10.0 ** (first_frf_db / 10.0)
            second_frf_ratio = 10.0 ** (second_frf_interp / 10.0)
            average_results[f"平均 {label}"]["frf_freqs"] = first_frf_freqs
            average_results[f"平均 {label}"]["accel_force_frf_db"] = 10.0 * np.log10(np.maximum(0.5 * (first_frf_ratio + second_frf_ratio), tiny))

    return average_results


def _average_bar_sort_key(condition_name: str) -> Tuple[int, int, str]:
    label = re.sub(r"^平均\s+", "", str(condition_name)).strip()
    canonical = normalize_condition_label(label)
    if canonical == "no-load":
        return (0, 0, condition_name)
    if canonical == "DVA":
        return (1, 0, condition_name)
    match = re.match(r"(\d+)_PCs", label, flags=re.IGNORECASE)
    if match:
        return (2, int(match.group(1)), condition_name)
    match = re.search(r"plan[-_ ]*(\d+)", label, flags=re.IGNORECASE)
    if match:
        return (3, int(match.group(1)), condition_name)
    return (4, 0, condition_name)


def compute_weighted_band_level_db(
    result: Dict[str, Any],
    sensor_names: List[str],
    weights: List[float],
    band_min: float,
    band_max: float,
) -> float:
    """先在六测点和频段内对线性 PSD 传递率取平均，再转换为 dB。"""
    band_ratio = compute_weighted_band_ratio_mean(result, sensor_names, weights, band_min, band_max)
    return 10.0 * np.log10(max(band_ratio, np.finfo(float).tiny))


def compute_weighted_band_ratio_mean(
    result: Dict[str, Any],
    sensor_names: List[str],
    weights: List[float],
    band_min: float,
    band_max: float,
) -> float:
    """先在测点线性 PSD 比值域等权/加权平均，再取目标频段均值。"""
    output_names = [str(name) for name in result.get("output_names", [])]
    tl_data = result["transfer_loss_db"]
    freqs = result["freqs"]

    selected_rows: List[np.ndarray] = []
    selected_weights: List[float] = []
    for sensor_name, weight in zip(sensor_names, weights):
        if sensor_name not in output_names:
            continue
        row_index = output_names.index(sensor_name)
        if row_index >= tl_data.shape[0]:
            continue
        selected_rows.append(10.0 ** (tl_data[row_index, :] / 10.0))
        selected_weights.append(float(weight))

    if not selected_rows:
        raise ValueError("没有可用于频段均值计算的测点")

    freq_mask = (freqs >= band_min) & (freqs <= band_max)
    if not np.any(freq_mask):
        raise ValueError(f"频率轴中没有 {band_min}-{band_max} Hz 数据")

    selected_weight_array = np.asarray(selected_weights, dtype=float)
    selected_weight_array = selected_weight_array / np.sum(selected_weight_array)
    weighted_ratio = np.average(np.vstack(selected_rows), axis=0, weights=selected_weight_array)
    band_ratio = float(np.mean(weighted_ratio[freq_mask]))
    return max(band_ratio, np.finfo(float).tiny)


def plot_average_damping_rate_bar(
    average_results: Dict[str, Dict[str, Any]],
    sensor_names: List[str],
    weights: List[float],
    band_min: float,
    band_max: float,
    save_path: Path,
) -> Optional[plt.Figure]:
    """绘制 5.22/5.23 同名工况平均后的频段平均减振率柱状图。"""
    baseline_name = next(
        (name for name in average_results if normalize_condition_label(name) == "no-load"),
        None,
    )
    if baseline_name is None:
        print("  warning: 平均结果中没有 no-load 基准，未生成减振率柱状图。")
        return None

    baseline_ratio_mean = compute_weighted_band_ratio_mean(
        average_results[baseline_name],
        sensor_names=sensor_names,
        weights=weights,
        band_min=band_min,
        band_max=band_max,
    )

    labels: List[str] = []
    dr_values: List[float] = []
    delta_values: List[float] = []
    colors: List[str] = []
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]

    palette_index = 0
    for condition_name in sorted(average_results, key=_average_bar_sort_key):
        if condition_name == baseline_name:
            continue

        ratio_mean = compute_weighted_band_ratio_mean(
            average_results[condition_name],
            sensor_names=sensor_names,
            weights=weights,
            band_min=band_min,
            band_max=band_max,
        )
        delta_tl = 10.0 * np.log10(ratio_mean / baseline_ratio_mean)
        damping_rate = (1.0 - ratio_mean / baseline_ratio_mean) * 100.0

        labels.append(condition_name)
        dr_values.append(damping_rate)
        delta_values.append(delta_tl)
        if normalize_condition_label(condition_name) == "DVA":
            colors.append("red")
        else:
            colors.append(palette[palette_index % len(palette)])
            palette_index += 1

    if not labels:
        print("  warning: 平均结果中没有可与 no-load 对比的工况，未生成减振率柱状图。")
        return None

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, dr_values, color=colors, edgecolor="black", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot([], [], color="black", linewidth=1.2, label="no-load 基准线")

    y_min = min(-10.0, min(dr_values) - 8.0)
    y_max = max(10.0, max(dr_values) + 12.0)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"{band_min:.0f}-{band_max:.0f} Hz 六点平均减振率 / %")
    ax.set_title(
        f"5.22+5.23平均数据：{band_min:.0f}-{band_max:.0f} Hz六点平均PSD传递率减振率对比（基准：no-load）\n"
        "5.22输入基准：Data第2行；5.23输入基准：Data第1行"
    )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper right", frameon=True)

    for bar, dr, delta_tl in zip(bars, dr_values, delta_values):
        va = "bottom" if dr >= 0 else "top"
        offset = 1.5 if dr >= 0 else -1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            dr + offset,
            f"{dr:.1f}%\nΔTL={delta_tl:.2f} dB",
            ha="center",
            va=va,
            fontsize=10,
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=figure_save_dpi, bbox_inches="tight")
    print(f"  {band_min:.0f}-{band_max:.0f} Hz 平均减振率图已保存: {save_path}")
    return fig


def _weighted_ratio_curve(result: Dict[str, Any], sensor_names: List[str], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    output_names = [str(name) for name in result.get("output_names", [])]
    tl_data = result["transfer_loss_db"]
    selected_rows: List[np.ndarray] = []
    selected_weights: List[float] = []
    for sensor_name, weight in zip(sensor_names, weights):
        if sensor_name not in output_names:
            continue
        row_index = output_names.index(sensor_name)
        if row_index < tl_data.shape[0]:
            selected_rows.append(10.0 ** (tl_data[row_index, :] / 10.0))
            selected_weights.append(float(weight))
    if not selected_rows:
        raise ValueError("没有可用于统计的测点，请检查数据结构和测点名称。")
    weight_array = np.asarray(selected_weights, dtype=float)
    weight_array = weight_array / np.sum(weight_array)
    return result["freqs"], np.average(np.vstack(selected_rows), axis=0, weights=weight_array)


def _band_average_level_db(result: Dict[str, Any], sensor_names: List[str], weights: List[float], band_min: float, band_max: float) -> float:
    return 10.0 * np.log10(_band_average_ratio_mean(result, sensor_names, weights, band_min, band_max))


def _band_average_ratio_mean(result: Dict[str, Any], sensor_names: List[str], weights: List[float], band_min: float, band_max: float) -> float:
    freqs, ratio = _weighted_ratio_curve(result, sensor_names, weights)
    mask = (freqs >= band_min) & (freqs <= band_max)
    if not np.any(mask):
        raise ValueError(f"频率轴中没有 {band_min:g}-{band_max:g} Hz 数据。")
    return max(float(np.mean(ratio[mask])), np.finfo(float).tiny)


def _band_total_level_db(result: Dict[str, Any], sensor_names: List[str], weights: List[float], band_min: float, band_max: float) -> float:
    freqs, ratio = _weighted_ratio_curve(result, sensor_names, weights)
    mask = (freqs >= band_min) & (freqs <= band_max)
    if not np.any(mask):
        raise ValueError(f"频率轴中没有 {band_min:g}-{band_max:g} Hz 数据。")
    band_freqs = freqs[mask]
    band_ratio = ratio[mask]
    if band_freqs.size > 1:
        band_value = float(np.trapz(band_ratio, band_freqs))
    else:
        band_value = float(band_ratio[0])
    return 10.0 * np.log10(max(band_value, np.finfo(float).tiny))


def _stat_condition_order(names: List[str]) -> List[str]:
    def key(name: str) -> Tuple[int, int, str]:
        label = normalize_condition_label(name)
        if label == "no-load":
            return (0, 0, name)
        if label == "DVA":
            return (1, 0, name)
        match = re.search(r"plan[-_ ]*(\d+)", name, flags=re.IGNORECASE)
        if match:
            return (2, int(match.group(1)), name)
        match = re.match(r"(\d+)", name)
        if match:
            return (3, int(match.group(1)), name)
        return (4, 0, name)
    return sorted(names, key=key)


def plot_band_total_level_bar(
    all_results: Dict[str, Dict[str, Any]],
    sensor_names: List[str],
    weights: List[float],
    band_min: float,
    band_max: float,
    condition_order: Optional[List[str]] = None,
) -> Optional[plt.Figure]:
    rows: List[Tuple[str, float]] = []
    for condition_name in resolve_condition_order(all_results, condition_order):
        try:
            rows.append((condition_name, _band_total_level_db(all_results[condition_name], sensor_names, weights, band_min, band_max)))
        except Exception as exc:
            print(f"  warning: {condition_name} 总振级计算失败: {exc}")
    if not rows:
        return None

    labels = [name for name, _ in rows]
    values = [value for _, value in rows]
    colors = [get_condition_color(name) or f"C{idx}" for idx, name in enumerate(labels)]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bars = ax.bar(np.arange(len(labels)), values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("频段总振级 / dB")
    ax.set_title(f"{band_min:.0f}-{band_max:.0f} Hz 输入归一化PSD传递率频段总振级")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def plot_damping_rate_bar_for_results(
    all_results: Dict[str, Dict[str, Any]],
    sensor_names: List[str],
    weights: List[float],
    band_min: float,
    band_max: float,
    condition_order: Optional[List[str]] = None,
) -> Optional[plt.Figure]:
    baseline_name = next((name for name in all_results if normalize_condition_label(name) == "no-load"), None)
    if baseline_name is None:
        print("  warning: 缺少 no-load 基准，平均减振率图未生成。")
        return None
    baseline_ratio_mean = _band_average_ratio_mean(all_results[baseline_name], sensor_names, weights, band_min, band_max)

    labels: List[str] = []
    dr_values: List[float] = []
    delta_values: List[float] = []
    colors: List[str] = []
    ordered_names = [name for name in resolve_condition_order(all_results, condition_order) if name != baseline_name]
    for condition_name in ordered_names:
        try:
            ratio_mean = _band_average_ratio_mean(all_results[condition_name], sensor_names, weights, band_min, band_max)
        except Exception as exc:
            print(f"  warning: {condition_name} 减振率计算失败: {exc}")
            continue
        delta_tl = 10.0 * np.log10(ratio_mean / baseline_ratio_mean)
        damping_rate = (1.0 - ratio_mean / baseline_ratio_mean) * 100.0
        labels.append(condition_name)
        dr_values.append(damping_rate)
        delta_values.append(delta_tl)
        colors.append(get_condition_color(condition_name) or f"C{len(colors)}")
    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bars = ax.bar(np.arange(len(labels)), dr_values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot([], [], color="black", linewidth=1.2, label="no-load 基准")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"{band_min:.0f}-{band_max:.0f} Hz 平均减振率 / %")
    ax.set_title(f"{band_min:.0f}-{band_max:.0f} Hz 平均PSD传递率减振率对比（基准：no-load）")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best", frameon=True)
    y_min = min(-10.0, min(dr_values) - 8.0)
    y_max = max(10.0, max(dr_values) + 12.0)
    ax.set_ylim(y_min, y_max)
    for bar, dr, delta_tl in zip(bars, dr_values, delta_values):
        va = "bottom" if dr >= 0 else "top"
        offset = 1.2 if dr >= 0 else -1.2
        ax.text(bar.get_x() + bar.get_width() / 2, dr + offset, f"{dr:.1f}%\nΔTL={delta_tl:.2f} dB", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    return fig
