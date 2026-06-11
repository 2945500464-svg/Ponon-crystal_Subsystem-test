from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import fs
from .exporters import save_figure_as
from .mat_io import load_mat_file
from .microphone_processing import (
    _fractional_octave_bands,
    _frequency_band_spl,
    _normalize_octave_denominator,
    _octave_label,
    _safe_db_from_pressure_power,
)
from .plot_style import apply_plot_style, normalize_plot_style
from .plotting import resolve_condition_order
from .signal_processing import choose_nfft, compute_channel_psd, get_analysis_segments, trim_data_edges
from .utils import get_condition_color, get_display_condition_name, normalize_condition_label


VEHICLE_611_FOLDER = "6.11"
VEHICLE_MIC_NAMES = ["前排麦克风", "中排麦克风", "后排麦克风"]
VEHICLE_ROAD_NAMES = ["左半轴X", "左半轴Y", "左半轴Z", "右半轴X", "右半轴Y", "右半轴Z"]
VEHICLE_SUBFRAME_NAMES = ["左前", "左中", "左后", "右前", "右中", "右后", "左动力吸振器", "右动力吸振器"]
VEHICLE_REQUIRED_CHANNELS = 17
VEHICLE_MIC_INDICES = [0, 1, 2]
VEHICLE_LEFT_ROAD_INDICES = [3, 4, 5]
VEHICLE_RIGHT_ROAD_INDICES = [6, 7, 8]
VEHICLE_ROAD_INDICES = VEHICLE_LEFT_ROAD_INDICES + VEHICLE_RIGHT_ROAD_INDICES
VEHICLE_SUBFRAME_INDICES = list(range(9, 17))


VEHICLE_PLOT_GROUPS = {
    "麦克风声学图": [
        ("FFT声压级", "mic_fft_spl"),
        ("总声压级", "mic_total_spl"),
        ("分数倍频程声压级", "mic_fractional_octave"),
    ],
    "加速度振动图": [
        ("半轴路谱PSD", "road_psd"),
        ("副车架8测点PSD", "subframe_psd"),
        ("副车架8点平均PSD", "subframe_avg_psd"),
        ("路谱归一化副车架振动", "road_normalized_subframe"),
    ],
    "声振关联图": [
        ("路谱归一化麦克风声压", "road_normalized_mic"),
        ("副车架-麦克风传递", "subframe_to_mic"),
        ("声振相干性", "coherence"),
        ("频段改善散点图", "improvement_scatter"),
    ],
}


def vehicle_condition_prefix(stem: str) -> str:
    """Remove a trial suffix such as _1, -2, run3, test4, or rep5."""
    text = str(stem).strip()
    for pattern in (
        r"^(?P<prefix>.+?)[_\-\s]+(?:run|test|trial|rep)?\d+$",
        r"^(?P<prefix>.+?)[（(]\d+[）)]$",
    ):
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group("prefix").strip()
    return text


def _safe_db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, np.finfo(float).tiny))


def _band_mean_db(freqs: np.ndarray, db_values: np.ndarray, freq_min: float, freq_max: float) -> float:
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(mask):
        return float("nan")
    linear = 10.0 ** (db_values[mask] / 10.0)
    return float(10.0 * np.log10(np.maximum(np.mean(linear), np.finfo(float).tiny)))


def _compute_vehicle_coherence(
    segments: np.ndarray,
    sample_rate: float,
    desired_df: float,
    freq_min: float,
    freq_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average magnitude-squared coherence between each microphone and 8 subframe sensors."""
    sxx_sum: Optional[np.ndarray] = None
    syy_sum: Optional[np.ndarray] = None
    sxy_sum: Optional[np.ndarray] = None
    freqs: Optional[np.ndarray] = None

    for segment in segments:
        n_samples = segment.shape[1]
        nfft = choose_nfft(n_samples, sample_rate, desired_df)
        window = np.hanning(n_samples)
        window_power = float(np.sum(window ** 2))
        if window_power <= 0:
            window = np.ones(n_samples)
            window_power = float(n_samples)

        centered = segment - np.mean(segment, axis=1, keepdims=True)
        spectrum = np.fft.rfft(centered * window[np.newaxis, :], n=nfft, axis=1)
        segment_freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        mic_fft = spectrum[VEHICLE_MIC_INDICES, :]
        sub_fft = spectrum[VEHICLE_SUBFRAME_INDICES, :]
        scale = 1.0 / (sample_rate * window_power)

        sxx = (np.abs(mic_fft) ** 2) * scale
        syy = (np.abs(sub_fft) ** 2) * scale
        sxy = mic_fft[:, np.newaxis, :] * np.conj(sub_fft[np.newaxis, :, :]) * scale

        sxx_sum = sxx if sxx_sum is None else sxx_sum + sxx
        syy_sum = syy if syy_sum is None else syy_sum + syy
        sxy_sum = sxy if sxy_sum is None else sxy_sum + sxy
        if freqs is None:
            freqs = segment_freqs

    if freqs is None or sxx_sum is None or syy_sum is None or sxy_sum is None:
        raise ValueError("没有可用于相干性计算的数据段")

    n_segments = float(max(len(segments), 1))
    sxx_mean = sxx_sum / n_segments
    syy_mean = syy_sum / n_segments
    sxy_mean = sxy_sum / n_segments
    denominator = np.maximum(sxx_mean[:, np.newaxis, :] * syy_mean[np.newaxis, :, :], np.finfo(float).tiny)
    coherence = np.abs(sxy_mean) ** 2 / denominator
    coherence = np.clip(coherence, 0.0, 1.0)
    coherence_avg = np.mean(coherence, axis=1)

    mask = (freqs >= freq_min) & (freqs <= freq_max)
    return freqs[mask], coherence_avg[:, mask]


def _build_vehicle_result(
    condition_name: str,
    source_files: List[str],
    freqs: np.ndarray,
    psd_all: np.ndarray,
    coherence_avg: np.ndarray,
    octave_denominator: int,
) -> Dict[str, Any]:
    tiny = np.finfo(float).tiny
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    mic_psd = psd_all[VEHICLE_MIC_INDICES, :]
    left_road_psd = np.sum(psd_all[VEHICLE_LEFT_ROAD_INDICES, :], axis=0)
    right_road_psd = np.sum(psd_all[VEHICLE_RIGHT_ROAD_INDICES, :], axis=0)
    road_psd = 0.5 * (left_road_psd + right_road_psd)
    subframe_psd = psd_all[VEHICLE_SUBFRAME_INDICES, :]
    subframe_avg_psd = np.mean(subframe_psd, axis=0)

    mic_fft_spl = _safe_db_from_pressure_power(mic_psd * df)
    mic_total_spl = _safe_db_from_pressure_power(np.sum(mic_psd * df, axis=1))

    octave_denominator = _normalize_octave_denominator(octave_denominator)
    octave_centers: List[float] = []
    octave_rows: List[np.ndarray] = []
    for center, lower, upper in _fractional_octave_bands(float(freqs[0]), float(freqs[-1]), octave_denominator):
        mask = (freqs >= lower) & (freqs < upper)
        if not np.any(mask):
            continue
        octave_centers.append(center)
        octave_rows.append(_safe_db_from_pressure_power(np.sum(mic_psd[:, mask] * df, axis=1)))
    octave_spl = np.vstack(octave_rows).T if octave_rows else np.empty((3, 0))

    return {
        "source_file": source_files[0] if len(source_files) == 1 else "; ".join(source_files),
        "source_files": source_files,
        "condition_name": condition_name,
        "freqs": freqs,
        "psd_all": psd_all,
        "mic_psd": mic_psd,
        "left_road_psd": left_road_psd,
        "right_road_psd": right_road_psd,
        "road_psd": road_psd,
        "subframe_psd": subframe_psd,
        "subframe_avg_psd": subframe_avg_psd,
        "mic_fft_spl": mic_fft_spl,
        "mic_total_spl": mic_total_spl,
        "octave_centers": np.asarray(octave_centers, dtype=float),
        "octave_spl": octave_spl,
        "octave_denominator": octave_denominator,
        "subframe_road_norm_db": _safe_db(subframe_avg_psd / np.maximum(road_psd, tiny)),
        "mic_road_norm_db": _safe_db(mic_psd / np.maximum(road_psd[np.newaxis, :], tiny)),
        "mic_subframe_norm_db": _safe_db(mic_psd / np.maximum(subframe_avg_psd[np.newaxis, :], tiny)),
        "coherence_avg": coherence_avg,
    }


def analyze_vehicle_files(
    mat_files: List[Path],
    freq_min: float,
    freq_max: float,
    desired_df: float,
    duration_sec: float,
    mode: str,
    start_sec: float,
    trim_start: float,
    trim_end: float,
    octave_denominator: int,
    average_by_condition: bool = False,
) -> Dict[str, Dict[str, Any]]:
    raw_results: Dict[str, Dict[str, Any]] = {}
    for mat_file in mat_files:
        print("-" * 72)
        print(f"正在处理6.11实车数据: {mat_file.name}")
        _time_data, data = load_mat_file(mat_file)
        if data.shape[0] < VEHICLE_REQUIRED_CHANNELS:
            print(f"  warning: {mat_file.name} Data只有 {data.shape[0]} 行，6.11实车流程需要至少17行，已跳过")
            continue
        trimmed_data, _ = trim_data_edges(data, fs, trim_start, trim_end)
        segments, _segment_info = get_analysis_segments(trimmed_data, fs, duration_sec, mode, start_sec)
        freqs, psd_all = compute_channel_psd(
            segments=segments,
            sample_rate=fs,
            desired_df=desired_df,
            freq_min=freq_min,
            freq_max=freq_max,
            channel_indices=list(range(VEHICLE_REQUIRED_CHANNELS)),
        )
        coh_freqs, coherence_avg = _compute_vehicle_coherence(segments, fs, desired_df, freq_min, freq_max)
        if coh_freqs.shape != freqs.shape or not np.allclose(coh_freqs, freqs):
            coherence_avg = np.vstack([np.interp(freqs, coh_freqs, row) for row in coherence_avg])

        raw_results[mat_file.stem] = _build_vehicle_result(
            condition_name=mat_file.stem,
            source_files=[mat_file.name],
            freqs=freqs,
            psd_all=psd_all,
            coherence_avg=coherence_avg,
            octave_denominator=octave_denominator,
        )

    if not average_by_condition:
        return raw_results

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for name, result in raw_results.items():
        grouped.setdefault(vehicle_condition_prefix(name), []).append(result)

    averaged: Dict[str, Dict[str, Any]] = {}
    for group_name, items in grouped.items():
        freqs = items[0]["freqs"]
        psd_stack = np.stack([item["psd_all"] for item in items], axis=0)
        coh_stack = np.stack([item["coherence_avg"] for item in items], axis=0)
        source_files: List[str] = []
        for item in items:
            source_files.extend(item.get("source_files") or [str(item.get("source_file", ""))])
        averaged[group_name] = _build_vehicle_result(
            condition_name=group_name,
            source_files=source_files,
            freqs=freqs,
            psd_all=np.mean(psd_stack, axis=0),
            coherence_avg=np.mean(coh_stack, axis=0),
            octave_denominator=octave_denominator,
        )
    return averaged


def _ordered_names(results: Dict[str, Dict[str, Any]], condition_order: Optional[List[str]]) -> List[str]:
    return resolve_condition_order(results, condition_order)


def _plot_line(ax: Any, freqs: np.ndarray, values: np.ndarray, name: str) -> None:
    ax.plot(freqs, values, label=name, color=get_condition_color(name), linewidth=1.6)


def _finish_axis(ax: Any, freq_min: float, freq_max: float, title: str, ylabel: str) -> None:
    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best", frameon=True)


def _new_line_fig(title: str, ylabel: str) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_vehicle_mic_fft_spl(results: Dict[str, Dict[str, Any]], mic_index: int, freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    fig, ax = _new_line_fig(f"6.11 {VEHICLE_MIC_NAMES[mic_index]} FFT声压级", "声压级 / dB SPL")
    has_data = False
    for name in _ordered_names(results, condition_order):
        result = results[name]
        freqs, spl = _frequency_band_spl(result["freqs"], result["mic_psd"], freq_min, freq_max)
        if freqs.size == 0:
            continue
        _plot_line(ax, freqs, spl[mic_index, :], name)
        has_data = True
    if not has_data:
        plt.close(fig)
        return None
    _finish_axis(ax, freq_min, freq_max, f"6.11 {VEHICLE_MIC_NAMES[mic_index]} FFT声压级", "声压级 / dB SPL")
    fig.tight_layout()
    return fig


def plot_vehicle_total_spl(results: Dict[str, Dict[str, Any]], mic_index: int, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    names = _ordered_names(results, condition_order)
    if not names:
        return None
    values = [float(results[name]["mic_total_spl"][mic_index]) for name in names]
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    colors = [get_condition_color(name) for name in names]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([get_display_condition_name(name) for name in names], rotation=25, ha="right")
    ax.set_ylabel("总声压级 / dB SPL")
    ax.set_title(f"6.11 {VEHICLE_MIC_NAMES[mic_index]} 频段总声压级")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_vehicle_fractional_octave(results: Dict[str, Dict[str, Any]], mic_index: int, freq_min: float, freq_max: float, octave_denominator: int, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    octave_label = _octave_label(octave_denominator)
    fig, ax = _new_line_fig(f"6.11 {VEHICLE_MIC_NAMES[mic_index]} {octave_label}声压级", "声压级 / dB SPL")
    has_data = False
    for name in _ordered_names(results, condition_order):
        result = results[name]
        centers = result["octave_centers"]
        spl = result["octave_spl"]
        if centers.size == 0 or spl.shape[1] == 0:
            continue
        ax.plot(centers, spl[mic_index, :], marker="o", label=name, color=get_condition_color(name), linewidth=1.6)
        has_data = True
    if not has_data:
        plt.close(fig)
        return None
    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel(f"{octave_label}中心频率 / Hz")
    ax.set_ylabel("声压级 / dB SPL")
    ax.set_title(f"6.11 {VEHICLE_MIC_NAMES[mic_index]} {octave_label}声压级")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def plot_vehicle_road_psd(results: Dict[str, Dict[str, Any]], freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 8.2), sharex=True)
    series = [("左半轴三向合成PSD", "left_road_psd"), ("右半轴三向合成PSD", "right_road_psd"), ("左右半轴平均路谱PSD", "road_psd")]
    for ax, (title, key) in zip(axes, series):
        for name in _ordered_names(results, condition_order):
            _plot_line(ax, results[name]["freqs"], _safe_db(results[name][key]), name)
        _finish_axis(ax, freq_min, freq_max, f"6.11 {title}", "PSD / dB")
    fig.tight_layout()
    return fig


def plot_vehicle_subframe_sensor_psd(results: Dict[str, Dict[str, Any]], sensor_index: int, freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    fig, ax = _new_line_fig(f"6.11 副车架{VEHICLE_SUBFRAME_NAMES[sensor_index]} PSD", "加速度PSD / dB")
    for name in _ordered_names(results, condition_order):
        _plot_line(ax, results[name]["freqs"], _safe_db(results[name]["subframe_psd"][sensor_index, :]), name)
    _finish_axis(ax, freq_min, freq_max, f"6.11 副车架{VEHICLE_SUBFRAME_NAMES[sensor_index]} PSD", "加速度PSD / dB")
    fig.tight_layout()
    return fig


def plot_vehicle_subframe_average_psd(results: Dict[str, Dict[str, Any]], freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    fig, ax = _new_line_fig("6.11 副车架8点平均PSD", "加速度PSD / dB")
    for name in _ordered_names(results, condition_order):
        _plot_line(ax, results[name]["freqs"], _safe_db(results[name]["subframe_avg_psd"]), name)
    _finish_axis(ax, freq_min, freq_max, "6.11 副车架8点平均PSD", "加速度PSD / dB")
    fig.tight_layout()
    return fig


def plot_vehicle_subframe_road_norm(results: Dict[str, Dict[str, Any]], freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    fig, ax = _new_line_fig("6.11 路谱归一化副车架振动", "10lg(S_sub/S_road) / dB")
    for name in _ordered_names(results, condition_order):
        _plot_line(ax, results[name]["freqs"], results[name]["subframe_road_norm_db"], name)
    _finish_axis(ax, freq_min, freq_max, "6.11 路谱归一化副车架振动", "10lg(S_sub/S_road) / dB")
    fig.tight_layout()
    return fig


def plot_vehicle_mic_road_norm(results: Dict[str, Dict[str, Any]], mic_index: int, freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    fig, ax = _new_line_fig(f"6.11 {VEHICLE_MIC_NAMES[mic_index]} 路谱归一化声压", "10lg(S_p/S_road) / dB")
    for name in _ordered_names(results, condition_order):
        _plot_line(ax, results[name]["freqs"], results[name]["mic_road_norm_db"][mic_index, :], name)
    _finish_axis(ax, freq_min, freq_max, f"6.11 {VEHICLE_MIC_NAMES[mic_index]} 路谱归一化声压", "10lg(S_p/S_road) / dB")
    fig.tight_layout()
    return fig


def plot_vehicle_mic_subframe_norm(results: Dict[str, Dict[str, Any]], mic_index: int, freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    fig, ax = _new_line_fig(f"6.11 {VEHICLE_MIC_NAMES[mic_index]} 副车架-声压传递", "10lg(S_p/S_sub) / dB")
    for name in _ordered_names(results, condition_order):
        _plot_line(ax, results[name]["freqs"], results[name]["mic_subframe_norm_db"][mic_index, :], name)
    _finish_axis(ax, freq_min, freq_max, f"6.11 {VEHICLE_MIC_NAMES[mic_index]} 副车架-声压传递", "10lg(S_p/S_sub) / dB")
    fig.tight_layout()
    return fig


def plot_vehicle_coherence(results: Dict[str, Dict[str, Any]], mic_index: int, freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    fig, ax = _new_line_fig(f"6.11 {VEHICLE_MIC_NAMES[mic_index]} 与副车架8点平均相干性", "相干系数")
    for name in _ordered_names(results, condition_order):
        _plot_line(ax, results[name]["freqs"], results[name]["coherence_avg"][mic_index, :], name)
    ax.set_ylim(0, 1.02)
    _finish_axis(ax, freq_min, freq_max, f"6.11 {VEHICLE_MIC_NAMES[mic_index]} 与副车架8点平均相干性", "相干系数")
    fig.tight_layout()
    return fig


def plot_vehicle_improvement_scatter(results: Dict[str, Dict[str, Any]], mic_indices: List[int], freq_min: float, freq_max: float, condition_order: Optional[List[str]]) -> Optional[plt.Figure]:
    baseline_name = next((name for name in results if normalize_condition_label(name) == "no-load"), None)
    if baseline_name is None:
        return None
    baseline = results[baseline_name]
    base_sub = _band_mean_db(baseline["freqs"], baseline["subframe_road_norm_db"], freq_min, freq_max)
    base_mic_values = [_band_mean_db(baseline["freqs"], baseline["mic_road_norm_db"][idx, :], freq_min, freq_max) for idx in mic_indices]
    base_mic = float(np.nanmean(base_mic_values))

    names = [name for name in _ordered_names(results, condition_order) if name != baseline_name]
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for name in names:
        result = results[name]
        sub_delta = _band_mean_db(result["freqs"], result["subframe_road_norm_db"], freq_min, freq_max) - base_sub
        mic_values = [_band_mean_db(result["freqs"], result["mic_road_norm_db"][idx, :], freq_min, freq_max) for idx in mic_indices]
        mic_delta = float(np.nanmean(mic_values)) - base_mic
        ax.scatter(sub_delta, mic_delta, s=70, color=get_condition_color(name), label=name)
        ax.text(sub_delta, mic_delta, f" {get_display_condition_name(name)}", va="center")
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("副车架路谱归一化振动变化 / dB")
    ax.set_ylabel("麦克风路谱归一化声压变化 / dB")
    ax.set_title(f"6.11 {freq_min:.0f}-{freq_max:.0f} Hz 声振改善关系（相对no-load）")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def save_vehicle_outputs(
    results: Dict[str, Dict[str, Any]],
    plot_options: Dict[str, bool],
    save_dir: Path,
    image_format: str,
    dpi_value: int,
    show_after: bool,
    save_figures_flag: bool,
    plot_style: Dict[str, Any],
    freq_min: float,
    freq_max: float,
    mic_indices: List[int],
    octave_denominator: int,
) -> List[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    active_style = normalize_plot_style(plot_style)
    condition_order = list(active_style.get("condition_order") or [])
    saved_paths: List[Path] = []
    generated_figures: List[plt.Figure] = []

    def handle(fig: Optional[plt.Figure], filename: str) -> None:
        if fig is None:
            return
        apply_plot_style(fig, active_style)
        if save_figures_flag:
            saved_paths.append(save_figure_as(fig, save_dir, filename, image_format, dpi_value))
        generated_figures.append(fig)

    for mic_index in mic_indices:
        if plot_options.get("mic_fft_spl"):
            handle(plot_vehicle_mic_fft_spl(results, mic_index, freq_min, freq_max, condition_order), f"6.11_{VEHICLE_MIC_NAMES[mic_index]}_FFT声压级")
        if plot_options.get("mic_total_spl"):
            handle(plot_vehicle_total_spl(results, mic_index, condition_order), f"6.11_{VEHICLE_MIC_NAMES[mic_index]}_总声压级")
        if plot_options.get("mic_fractional_octave"):
            label = _octave_label(octave_denominator)
            handle(plot_vehicle_fractional_octave(results, mic_index, freq_min, freq_max, octave_denominator, condition_order), f"6.11_{VEHICLE_MIC_NAMES[mic_index]}_{label}声压级")
        if plot_options.get("road_normalized_mic"):
            handle(plot_vehicle_mic_road_norm(results, mic_index, freq_min, freq_max, condition_order), f"6.11_{VEHICLE_MIC_NAMES[mic_index]}_路谱归一化声压")
        if plot_options.get("subframe_to_mic"):
            handle(plot_vehicle_mic_subframe_norm(results, mic_index, freq_min, freq_max, condition_order), f"6.11_{VEHICLE_MIC_NAMES[mic_index]}_副车架声压传递")
        if plot_options.get("coherence"):
            handle(plot_vehicle_coherence(results, mic_index, freq_min, freq_max, condition_order), f"6.11_{VEHICLE_MIC_NAMES[mic_index]}_声振相干性")

    if plot_options.get("road_psd"):
        handle(plot_vehicle_road_psd(results, freq_min, freq_max, condition_order), "6.11_半轴路谱PSD")
    if plot_options.get("subframe_psd"):
        for idx, name in enumerate(VEHICLE_SUBFRAME_NAMES):
            handle(plot_vehicle_subframe_sensor_psd(results, idx, freq_min, freq_max, condition_order), f"6.11_副车架PSD_{name}")
    if plot_options.get("subframe_avg_psd"):
        handle(plot_vehicle_subframe_average_psd(results, freq_min, freq_max, condition_order), "6.11_副车架8点平均PSD")
    if plot_options.get("road_normalized_subframe"):
        handle(plot_vehicle_subframe_road_norm(results, freq_min, freq_max, condition_order), "6.11_路谱归一化副车架振动")
    if plot_options.get("improvement_scatter"):
        handle(plot_vehicle_improvement_scatter(results, mic_indices, freq_min, freq_max, condition_order), "6.11_声振改善散点图")

    if show_after and generated_figures:
        plt.show()
    else:
        for fig in generated_figures:
            plt.close(fig)
    return saved_paths
