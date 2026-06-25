from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .config import fs, target_df
from .exporters import save_figure_as
from .mat_io import load_mat_file
from .plot_style import apply_plot_style, normalize_plot_style
from .plotting import resolve_condition_order
from .signal_processing import choose_nfft
from .utils import get_condition_color, get_display_condition_name

MICROPHONE_NAMES = ["主驾驶麦克风", "中排麦克风", "后排麦克风"]
MICROPHONE_REFERENCE_PA = 20e-6
MICROPHONE_FREQ_MIN = 70.0
MICROPHONE_FREQ_MAX = 140.0
MICROPHONE_FFT_RESOLUTION_HZ = None
THIRD_OCTAVE_NOMINAL_CENTERS = np.array(
    [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000],
    dtype=float,
)
FRACTIONAL_OCTAVE_DENOMINATORS = [3, 6, 12, 24]


def _safe_db_from_pressure_power(pressure_power: np.ndarray) -> np.ndarray:
    tiny_power = np.finfo(float).tiny
    return 10.0 * np.log10(np.maximum(pressure_power, tiny_power) / (MICROPHONE_REFERENCE_PA ** 2))


def _compute_microphone_psd(
    data: np.ndarray,
    sample_rate: float = fs,
    desired_df: float = target_df,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算前三路麦克风信号的单边 PSD，输入单位 Pa，输出单位 Pa^2/Hz。"""
    if data.ndim != 2:
        raise ValueError(f"Data 维度异常：{data.shape}")
    if data.shape[0] < 3:
        raise ValueError(f"麦克风数据至少需要 3 行，当前只有 {data.shape[0]} 行")

    mic_data = np.asarray(data[:3, :], dtype=float)
    n_samples = mic_data.shape[1]
    if n_samples < 2:
        raise ValueError("麦克风数据样本数不足")

    nfft = choose_nfft(n_samples, sample_rate, desired_df)
    window = np.hanning(n_samples)
    window_power = float(np.sum(window ** 2))
    if window_power <= 0:
        window = np.ones(n_samples)
        window_power = float(n_samples)

    mic_data = mic_data - np.mean(mic_data, axis=1, keepdims=True)
    spectrum = np.fft.rfft(mic_data * window[np.newaxis, :], n=nfft, axis=1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    psd = (np.abs(spectrum) ** 2) / (sample_rate * window_power)
    if psd.shape[1] > 2:
        psd[:, 1:-1] *= 2.0
    return freqs, psd


def _octave_label(octave_denominator: int) -> str:
    denominator = int(octave_denominator)
    if denominator == 3:
        return "三分之一倍频程"
    if denominator == 6:
        return "六分之一倍频程"
    if denominator == 12:
        return "十二分之一倍频程"
    if denominator == 24:
        return "二十四分之一倍频程"
    return f"1/{denominator} 倍频程"


def _normalize_octave_denominator(octave_denominator: int) -> int:
    try:
        denominator = int(octave_denominator)
    except (TypeError, ValueError):
        denominator = 3
    if denominator not in FRACTIONAL_OCTAVE_DENOMINATORS:
        print(f"  warning: 不支持 1/{denominator} 倍频程，已回退到 1/3 倍频程")
        denominator = 3
    return denominator


def _fractional_octave_bands(freq_min: float, freq_max: float, octave_denominator: int = 3) -> List[Tuple[float, float, float]]:
    """返回与目标频段有交集的分数倍频程频带：中心频率、下限、上限。"""
    octave_denominator = _normalize_octave_denominator(octave_denominator)
    bands: List[Tuple[float, float, float]] = []
    factor = 2.0 ** (1.0 / (2.0 * octave_denominator))
    if octave_denominator == 3:
        centers = THIRD_OCTAVE_NOMINAL_CENTERS
    else:
        # 用 1000 Hz 作为参考中心频率，按 2^(1/N) 的几何间隔生成 1/N 倍频程中心频率。
        k_min = int(np.floor(octave_denominator * np.log2(max(freq_min, 1e-12) / 1000.0))) - 2
        k_max = int(np.ceil(octave_denominator * np.log2(max(freq_max, 1e-12) / 1000.0))) + 2
        centers = np.asarray([1000.0 * (2.0 ** (k / octave_denominator)) for k in range(k_min, k_max + 1)], dtype=float)
    for center in centers:
        lower = center / factor
        upper = center * factor
        if upper >= freq_min and lower <= freq_max:
            bands.append((float(center), float(lower), float(upper)))
    return bands


def _frequency_band_spl(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_min: float,
    freq_max: float,
    band_width_hz: Optional[float] = MICROPHONE_FFT_RESOLUTION_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """将 PSD 转换为声压级；默认使用实际 FFT bin 宽度，也可指定分箱宽度。"""
    if freqs.size < 2:
        raise ValueError("频率轴点数不足")
    if band_width_hz is not None and band_width_hz <= 0:
        raise ValueError("FFT 声压级频率分辨率必须大于 0")

    df = float(freqs[1] - freqs[0])
    if band_width_hz is None:
        mask = (freqs >= freq_min) & (freqs <= freq_max)
        if not np.any(mask):
            return np.asarray([], dtype=float), np.empty((psd.shape[0], 0))
        return freqs[mask], _safe_db_from_pressure_power(psd[:, mask] * df)

    start_center = int(np.ceil(freq_min))
    end_center = int(np.floor(freq_max))
    if start_center <= end_center:
        centers = np.arange(start_center, end_center + 1, dtype=float)
        edges = [(center - band_width_hz / 2.0, center + band_width_hz / 2.0) for center in centers]
    else:
        centers = np.asarray([(freq_min + freq_max) / 2.0], dtype=float)
        edges = [(freq_min, freq_max)]

    spl_rows: List[np.ndarray] = []
    valid_centers: List[float] = []
    for center, (lower, upper) in zip(centers, edges):
        lower = max(float(lower), float(freq_min))
        upper = min(float(upper), float(freq_max))
        if center == centers[-1]:
            mask = (freqs >= lower) & (freqs <= upper)
        else:
            mask = (freqs >= lower) & (freqs < upper)
        if not np.any(mask):
            continue
        spl_rows.append(_safe_db_from_pressure_power(np.sum(psd[:, mask] * df, axis=1)))
        valid_centers.append(float(center))

    if not spl_rows:
        return np.asarray([], dtype=float), np.empty((psd.shape[0], 0))
    return np.asarray(valid_centers, dtype=float), np.vstack(spl_rows).T


def _compute_microphone_result_for_file(
    mat_file: Path,
    sample_rate: float,
    desired_df: float,
    freq_min: float,
    freq_max: float,
    octave_denominator: int,
) -> Dict[str, Any]:
    _time_data, data = load_mat_file(mat_file)
    if data.shape[0] > 3:
        print(f"  warning: {mat_file.name} 的 Data 有 {data.shape[0]} 行，麦克风处理只使用前 3 行")

    freqs, psd = _compute_microphone_psd(data, sample_rate=sample_rate, desired_df=desired_df)
    if freqs.size < 2:
        raise ValueError("频率轴点数不足")

    df = float(freqs[1] - freqs[0])
    fft_spl = _safe_db_from_pressure_power(psd * df)

    total_mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(total_mask):
        raise ValueError(f"频率轴中没有 {freq_min:g}-{freq_max:g} Hz 数据")
    total_spl = _safe_db_from_pressure_power(np.sum(psd[:, total_mask] * df, axis=1))

    third_centers: List[float] = []
    third_spl_rows: List[np.ndarray] = []
    octave_denominator = _normalize_octave_denominator(octave_denominator)
    octave_label = _octave_label(octave_denominator)
    for center, lower, upper in _fractional_octave_bands(freq_min, freq_max, octave_denominator):
        band_mask = (freqs >= lower) & (freqs < upper)
        if not np.any(band_mask):
            print(f"  warning: {mat_file.name} 的 {center:g} Hz {octave_label}没有有效频率点，已跳过")
            continue
        third_centers.append(center)
        third_spl_rows.append(_safe_db_from_pressure_power(np.sum(psd[:, band_mask] * df, axis=1)))

    if third_spl_rows:
        third_octave_spl = np.vstack(third_spl_rows).T
    else:
        third_octave_spl = np.empty((3, 0))

    return {
        "source_file": mat_file.name,
        "freqs": freqs,
        "psd": psd,
        "fft_spl": fft_spl,
        "total_spl": total_spl,
        "third_octave_centers": np.asarray(third_centers, dtype=float),
        "third_octave_spl": third_octave_spl,
        "octave_denominator": octave_denominator,
    }


def _microphone_average_prefix(stem: str) -> str:
    """去掉文件名最后的 _数字 后缀，用于重复测试数据分组平均。"""
    match = re.match(r"^(?P<prefix>.+)_\d+$", stem)
    if match:
        return match.group("prefix")
    return stem


def _build_microphone_result_from_psd(
    condition_name: str,
    source_files: List[str],
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_min: float,
    freq_max: float,
    octave_denominator: int,
) -> Dict[str, Any]:
    """基于线性 PSD 结果重新计算麦克风 SPL 图所需数据。"""
    if freqs.size < 2:
        raise ValueError("频率轴点数不足")

    df = float(freqs[1] - freqs[0])
    fft_spl = _safe_db_from_pressure_power(psd * df)

    total_mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(total_mask):
        raise ValueError(f"频率轴中没有 {freq_min:g}-{freq_max:g} Hz 数据")
    total_spl = _safe_db_from_pressure_power(np.sum(psd[:, total_mask] * df, axis=1))

    third_centers: List[float] = []
    third_spl_rows: List[np.ndarray] = []
    octave_denominator = _normalize_octave_denominator(octave_denominator)
    octave_label = _octave_label(octave_denominator)
    for center, lower, upper in _fractional_octave_bands(freq_min, freq_max, octave_denominator):
        band_mask = (freqs >= lower) & (freqs < upper)
        if not np.any(band_mask):
            print(f"  warning: {condition_name} 的 {center:g} Hz {octave_label}没有有效频率点，已跳过")
            continue
        third_centers.append(center)
        third_spl_rows.append(_safe_db_from_pressure_power(np.sum(psd[:, band_mask] * df, axis=1)))

    if third_spl_rows:
        third_octave_spl = np.vstack(third_spl_rows).T
    else:
        third_octave_spl = np.empty((3, 0))

    return {
        "source_file": source_files[0] if len(source_files) == 1 else "; ".join(source_files),
        "source_files": source_files,
        "averaged_condition": condition_name,
        "freqs": freqs,
        "psd": psd,
        "fft_spl": fft_spl,
        "total_spl": total_spl,
        "third_octave_centers": np.asarray(third_centers, dtype=float),
        "third_octave_spl": third_octave_spl,
        "octave_denominator": octave_denominator,
    }


def _average_microphone_results_by_prefix(
    raw_results: Dict[str, Dict[str, Any]],
    freq_min: float,
    freq_max: float,
    octave_denominator: int,
) -> Dict[str, Dict[str, Any]]:
    """将同名前缀重复文件按线性 PSD 能量平均。"""
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for stem, result in raw_results.items():
        grouped.setdefault(_microphone_average_prefix(stem), []).append((stem, result))

    averaged_results: Dict[str, Dict[str, Any]] = {}
    for prefix, items in grouped.items():
        items = sorted(items, key=lambda item: item[0])
        ref_freqs = np.asarray(items[0][1]["freqs"], dtype=float)
        psd_rows: List[np.ndarray] = []
        source_files: List[str] = []

        for _stem, result in items:
            freqs = np.asarray(result["freqs"], dtype=float)
            psd = np.asarray(result["psd"], dtype=float)
            source_files.append(str(result.get("source_file", _stem)))
            if freqs.shape == ref_freqs.shape and np.allclose(freqs, ref_freqs):
                psd_rows.append(psd)
            else:
                interp_psd = np.vstack([np.interp(ref_freqs, freqs, psd[channel, :]) for channel in range(psd.shape[0])])
                psd_rows.append(interp_psd)

        avg_psd = np.mean(np.stack(psd_rows, axis=0), axis=0)
        averaged_results[prefix] = _build_microphone_result_from_psd(
            condition_name=prefix,
            source_files=source_files,
            freqs=ref_freqs,
            psd=avg_psd,
            freq_min=freq_min,
            freq_max=freq_max,
            octave_denominator=octave_denominator,
        )
        print(f"  平均完成: {prefix} <- {', '.join(source_files)}")

    return averaged_results


def analyze_microphone_files(
    mat_files: List[Path],
    sample_rate: float = fs,
    desired_df: float = target_df,
    freq_min: float = MICROPHONE_FREQ_MIN,
    freq_max: float = MICROPHONE_FREQ_MAX,
    average_by_prefix: bool = False,
    octave_denominator: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """批量读取麦克风 mat 文件，返回每个工况的声压级分析结果。"""
    octave_denominator = _normalize_octave_denominator(octave_denominator)
    results: Dict[str, Dict[str, Any]] = {}
    for mat_file in mat_files:
        print("-" * 72)
        print(f"正在处理麦克风数据: {mat_file.parent.name}\\{mat_file.name}")
        try:
            results[mat_file.stem] = _compute_microphone_result_for_file(
                mat_file=mat_file,
                sample_rate=sample_rate,
                desired_df=desired_df,
                freq_min=freq_min,
                freq_max=freq_max,
                octave_denominator=octave_denominator,
            )
            print(f"  完成: {mat_file.name}")
        except Exception as exc:
            print(f"  warning: {mat_file.name} 麦克风处理失败，已跳过。原因: {exc}")
    if average_by_prefix and results:
        print("-" * 72)
        print("按文件名前缀进行麦克风重复数据能量平均：去掉最后的 _数字 后缀。")
        return _average_microphone_results_by_prefix(results, freq_min=freq_min, freq_max=freq_max, octave_denominator=octave_denominator)
    return results


def _resolve_microphone_condition_order(
    results: Dict[str, Dict[str, Any]],
    condition_order: Optional[List[str]],
) -> List[str]:
    return resolve_condition_order(results, condition_order)


def plot_microphone_fft_spl(
    results: Dict[str, Dict[str, Any]],
    mic_index: int,
    mic_name: str,
    freq_min: float = MICROPHONE_FREQ_MIN,
    freq_max: float = MICROPHONE_FREQ_MAX,
    condition_order: Optional[List[str]] = None,
    band_width_hz: Optional[float] = None,
) -> Optional[plt.Figure]:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    has_data = False
    plot_df: Optional[float] = None
    for condition_name in _resolve_microphone_condition_order(results, condition_order):
        result = results[condition_name]
        freqs = result["freqs"]
        centers, spl = _frequency_band_spl(
            freqs=freqs,
            psd=result["psd"],
            freq_min=freq_min,
            freq_max=freq_max,
            band_width_hz=band_width_hz,
        )
        if centers.size == 0 or spl.shape[1] == 0:
            continue
        if plot_df is None:
            if centers.size > 1:
                plot_df = float(np.nanmedian(np.diff(centers)))
            elif freqs.size > 1:
                plot_df = float(freqs[1] - freqs[0])
        ax.plot(
            centers,
            spl[mic_index, :],
            linewidth=1.4,
            label=get_display_condition_name(condition_name),
            color=get_condition_color(condition_name),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None
    df_label = f"{plot_df:g} Hz" if plot_df is not None and np.isfinite(plot_df) else "FFT"
    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("频率 / Hz")
    ax.set_ylabel(f"{df_label} 频带声压级 / dB SPL")
    ax.set_title(f"{mic_name} {freq_min:.0f}-{freq_max:.0f} Hz FFT声压级（df≈{df_label}）")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def plot_microphone_total_spl(
    results: Dict[str, Dict[str, Any]],
    mic_index: int,
    mic_name: str,
    freq_min: float = MICROPHONE_FREQ_MIN,
    freq_max: float = MICROPHONE_FREQ_MAX,
    condition_order: Optional[List[str]] = None,
) -> Optional[plt.Figure]:
    names = _resolve_microphone_condition_order(results, condition_order)
    if not names:
        return None
    values = [float(results[name]["total_spl"][mic_index]) for name in names]
    colors = [get_condition_color(name) or f"C{idx}" for idx, name in enumerate(names)]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(np.arange(len(names)), values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([get_display_condition_name(name) for name in names], rotation=25, ha="right")
    ax.set_ylabel("总声压级 / dB SPL")
    ax.set_title(f"{mic_name} {freq_min:.0f}-{freq_max:.0f} Hz 总声压级")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def plot_microphone_third_octave_spl(
    results: Dict[str, Dict[str, Any]],
    mic_index: int,
    mic_name: str,
    freq_min: float = MICROPHONE_FREQ_MIN,
    freq_max: float = MICROPHONE_FREQ_MAX,
    condition_order: Optional[List[str]] = None,
    octave_denominator: int = 3,
) -> Optional[plt.Figure]:
    octave_denominator = _normalize_octave_denominator(octave_denominator)
    octave_label = _octave_label(octave_denominator)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    has_data = False
    for condition_name in _resolve_microphone_condition_order(results, condition_order):
        result = results[condition_name]
        centers = result["third_octave_centers"]
        spl = result["third_octave_spl"]
        if centers.size == 0 or spl.shape[1] == 0:
            continue
        ax.plot(
            np.arange(centers.size),
            spl[mic_index, :],
            marker="o",
            linewidth=1.4,
            label=get_display_condition_name(condition_name),
            color=get_condition_color(condition_name),
        )
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    centers = next(result["third_octave_centers"] for result in results.values() if result["third_octave_centers"].size)
    labels = [f"{center:g}" for center in centers]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(f"{octave_label}中心频率 / Hz")
    ax.set_ylabel("声压级 / dB SPL")
    ax.set_title(f"{mic_name} {octave_label}声压级（与{freq_min:.0f}-{freq_max:.0f} Hz有交集）")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def save_microphone_outputs(
    results: Dict[str, Dict[str, Any]],
    plot_options: Dict[str, bool],
    save_dir: Path,
    image_format: str,
    dpi_value: int,
    show_after: bool,
    save_figures_flag: bool,
    plot_style: Optional[Dict[str, Any]] = None,
    freq_min: float = MICROPHONE_FREQ_MIN,
    freq_max: float = MICROPHONE_FREQ_MAX,
    fft_freq_min: Optional[float] = None,
    fft_freq_max: Optional[float] = None,
    mic_indices: Optional[List[int]] = None,
    octave_denominator: int = 3,
    fft_band_width_hz: Optional[float] = None,
) -> List[Path]:
    saved_paths: List[Path] = []
    generated_figures: List[plt.Figure] = []
    active_style = normalize_plot_style(plot_style)
    condition_order = list(active_style.get("condition_order") or [])

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = active_style["font_family_list"]
    plt.rcParams["axes.unicode_minus"] = False

    octave_denominator = _normalize_octave_denominator(octave_denominator)
    octave_label = _octave_label(octave_denominator)
    plotters = [
        ("fft_spl", "FFT声压级", plot_microphone_fft_spl),
        ("total_spl", "总声压级", plot_microphone_total_spl),
        ("third_octave", f"{octave_label}声压级", plot_microphone_third_octave_spl),
    ]

    selected_mic_indices = mic_indices if mic_indices is not None else list(range(len(MICROPHONE_NAMES)))
    selected_mic_indices = [idx for idx in selected_mic_indices if 0 <= idx < len(MICROPHONE_NAMES)]

    for mic_index in selected_mic_indices:
        mic_name = MICROPHONE_NAMES[mic_index]
        for option_key, filename_part, plotter in plotters:
            if not plot_options.get(option_key):
                continue
            plot_freq_min = fft_freq_min if option_key == "fft_spl" and fft_freq_min is not None else freq_min
            plot_freq_max = fft_freq_max if option_key == "fft_spl" and fft_freq_max is not None else freq_max
            extra_kwargs = {}
            if option_key == "third_octave":
                extra_kwargs["octave_denominator"] = octave_denominator
            elif option_key == "fft_spl":
                extra_kwargs["band_width_hz"] = fft_band_width_hz
            fig = plotter(
                results=results,
                mic_index=mic_index,
                mic_name=mic_name,
                freq_min=plot_freq_min,
                freq_max=plot_freq_max,
                condition_order=condition_order,
                **extra_kwargs,
            )
            if fig is None:
                continue
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, f"麦克风_{mic_name}_{filename_part}", image_format, dpi_value))
            generated_figures.append(fig)

    if show_after and generated_figures:
        plt.show()
    else:
        for fig in generated_figures:
            plt.close(fig)
    return saved_paths
