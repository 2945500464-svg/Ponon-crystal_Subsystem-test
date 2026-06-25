from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import apply_plot_style, normalize_plot_style
from .plotting import plot_acceleration_fft_comparison, plot_band_total_level_bar, plot_damping_rate_bar_for_results, plot_force_acceleration_frf_comparison, plot_force_fft_comparison, plot_force_psd_comparison, plot_input_accel_psd_comparison, plot_normalized_psd_ratio_heatmap, plot_six_sensor_psd_comparison, plot_transfer_loss_comparison, plot_weighted_transfer_rate_comparison, resolve_condition_order
from .utils import get_condition_plot_order, normalize_condition_label, sanitize_filename, sanitize_sheet_name

def export_to_excel(
    all_results: Dict[str, Dict[str, Any]],
    output_index: int,
    output_name: str,
    xlsx_dir: Path,
) -> Optional[Path]:
    """导出单个输出通道的所有工况传递损失到 xlsx。"""
    reference_freqs: Optional[np.ndarray] = None

    for result in all_results.values():
        tl_data = result["transfer_loss_db"]
        if output_index < tl_data.shape[0]:
            reference_freqs = result["freqs"]
            break

    if reference_freqs is None:
        return None

    table = pd.DataFrame({"Frequency_Hz": reference_freqs})

    for condition_name, result in all_results.items():
        tl_data = result["transfer_loss_db"]
        if output_index >= tl_data.shape[0]:
            continue

        freqs_current = result["freqs"]
        values_current = tl_data[output_index, :]

        if len(freqs_current) != len(reference_freqs) or not np.allclose(freqs_current, reference_freqs):
            print(f"  warning: {condition_name} 频率轴不一致，导出时已插值到参考频率轴")
            values_current = np.interp(reference_freqs, freqs_current, values_current)

        table[condition_name] = values_current

    xlsx_path = xlsx_dir / f"TL_{sanitize_filename(output_name)}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name=sanitize_sheet_name(output_name))

    return xlsx_path


def save_figure_as(fig: plt.Figure, save_dir: Path, filename: str, fmt: str, dpi_value: int) -> Path:
    """Save a matplotlib figure with selected format and dpi."""
    save_dir.mkdir(parents=True, exist_ok=True)
    clean_name = sanitize_filename(filename, "figure")
    path = save_dir / f"{clean_name}.{fmt.lower()}"
    kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
        kwargs["dpi"] = dpi_value
    fig.savefig(path, **kwargs)
    return path


def save_selected_outputs(
    all_results: Dict[str, Dict[str, Any]],
    layout_info: Dict[str, Any],
    plot_options: Dict[str, bool],
    save_dir: Path,
    image_format: str,
    dpi_value: int,
    freq_min: float,
    freq_max: float,
    show_after: bool,
    save_figures_flag: bool,
    export_excel_flag: bool,
    stat_point_mode: str = "four",
    stat_band_min: Optional[float] = None,
    stat_band_max: Optional[float] = None,
    plot_style: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    saved_paths: List[Path] = []
    generated_figures: List[plt.Figure] = []

    active_style = normalize_plot_style(plot_style)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = active_style["font_family_list"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["legend.fontsize"] = active_style["legend_fontsize"]
    condition_order = list(active_style.get("condition_order") or [])

    stat_band_min = freq_min if stat_band_min is None else stat_band_min
    stat_band_max = freq_max if stat_band_max is None else stat_band_max
    if stat_band_max <= stat_band_min:
        raise ValueError("统计频段上限必须大于下限。")
    print(f"统计图频段: {stat_band_min:g}-{stat_band_max:g} Hz")
    four_names = layout_info.get("four_names") or ["左前", "左后", "右前", "右后"]
    six_names = layout_info.get("six_names") or ["左前", "左后", "左中", "右前", "右后", "右中"]
    auto_names = layout_info.get("auto_stat_names") or (
        six_names if layout_info.get("auto_stat_mode") == "six" else four_names
    )
    if stat_point_mode == "auto":
        stat_names = auto_names
    else:
        stat_names = six_names if stat_point_mode == "six" else four_names
    stat_weights = [1.0] * len(stat_names)

    if plot_options.get("single_outputs"):
        max_outputs = max((result["transfer_loss_db"].shape[0] for result in all_results.values()), default=0)
        for output_index in range(max_outputs):
            output_name = f"Output_{output_index + 1}"
            for result in all_results.values():
                names = result.get("output_names", [])
                if output_index < len(names):
                    output_name = str(names[output_index])
                    break
            fig, _ = plot_transfer_loss_comparison(all_results, output_index, output_name, freq_min, freq_max, False, save_dir, condition_order=condition_order)
            if fig is not None:
                apply_plot_style(fig, active_style)
                if save_figures_flag:
                    saved_paths.append(save_figure_as(fig, save_dir, f"单输出_{output_index + 1}_{output_name}", image_format, dpi_value))
                generated_figures.append(fig)
            if export_excel_flag:
                path = export_to_excel(all_results, output_index, output_name, save_dir)
                if path is not None:
                    saved_paths.append(path)

    if plot_options.get("accel_fft"):
        sensor_names: List[str] = []
        seen_sensor_names = set()
        for condition_name in resolve_condition_order(all_results, condition_order):
            result = all_results[condition_name]
            for sensor_name in result.get("accel_fft_names", result.get("output_names", [])):
                sensor_name = str(sensor_name)
                if sensor_name and sensor_name not in seen_sensor_names:
                    sensor_names.append(sensor_name)
                    seen_sensor_names.add(sensor_name)

        for sensor_name in sensor_names:
            fig = plot_acceleration_fft_comparison(all_results, sensor_name, freq_min, freq_max, condition_order=condition_order)
            if fig is not None:
                apply_plot_style(fig, active_style)
                if save_figures_flag:
                    saved_paths.append(save_figure_as(fig, save_dir, f"加速度FFT_{sensor_name}", image_format, dpi_value))
                generated_figures.append(fig)

    if plot_options.get("six_sensor_psd"):
        sensor_names = []
        seen_sensor_names = set()
        for condition_name in resolve_condition_order(all_results, condition_order):
            result = all_results[condition_name]
            for sensor_name in result.get("six_sensor_psd_names", six_names):
                sensor_name = str(sensor_name)
                if sensor_name and sensor_name not in seen_sensor_names:
                    sensor_names.append(sensor_name)
                    seen_sensor_names.add(sensor_name)

        for sensor_name in sensor_names:
            fig = plot_six_sensor_psd_comparison(all_results, sensor_name, freq_min, freq_max, condition_order=condition_order)
            if fig is not None:
                apply_plot_style(fig, active_style)
                if save_figures_flag:
                    saved_paths.append(save_figure_as(fig, save_dir, f"六点PSD_{sensor_name}", image_format, dpi_value))
                generated_figures.append(fig)

    if plot_options.get("four_average"):
        fig, _ = plot_weighted_transfer_rate_comparison(all_results, four_names, [1.0] * len(four_names), freq_min, freq_max, False, save_dir, condition_order=condition_order)
        if fig is not None:
            if fig.axes:
                fig.axes[0].set_title("四点平均 PSD 传递率对比")
                fig.axes[0].set_ylabel("平均 PSD 传递率 / dB")
                fig.tight_layout()
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, "四点平均PSD传递率对比", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("six_average"):
        fig, _ = plot_weighted_transfer_rate_comparison(all_results, six_names, [1.0] * len(six_names), freq_min, freq_max, False, save_dir, condition_order=condition_order)
        if fig is not None:
            if fig.axes:
                fig.axes[0].set_title("六点平均 PSD 传递率对比")
                fig.axes[0].set_ylabel("平均 PSD 传递率 / dB")
                fig.tight_layout()
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, "六点平均PSD传递率对比", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("input_psd"):
        fig = plot_input_accel_psd_comparison(all_results, freq_min, freq_max, condition_order=condition_order)
        if fig is not None:
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, "输入点PSD对比", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("force_psd"):
        fig = plot_force_psd_comparison(all_results, freq_min, freq_max, condition_order=condition_order)
        if fig is not None:
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, "输入力PSD对比", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("force_fft"):
        fig = plot_force_fft_comparison(all_results, freq_min, freq_max, condition_order=condition_order)
        if fig is not None:
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, "力传感器FFT对比", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("frf"):
        fig, _ = plot_force_acceleration_frf_comparison(all_results, freq_min, freq_max, False, save_dir, condition_order=condition_order)
        if fig is not None:
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, "输入点加速度力FRF对比", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("heatmap_no_load"):
        fig = plot_normalized_psd_ratio_heatmap(
            all_results,
            baseline_label="no-load",
            condition_order=[name for name in condition_order if name in all_results and normalize_condition_label(name) != "no-load"] or [name for name in get_condition_plot_order(all_results) if normalize_condition_label(name) != "no-load"],
            title="频率-方案改善：输入归一化PSD比值，相对no-load",
            freq_min=freq_min,
            freq_max=freq_max,
        )
        if fig is not None:
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, "频率方案改善_相对no-load", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("heatmap_dva"):
        fig = plot_normalized_psd_ratio_heatmap(
            all_results,
            baseline_label="DVA",
            condition_order=[name for name in condition_order if name in all_results and normalize_condition_label(name) != "DVA"] or [name for name in get_condition_plot_order(all_results) if normalize_condition_label(name) != "DVA"],
            title="频率-方案改善：输入归一化PSD比值，相对DVA",
            freq_min=freq_min,
            freq_max=freq_max,
        )
        if fig is not None:
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, "频率方案改善_相对DVA", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("total_level"):
        fig = plot_band_total_level_bar(all_results, stat_names, stat_weights, stat_band_min, stat_band_max, condition_order=condition_order)
        if fig is not None:
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, f"{stat_band_min:.0f}-{stat_band_max:.0f}Hz总振级", image_format, dpi_value))
            generated_figures.append(fig)

    if plot_options.get("damping_rate"):
        fig = plot_damping_rate_bar_for_results(all_results, stat_names, stat_weights, stat_band_min, stat_band_max, condition_order=condition_order)
        if fig is not None:
            apply_plot_style(fig, active_style)
            if save_figures_flag:
                saved_paths.append(save_figure_as(fig, save_dir, f"{stat_band_min:.0f}-{stat_band_max:.0f}Hz平均减振率", image_format, dpi_value))
            generated_figures.append(fig)

    if show_after and generated_figures:
        plt.show()
    else:
        for fig in generated_figures:
            plt.close(fig)
    return saved_paths
