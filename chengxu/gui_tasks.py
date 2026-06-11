from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analysis import analyze_selected_files
from .config import RAW_DATA_DIR
from .exporters import save_selected_outputs
from .microphone_processing import analyze_microphone_files, save_microphone_outputs
from .plotting import build_average_results_between_dates
from .utils import normalize_condition_label
from .vehicle_processing import analyze_vehicle_files, save_vehicle_outputs, vehicle_condition_prefix


MODE_DISPLAY_TO_VALUE = {
    "单段分析": "single_segment",
    "分段平均": "segment_average",
}


@dataclass
class VibrationTaskConfig:
    mat_files: List[Path]
    input_mode: str
    freq_min: float
    freq_max: float
    desired_df: float
    duration_sec: float
    mode: str
    start_sec: float
    trim_start: float
    trim_end: float
    plot_options: Dict[str, bool]
    save_dir: Path
    image_format: str
    dpi_value: int
    show_after: bool
    save_figures: bool
    export_excel: bool
    plot_style: Dict[str, Any]


@dataclass
class PairedAverageTaskConfig:
    file_names: List[str]
    freq_min: float
    freq_max: float
    desired_df: float
    duration_sec: float
    mode: str
    start_sec: float
    trim_start: float
    trim_end: float
    plot_options: Dict[str, bool]
    save_dir: Path
    image_format: str
    dpi_value: int
    show_after: bool
    save_figures: bool
    export_excel: bool
    plot_style: Dict[str, Any]


@dataclass
class MicrophoneTaskConfig:
    mat_files: List[Path]
    plot_options: Dict[str, bool]
    save_dir: Path
    image_format: str
    dpi_value: int
    show_after: bool
    save_figures: bool
    plot_style: Dict[str, Any]
    fft_freq_min: float
    fft_freq_max: float
    total_freq_min: float
    total_freq_max: float
    mic_indices: List[int]
    octave_denominator: int
    average_by_prefix: bool


@dataclass
class Vehicle611TaskConfig:
    mat_files: List[Path]
    plot_options: Dict[str, bool]
    save_dir: Path
    image_format: str
    dpi_value: int
    show_after: bool
    save_figures: bool
    plot_style: Dict[str, Any]
    freq_min: float
    freq_max: float
    desired_df: float
    duration_sec: float
    mode: str
    start_sec: float
    trim_start: float
    trim_end: float
    mic_indices: List[int]
    octave_denominator: int
    average_by_condition: bool


@dataclass
class TaskRunResult:
    saved_paths: List[Path]
    result_count: int
    message: str


def _require_positive_range(freq_min: float, freq_max: float, label: str) -> None:
    if freq_min < 0:
        raise ValueError(f"{label}下限不能小于 0 Hz。")
    if freq_max <= freq_min:
        raise ValueError(f"{label}上限必须大于下限。")


def _selected_plot_count(plot_options: Dict[str, bool]) -> int:
    return sum(1 for enabled in plot_options.values() if enabled)


def _has_condition(names: List[str], normalized: str) -> bool:
    return any(normalize_condition_label(name) == normalized for name in names)


class VibrationTask:
    def __init__(self, config: VibrationTaskConfig) -> None:
        self.config = config

    def validate(self) -> None:
        cfg = self.config
        if not cfg.mat_files:
            raise ValueError("请至少选择一个 .mat 数据文件。")
        if _selected_plot_count(cfg.plot_options) == 0:
            raise ValueError("请至少选择一种出图类型。")
        _require_positive_range(cfg.freq_min, cfg.freq_max, "频率范围")
        if cfg.desired_df <= 0:
            raise ValueError("目标频率分辨率必须大于 0。")
        if cfg.duration_sec <= 0:
            raise ValueError("分析时长必须大于 0。")
        if cfg.dpi_value <= 0:
            raise ValueError("DPI 必须大于 0。")
        condition_names = [path.stem for path in cfg.mat_files]
        if (cfg.plot_options.get("damping_rate") or cfg.plot_options.get("heatmap_no_load")) and not _has_condition(condition_names, "no-load"):
            raise ValueError("所选数据缺少 no-load，无法计算相对 no-load 的评价图。")
        if cfg.plot_options.get("heatmap_dva") and not _has_condition(condition_names, "DVA"):
            raise ValueError("所选数据缺少 DVA，无法计算相对 DVA 的热力图。")

    def run(self) -> TaskRunResult:
        self.validate()
        cfg = self.config
        all_results, layout_info = analyze_selected_files(
            mat_files=cfg.mat_files,
            input_mode=cfg.input_mode,
            freq_min=cfg.freq_min,
            freq_max=cfg.freq_max,
            desired_df=cfg.desired_df,
            duration_sec=cfg.duration_sec,
            mode=cfg.mode,
            start_sec=cfg.start_sec,
            trim_start=cfg.trim_start,
            trim_end=cfg.trim_end,
            prefix_date=False,
        )
        if not all_results:
            raise ValueError("没有成功处理任何结构振动数据文件，请查看终端 warning 信息。")
        saved_paths = save_selected_outputs(
            all_results=all_results,
            layout_info=layout_info,
            plot_options=cfg.plot_options,
            save_dir=cfg.save_dir,
            image_format=cfg.image_format,
            dpi_value=cfg.dpi_value,
            freq_min=cfg.freq_min,
            freq_max=cfg.freq_max,
            show_after=cfg.show_after,
            save_figures_flag=cfg.save_figures,
            export_excel_flag=cfg.export_excel,
            stat_point_mode="auto",
            plot_style=cfg.plot_style,
        )
        return TaskRunResult(saved_paths=saved_paths, result_count=len(all_results), message="副车架振动数据处理完成。")


class PairedAverageTask:
    def __init__(self, config: PairedAverageTaskConfig) -> None:
        self.config = config

    def validate(self) -> None:
        cfg = self.config
        if not cfg.file_names:
            raise ValueError("请至少选择一个 5.22 与 5.23 同名 .mat 文件。")
        if _selected_plot_count(cfg.plot_options) == 0:
            raise ValueError("请至少选择一种出图类型。")
        _require_positive_range(cfg.freq_min, cfg.freq_max, "频率范围")
        if cfg.desired_df <= 0:
            raise ValueError("目标频率分辨率必须大于 0。")
        if cfg.duration_sec <= 0:
            raise ValueError("分析时长必须大于 0。")
        if cfg.dpi_value <= 0:
            raise ValueError("DPI 必须大于 0。")
        missing = [
            name for name in cfg.file_names
            if not (RAW_DATA_DIR / "5.22" / name).exists() or not (RAW_DATA_DIR / "5.23" / name).exists()
        ]
        if missing:
            raise ValueError("以下文件没有在 5.22 和 5.23 中同时出现：\n" + "\n".join(missing[:20]))
        condition_names = [Path(name).stem for name in cfg.file_names]
        if (cfg.plot_options.get("damping_rate") or cfg.plot_options.get("heatmap_no_load")) and not _has_condition(condition_names, "no-load"):
            raise ValueError("所选同名数据缺少 no-load，无法计算相对 no-load 的评价图。")
        if cfg.plot_options.get("heatmap_dva") and not _has_condition(condition_names, "DVA"):
            raise ValueError("所选同名数据缺少 DVA，无法计算相对 DVA 的热力图。")

    def run(self) -> TaskRunResult:
        self.validate()
        cfg = self.config
        analysis_files = [RAW_DATA_DIR / "5.22" / name for name in cfg.file_names]
        analysis_files.extend(RAW_DATA_DIR / "5.23" / name for name in cfg.file_names)
        all_results, layout_info = analyze_selected_files(
            mat_files=analysis_files,
            input_mode="自动",
            freq_min=cfg.freq_min,
            freq_max=cfg.freq_max,
            desired_df=cfg.desired_df,
            duration_sec=cfg.duration_sec,
            mode=cfg.mode,
            start_sec=cfg.start_sec,
            trim_start=cfg.trim_start,
            trim_end=cfg.trim_end,
            prefix_date=True,
        )
        averaged_results = build_average_results_between_dates(all_results, "5.22", "5.23")
        ordered_average_names = [f"平均 {Path(name).stem}" for name in cfg.file_names]
        averaged_results = {
            name: averaged_results[name]
            for name in ordered_average_names
            if name in averaged_results
        }
        if not averaged_results:
            raise ValueError("没有生成任何 5.22/5.23 同名平均结果，请检查两边数据结构和文件名。")

        saved_paths = save_selected_outputs(
            all_results=averaged_results,
            layout_info=layout_info,
            plot_options=cfg.plot_options,
            save_dir=cfg.save_dir,
            image_format=cfg.image_format,
            dpi_value=cfg.dpi_value,
            freq_min=cfg.freq_min,
            freq_max=cfg.freq_max,
            show_after=cfg.show_after,
            save_figures_flag=cfg.save_figures,
            export_excel_flag=cfg.export_excel,
            stat_point_mode="auto",
            plot_style=cfg.plot_style,
        )
        return TaskRunResult(saved_paths=saved_paths, result_count=len(averaged_results), message="5.22 + 5.23 同名平均处理完成。")


class MicrophoneTask:
    def __init__(self, config: MicrophoneTaskConfig) -> None:
        self.config = config

    def validate(self) -> None:
        cfg = self.config
        if not cfg.mat_files:
            raise ValueError("请至少选择一个麦克风 .mat 文件。")
        if _selected_plot_count(cfg.plot_options) == 0:
            raise ValueError("请至少选择一种麦克风出图类型。")
        if not cfg.mic_indices:
            raise ValueError("请至少选择一个麦克风位置。")
        _require_positive_range(cfg.fft_freq_min, cfg.fft_freq_max, "FFT 声压级频率范围")
        _require_positive_range(cfg.total_freq_min, cfg.total_freq_max, "总声压级/倍频程频率范围")
        if cfg.dpi_value <= 0:
            raise ValueError("DPI 必须大于 0。")
        if cfg.octave_denominator not in {3, 6, 12, 24}:
            raise ValueError("倍频程类型只支持 1/3、1/6、1/12、1/24。")

    def run(self) -> TaskRunResult:
        self.validate()
        cfg = self.config
        results = analyze_microphone_files(
            mat_files=cfg.mat_files,
            freq_min=cfg.total_freq_min,
            freq_max=cfg.total_freq_max,
            average_by_prefix=cfg.average_by_prefix,
            octave_denominator=cfg.octave_denominator,
        )
        if not results:
            raise ValueError("没有成功处理任何麦克风数据文件，请查看终端 warning 信息。")
        saved_paths = save_microphone_outputs(
            results=results,
            plot_options=cfg.plot_options,
            save_dir=cfg.save_dir,
            image_format=cfg.image_format,
            dpi_value=cfg.dpi_value,
            show_after=cfg.show_after,
            save_figures_flag=cfg.save_figures,
            plot_style=cfg.plot_style,
            freq_min=cfg.total_freq_min,
            freq_max=cfg.total_freq_max,
            fft_freq_min=cfg.fft_freq_min,
            fft_freq_max=cfg.fft_freq_max,
            mic_indices=cfg.mic_indices,
            octave_denominator=cfg.octave_denominator,
        )
        return TaskRunResult(saved_paths=saved_paths, result_count=len(results), message="麦克风声学数据处理完成。")


class Vehicle611Task:
    def __init__(self, config: Vehicle611TaskConfig) -> None:
        self.config = config

    def validate(self) -> None:
        cfg = self.config
        if not cfg.mat_files:
            raise ValueError("请至少选择一个 6.11 实车 .mat 数据文件。")
        if _selected_plot_count(cfg.plot_options) == 0:
            raise ValueError("请至少选择一种 6.11 实车出图类型。")
        if not cfg.mic_indices:
            raise ValueError("请至少选择一个麦克风位置。")
        _require_positive_range(cfg.freq_min, cfg.freq_max, "6.11 分析频率范围")
        if cfg.desired_df <= 0:
            raise ValueError("目标频率分辨率必须大于 0。")
        if cfg.duration_sec <= 0:
            raise ValueError("分析时长必须大于 0。")
        if cfg.dpi_value <= 0:
            raise ValueError("DPI 必须大于 0。")
        if cfg.octave_denominator not in {3, 6, 12, 24}:
            raise ValueError("倍频程类型只支持 1/3、1/6、1/12、1/24。")
        condition_names = [
            vehicle_condition_prefix(path.stem) if cfg.average_by_condition else path.stem
            for path in cfg.mat_files
        ]
        if cfg.plot_options.get("improvement_scatter") and not _has_condition(condition_names, "no-load"):
            raise ValueError("频段改善散点图需要选择 no-load 基准数据。")

    def run(self) -> TaskRunResult:
        self.validate()
        cfg = self.config
        results = analyze_vehicle_files(
            mat_files=cfg.mat_files,
            freq_min=cfg.freq_min,
            freq_max=cfg.freq_max,
            desired_df=cfg.desired_df,
            duration_sec=cfg.duration_sec,
            mode=cfg.mode,
            start_sec=cfg.start_sec,
            trim_start=cfg.trim_start,
            trim_end=cfg.trim_end,
            octave_denominator=cfg.octave_denominator,
            average_by_condition=cfg.average_by_condition,
        )
        if not results:
            raise ValueError("没有成功处理任何 6.11 实车数据文件，请查看终端 warning 信息。")
        saved_paths = save_vehicle_outputs(
            results=results,
            plot_options=cfg.plot_options,
            save_dir=cfg.save_dir,
            image_format=cfg.image_format,
            dpi_value=cfg.dpi_value,
            show_after=cfg.show_after,
            save_figures_flag=cfg.save_figures,
            plot_style=cfg.plot_style,
            freq_min=cfg.freq_min,
            freq_max=cfg.freq_max,
            mic_indices=cfg.mic_indices,
            octave_denominator=cfg.octave_denominator,
        )
        return TaskRunResult(saved_paths=saved_paths, result_count=len(results), message="6.11 实车声振数据处理完成。")
