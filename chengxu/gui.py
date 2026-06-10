from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import (
    FIGURES_DIR,
    GUI_THEME_PATH,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    analysis_duration_sec,
    analysis_mode,
    ensure_directories,
    figure_save_dpi,
    fmax,
    fmin,
    legend_fontsize,
    require_runtime_dependencies,
    show_figures,
    start_time_sec,
    target_df,
    trim_end_sec,
    trim_start_sec,
)
from .data_format import get_input_channel_index_for_file, load_data_format_config
from .gui_tasks import (
    MODE_DISPLAY_TO_VALUE,
    MicrophoneTask,
    MicrophoneTaskConfig,
    PairedAverageTask,
    PairedAverageTaskConfig,
    VibrationTask,
    VibrationTaskConfig,
)
from .microphone_processing import MICROPHONE_FREQ_MAX, MICROPHONE_FREQ_MIN
from .utils import get_condition_color, normalize_condition_label


WORKFLOW_VIBRATION = "副车架振动数据处理"
WORKFLOW_PAIRED = "5.22 + 5.23 同名平均"
WORKFLOW_MICROPHONE = "麦克风声学数据处理"
WORKFLOWS = [WORKFLOW_VIBRATION, WORKFLOW_PAIRED, WORKFLOW_MICROPHONE]

STEPS = ["选择数据", "选择图型", "设置参数", "图形样式", "确认运行"]

STRUCTURE_PLOT_GROUPS = {
    "基础曲线": [
        ("单输出传递率", "single_outputs"),
        ("四点平均传递率", "four_average"),
        ("六点平均传递率", "six_average"),
        ("输入 PSD", "input_psd"),
        ("输入力 PSD", "force_psd"),
        ("输入点 FRF", "frf"),
    ],
    "评价图": [
        ("归一化热力图：相对 no-load", "heatmap_no_load"),
        ("归一化热力图：相对 DVA", "heatmap_dva"),
        ("总振级", "total_level"),
        ("平均减振率", "damping_rate"),
    ],
}

MICROPHONE_PLOTS = [
    ("FFT 声压级", "fft_spl"),
    ("总声压级", "total_spl"),
    ("分数倍频程声压级", "third_octave"),
]

STYLE_PRESETS: Dict[str, Dict[str, str]] = {
    "论文风格": {
        "font": "Microsoft YaHei",
        "title": "14",
        "label": "12",
        "tick": "10",
        "legend": "10",
        "annotation": "9",
        "line_width": "1.6",
        "line_style": "-",
        "alpha": "1.0",
        "brightness": "1.0",
        "grid": "0.35",
        "width": "7.2",
        "height": "4.6",
    },
    "PPT汇报": {
        "font": "Microsoft YaHei UI",
        "title": "18",
        "label": "15",
        "tick": "12",
        "legend": "13",
        "annotation": "11",
        "line_width": "2.2",
        "line_style": "-",
        "alpha": "1.0",
        "brightness": "1.0",
        "grid": "0.32",
        "width": "10.5",
        "height": "5.8",
    },
    "屏幕查看": {
        "font": "Microsoft YaHei UI",
        "title": "15",
        "label": "13",
        "tick": "11",
        "legend": "11",
        "annotation": "10",
        "line_width": "1.8",
        "line_style": "-",
        "alpha": "1.0",
        "brightness": "1.0",
        "grid": "0.45",
        "width": "",
        "height": "",
    },
    "黑白打印": {
        "font": "SimSun",
        "title": "14",
        "label": "12",
        "tick": "10",
        "legend": "10",
        "annotation": "9",
        "line_width": "1.8",
        "line_style": "-",
        "alpha": "1.0",
        "brightness": "0.85",
        "grid": "0.25",
        "width": "7.2",
        "height": "4.6",
    },
}

LINE_WIDTH_OPTIONS = ["默认", "0.8", "1.0", "1.2", "1.5", "1.8", "2.0", "2.5", "3.0", "4.0"]
BRIGHTNESS_OPTIONS = ["默认", "0.60", "0.75", "0.90", "1.00", "1.15", "1.30", "1.50"]
LINE_STYLE_OPTIONS = ["默认", "实线", "虚线", "点划线", "点线"]
LINE_STYLE_VALUES = {"默认": "", "实线": "-", "虚线": "--", "点划线": "-.", "点线": ":"}
PALETTE = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#1b9e77"]


@dataclass
class DataItem:
    key: str
    display_label: str
    date_label: str
    paths: List[Path]
    layout_desc: str
    input_desc: str
    status: str
    stem_for_style: str


def launch_analysis_gui() -> None:
    try:
        import customtkinter as ctk
        from tkinter import colorchooser, filedialog, messagebox
    except Exception as exc:
        from tkinter import messagebox

        messagebox.showerror("缺少依赖", f"需要安装 customtkinter。\n请在程序目录执行: pip install -r requirements.txt\n\n{exc}")
        return

    ensure_directories()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "xlsx").mkdir(parents=True, exist_ok=True)

    ctk.set_appearance_mode("light")
    try:
        if GUI_THEME_PATH.exists():
            ctk.set_default_color_theme(str(GUI_THEME_PATH))
        else:
            ctk.set_default_color_theme("blue")
    except Exception:
        ctk.set_default_color_theme("blue")

    colors = {
        "bg": "#f5f7fb",
        "panel": "#ffffff",
        "card": "#f8fafc",
        "line": "#d7dee8",
        "text": "#172033",
        "muted": "#64748b",
        "accent": "#2563eb",
        "accent_dark": "#1d4ed8",
        "soft": "#eaf1ff",
        "danger": "#dc2626",
        "success": "#15803d",
    }
    control_height = 56
    compact_height = 48
    option_font = ("Microsoft YaHei UI", 16)
    compact_option_font = ("Microsoft YaHei UI", 15)

    app = ctk.CTk()
    app.title("AITO 数据处理程序")
    app.geometry("1320x820")
    app.minsize(1180, 760)
    try:
        icon_path = PROJECT_ROOT / "assets" / "aito_data_processing.ico"
        if icon_path.exists():
            app.iconbitmap(str(icon_path))
    except Exception:
        pass

    app.grid_columnconfigure(0, weight=0)
    app.grid_columnconfigure(1, weight=1)
    app.grid_rowconfigure(1, weight=1)
    app.configure(fg_color=colors["bg"])

    workflow_var = ctk.StringVar(value=WORKFLOW_VIBRATION)
    step_var = ctk.StringVar(value=STEPS[0])
    folder_var = ctk.StringVar(value="")
    search_var = ctk.StringVar(value="")
    status_var = ctk.StringVar(value="请选择工作流和数据。")
    summary_var = ctk.StringVar(value="")

    # Structure analysis parameters.
    fmin_var = ctk.StringVar(value=str(fmin))
    fmax_var = ctk.StringVar(value=str(fmax))
    df_var = ctk.StringVar(value=str(target_df))
    duration_var = ctk.StringVar(value=str(analysis_duration_sec))
    mode_var = ctk.StringVar(value="单段分析" if analysis_mode == "single_segment" else "分段平均")
    start_var = ctk.StringVar(value=str(start_time_sec))
    trim_start_var = ctk.StringVar(value=str(trim_start_sec))
    trim_end_var = ctk.StringVar(value=str(trim_end_sec))
    input_mode_var = ctk.StringVar(value="自动")

    # Output parameters.
    save_figures_var = ctk.BooleanVar(value=False)
    show_figures_var = ctk.BooleanVar(value=show_figures)
    export_excel_var = ctk.BooleanVar(value=False)
    format_var = ctk.StringVar(value="png")
    dpi_var = ctk.StringVar(value=str(figure_save_dpi))
    save_dir_var = ctk.StringVar(value=str(FIGURES_DIR))

    # Microphone parameters.
    mic_fft_min_var = ctk.StringVar(value=str(MICROPHONE_FREQ_MIN))
    mic_fft_max_var = ctk.StringVar(value=str(MICROPHONE_FREQ_MAX))
    mic_total_min_var = ctk.StringVar(value=str(MICROPHONE_FREQ_MIN))
    mic_total_max_var = ctk.StringVar(value=str(MICROPHONE_FREQ_MAX))
    mic_octave_var = ctk.StringVar(value="1/3 倍频程")
    mic_average_var = ctk.BooleanVar(value=False)
    mic_position_vars = {0: ctk.BooleanVar(value=True), 1: ctk.BooleanVar(value=True), 2: ctk.BooleanVar(value=True)}

    structure_plot_vars = {
        key: ctk.BooleanVar(value=(key == "four_average"))
        for group in STRUCTURE_PLOT_GROUPS.values()
        for _label, key in group
    }
    paired_plot_vars = {
        key: ctk.BooleanVar(value=(key == "six_average"))
        for group in STRUCTURE_PLOT_GROUPS.values()
        for _label, key in group
    }
    mic_plot_vars = {
        "fft_spl": ctk.BooleanVar(value=True),
        "total_spl": ctk.BooleanVar(value=True),
        "third_octave": ctk.BooleanVar(value=True),
    }

    # Style variables.
    style_preset_var = ctk.StringVar(value="PPT汇报")
    font_var = ctk.StringVar(value="Microsoft YaHei UI")
    title_size_var = ctk.StringVar(value="18")
    label_size_var = ctk.StringVar(value="15")
    tick_size_var = ctk.StringVar(value="12")
    legend_size_var = ctk.StringVar(value=str(legend_fontsize))
    annotation_size_var = ctk.StringVar(value="10")
    line_width_var = ctk.StringVar(value="2.0")
    line_style_var = ctk.StringVar(value="实线")
    line_alpha_var = ctk.StringVar(value="1.0")
    brightness_var = ctk.StringVar(value="1.0")
    grid_alpha_var = ctk.StringVar(value="0.35")
    figure_width_var = ctk.StringVar(value="")
    figure_height_var = ctk.StringVar(value="")
    title_override_var = ctk.StringVar(value="")
    title_prefix_var = ctk.StringVar(value="")
    title_suffix_var = ctk.StringVar(value="")
    xlabel_var = ctk.StringVar(value="")
    ylabel_var = ctk.StringVar(value="")

    data_items: Dict[str, DataItem] = {}
    data_selected_vars: Dict[str, Any] = {}
    data_display_name_vars: Dict[str, Any] = {}
    style_color_values: Dict[str, str] = {}
    style_display_name_vars: Dict[str, Any] = {}
    style_line_width_vars: Dict[str, Any] = {}
    style_line_style_vars: Dict[str, Any] = {}
    style_brightness_vars: Dict[str, Any] = {}
    style_order: List[str] = []
    style_swatch_widgets: Dict[str, Any] = {}

    sidebar = ctk.CTkFrame(app, width=280, fg_color=colors["panel"], border_color=colors["line"], border_width=1, corner_radius=18)
    sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(18, 10), pady=18)
    sidebar.grid_propagate(False)
    sidebar.grid_columnconfigure(0, weight=1)

    main = ctk.CTkFrame(app, fg_color=colors["panel"], border_color=colors["line"], border_width=1, corner_radius=18)
    main.grid(row=0, column=1, sticky="nsew", padx=(0, 18), pady=(18, 10))
    main.grid_columnconfigure(0, weight=1)
    main.grid_rowconfigure(2, weight=1)

    footer = ctk.CTkFrame(app, fg_color=colors["panel"], border_color=colors["line"], border_width=1, corner_radius=18)
    footer.grid(row=1, column=1, sticky="ew", padx=(0, 18), pady=(0, 18))
    footer.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(sidebar, text="AITO\n数据处理程序", font=("Microsoft YaHei UI", 26, "bold"), text_color=colors["text"], justify="left").grid(row=0, column=0, sticky="w", padx=22, pady=(24, 8))
    ctk.CTkLabel(sidebar, text="按任务选择数据、图型和样式", font=("Microsoft YaHei UI", 13), text_color=colors["muted"], justify="left").grid(row=1, column=0, sticky="w", padx=22, pady=(0, 20))

    workflow_buttons: Dict[str, Any] = {}
    step_buttons: Dict[str, Any] = {}
    content_frame = ctk.CTkFrame(main, fg_color="transparent")
    content_frame.grid(row=2, column=0, sticky="nsew", padx=18, pady=(8, 18))
    content_frame.grid_columnconfigure(0, weight=1)
    content_frame.grid_rowconfigure(0, weight=1)
    content_frame.grid_rowconfigure(0, weight=1)

    def as_float(var: Any, label: str) -> float:
        try:
            return float(str(var.get()).strip())
        except ValueError as exc:
            raise ValueError(f"{label}必须填写为数字。") from exc

    def as_int(var: Any, label: str) -> int:
        value = int(round(as_float(var, label)))
        if value <= 0:
            raise ValueError(f"{label}必须大于 0。")
        return value

    def normalize_color(color: str) -> str:
        try:
            import matplotlib.colors as mcolors

            return mcolors.to_hex(mcolors.to_rgba(color), keep_alpha=False)
        except Exception:
            return "#1f77b4"

    def default_color_for_key(key: str, index: int = 0) -> str:
        fixed = get_condition_color(key)
        if fixed:
            return normalize_color(fixed)
        label = normalize_condition_label(key)
        fixed = get_condition_color(label)
        if fixed:
            return normalize_color(fixed)
        return PALETTE[index % len(PALETTE)]

    def data_folders() -> List[str]:
        if not RAW_DATA_DIR.exists():
            return []
        return sorted(path.name for path in RAW_DATA_DIR.iterdir() if path.is_dir())

    def current_workflow() -> str:
        return workflow_var.get() or WORKFLOW_VIBRATION

    def current_plot_vars() -> Dict[str, Any]:
        if current_workflow() == WORKFLOW_PAIRED:
            return paired_plot_vars
        if current_workflow() == WORKFLOW_MICROPHONE:
            return mic_plot_vars
        return structure_plot_vars

    def selected_data_items() -> List[DataItem]:
        selected = [item for key, item in data_items.items() if bool(data_selected_vars.get(key, ctk.BooleanVar(value=False)).get())]
        selected_keys = {item.key for item in selected}
        ordered = [data_items[key] for key in style_order if key in selected_keys and key in data_items]
        ordered.extend(item for item in selected if item.key not in {existing.key for existing in ordered})
        return ordered

    def selected_data_paths() -> List[Path]:
        paths: List[Path] = []
        for item in selected_data_items():
            paths.extend(item.paths)
        return paths

    def folder_layout_desc(folder: str) -> str:
        config = load_data_format_config().get(folder, {})
        layout = str(config.get("layout") or "auto")
        side = str(config.get("side") or "auto")
        return f"{layout} / {side}"

    def input_desc_for_path(path: Path) -> str:
        try:
            return f"Data第{get_input_channel_index_for_file(path) + 1}行"
        except Exception:
            return "自动"

    def rebuild_data_items(reset_selection: bool = True) -> None:
        data_items.clear()
        workflow = current_workflow()
        folders = data_folders()

        if workflow == WORKFLOW_PAIRED:
            left_dir = RAW_DATA_DIR / "5.22"
            right_dir = RAW_DATA_DIR / "5.23"
            left_names = {path.name for path in left_dir.glob("*.mat")} if left_dir.is_dir() else set()
            right_names = {path.name for path in right_dir.glob("*.mat")} if right_dir.is_dir() else set()
            for name in sorted(left_names & right_names):
                stem = Path(name).stem
                data_items[name] = DataItem(
                    key=name,
                    display_label=name,
                    date_label="5.22 + 5.23",
                    paths=[left_dir / name, right_dir / name],
                    layout_desc="10通道同名平均",
                    input_desc="5.22: Data第2行；5.23: Data第1行",
                    status="有效",
                    stem_for_style=f"平均 {stem}",
                )
        elif workflow == WORKFLOW_MICROPHONE:
            mic_dir = RAW_DATA_DIR / "5.31"
            for path in sorted(mic_dir.glob("*.mat")) if mic_dir.is_dir() else []:
                data_items[path.name] = DataItem(
                    key=path.name,
                    display_label=path.name,
                    date_label="5.31",
                    paths=[path],
                    layout_desc="3通道麦克风",
                    input_desc="主驾驶 / 中排 / 后排",
                    status="待处理",
                    stem_for_style=path.stem,
                )
        else:
            folder = folder_var.get() or (folders[0] if folders else "")
            if folder and folder_var.get() != folder:
                folder_var.set(folder)
            data_dir = RAW_DATA_DIR / folder
            for path in sorted(data_dir.glob("*.mat")) if data_dir.is_dir() else []:
                data_items[path.name] = DataItem(
                    key=path.name,
                    display_label=path.name,
                    date_label=folder,
                    paths=[path],
                    layout_desc=folder_layout_desc(folder),
                    input_desc=input_desc_for_path(path),
                    status="待处理",
                    stem_for_style=path.stem,
                )

        for key, item in data_items.items():
            if key not in data_selected_vars:
                data_selected_vars[key] = ctk.BooleanVar(value=False)
            elif reset_selection:
                data_selected_vars[key].set(False)
            if key not in data_display_name_vars:
                data_display_name_vars[key] = ctk.StringVar(value=Path(item.display_label).stem)
        if reset_selection:
            style_order.clear()

    def selected_style_targets() -> List[Dict[str, Any]]:
        workflow = current_workflow()
        targets: Dict[str, Dict[str, Any]] = {}
        for item in selected_data_items():
            if workflow == WORKFLOW_MICROPHONE and bool(mic_average_var.get()):
                stem = item.paths[0].stem
                parts = stem.rsplit("_", 1)
                key = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
                default_name = key
            elif workflow == WORKFLOW_PAIRED:
                key = item.stem_for_style
                default_name = str(data_display_name_vars.get(item.key, ctk.StringVar(value=key[3:] if key.startswith("平均 ") else key)).get() or "").strip() or (key[3:] if key.startswith("平均 ") else key)
            else:
                key = item.paths[0].stem
                default_name = str(data_display_name_vars.get(item.key, ctk.StringVar(value=key)).get() or "").strip() or key
            if key not in targets:
                targets[key] = {"key": key, "sources": [], "default_name": default_name}
            targets[key]["sources"].append(item.display_label)

        target_keys = set(targets)
        style_order[:] = [key for key in style_order if key in target_keys]
        for key in targets:
            if key not in style_order:
                style_order.append(key)
        return [targets[key] for key in style_order if key in targets]

    def style_var_for(mapping: Dict[str, Any], key: str, default: str) -> Any:
        if key not in mapping:
            mapping[key] = ctk.StringVar(value=default)
        return mapping[key]

    def build_plot_style() -> Dict[str, Any]:
        targets = selected_style_targets()
        color_map: Dict[str, str] = {}
        display_map: Dict[str, str] = {}
        line_width_map: Dict[str, float] = {}
        line_style_map: Dict[str, str] = {}
        brightness_map: Dict[str, float] = {}
        condition_order: List[str] = []

        for index, target in enumerate(targets):
            key = target["key"]
            condition_order.append(key)
            color_map[key] = style_color_values.get(key) or default_color_for_key(key, index)
            display_name = str(style_var_for(style_display_name_vars, key, target["default_name"]).get() or "").strip()
            if display_name:
                display_map[key] = display_name
            line_text = str(style_var_for(style_line_width_vars, key, "默认").get() or "默认")
            if line_text != "默认":
                line_width_map[key] = float(line_text)
            line_style_text = str(style_var_for(style_line_style_vars, key, "默认").get() or "默认")
            if line_style_text != "默认" and LINE_STYLE_VALUES.get(line_style_text):
                line_style_map[key] = LINE_STYLE_VALUES[line_style_text]
            brightness_text = str(style_var_for(style_brightness_vars, key, "默认").get() or "默认")
            if brightness_text != "默认":
                brightness_map[key] = float(brightness_text)

        return {
            "font_family": font_var.get(),
            "title_fontsize": title_size_var.get(),
            "label_fontsize": label_size_var.get(),
            "tick_fontsize": tick_size_var.get(),
            "legend_fontsize": legend_size_var.get(),
            "annotation_fontsize": annotation_size_var.get(),
            "line_width": line_width_var.get(),
            "line_style": LINE_STYLE_VALUES.get(line_style_var.get(), "-") or "-",
            "line_alpha": line_alpha_var.get(),
            "brightness": brightness_var.get(),
            "grid_alpha": grid_alpha_var.get(),
            "figure_width": figure_width_var.get(),
            "figure_height": figure_height_var.get(),
            "title_override": title_override_var.get(),
            "title_prefix": title_prefix_var.get(),
            "title_suffix": title_suffix_var.get(),
            "xlabel_override": xlabel_var.get(),
            "ylabel_override": ylabel_var.get(),
            "color_map": color_map,
            "display_name_map": display_map,
            "line_width_map": line_width_map,
            "line_style_map": line_style_map,
            "brightness_map": brightness_map,
            "condition_order": condition_order,
        }

    def plot_options_for_current_workflow() -> Dict[str, bool]:
        return {key: bool(var.get()) for key, var in current_plot_vars().items()}

    def selected_plot_names() -> List[str]:
        if current_workflow() == WORKFLOW_MICROPHONE:
            names = {key: label for label, key in MICROPHONE_PLOTS}
            return [names[key] for key, var in mic_plot_vars.items() if bool(var.get())]
        names = {key: label for group in STRUCTURE_PLOT_GROUPS.values() for label, key in group}
        return [names[key] for key, var in current_plot_vars().items() if bool(var.get())]

    def selected_microphone_indices() -> List[int]:
        return [idx for idx, var in mic_position_vars.items() if bool(var.get())]

    def microphone_octave_denominator() -> int:
        text = mic_octave_var.get()
        if "1/24" in text:
            return 24
        if "1/12" in text:
            return 12
        if "1/6" in text:
            return 6
        return 3

    def selected_file_names_for_paired() -> List[str]:
        return [item.key for item in selected_data_items()]

    def build_vibration_task() -> VibrationTask:
        return VibrationTask(
            VibrationTaskConfig(
                mat_files=selected_data_paths(),
                input_mode=input_mode_var.get(),
                freq_min=as_float(fmin_var, "频率下限"),
                freq_max=as_float(fmax_var, "频率上限"),
                desired_df=as_float(df_var, "目标频率分辨率"),
                duration_sec=as_float(duration_var, "分析时长"),
                mode=MODE_DISPLAY_TO_VALUE.get(mode_var.get(), "single_segment"),
                start_sec=as_float(start_var, "起始时间"),
                trim_start=as_float(trim_start_var, "前裁时间"),
                trim_end=as_float(trim_end_var, "后裁时间"),
                plot_options=plot_options_for_current_workflow(),
                save_dir=Path(save_dir_var.get()).expanduser(),
                image_format=format_var.get().lower(),
                dpi_value=as_int(dpi_var, "DPI"),
                show_after=bool(show_figures_var.get()),
                save_figures=bool(save_figures_var.get()),
                export_excel=bool(export_excel_var.get()),
                plot_style=build_plot_style(),
            )
        )

    def build_paired_task() -> PairedAverageTask:
        return PairedAverageTask(
            PairedAverageTaskConfig(
                file_names=selected_file_names_for_paired(),
                freq_min=as_float(fmin_var, "频率下限"),
                freq_max=as_float(fmax_var, "频率上限"),
                desired_df=as_float(df_var, "目标频率分辨率"),
                duration_sec=as_float(duration_var, "分析时长"),
                mode=MODE_DISPLAY_TO_VALUE.get(mode_var.get(), "single_segment"),
                start_sec=as_float(start_var, "起始时间"),
                trim_start=as_float(trim_start_var, "前裁时间"),
                trim_end=as_float(trim_end_var, "后裁时间"),
                plot_options=plot_options_for_current_workflow(),
                save_dir=Path(save_dir_var.get()).expanduser(),
                image_format=format_var.get().lower(),
                dpi_value=as_int(dpi_var, "DPI"),
                show_after=bool(show_figures_var.get()),
                save_figures=bool(save_figures_var.get()),
                export_excel=bool(export_excel_var.get()),
                plot_style=build_plot_style(),
            )
        )

    def build_microphone_task() -> MicrophoneTask:
        return MicrophoneTask(
            MicrophoneTaskConfig(
                mat_files=selected_data_paths(),
                plot_options=plot_options_for_current_workflow(),
                save_dir=Path(save_dir_var.get()).expanduser(),
                image_format=format_var.get().lower(),
                dpi_value=as_int(dpi_var, "DPI"),
                show_after=bool(show_figures_var.get()),
                save_figures=bool(save_figures_var.get()),
                plot_style=build_plot_style(),
                fft_freq_min=as_float(mic_fft_min_var, "FFT 频率下限"),
                fft_freq_max=as_float(mic_fft_max_var, "FFT 频率上限"),
                total_freq_min=as_float(mic_total_min_var, "总声压级频率下限"),
                total_freq_max=as_float(mic_total_max_var, "总声压级频率上限"),
                mic_indices=selected_microphone_indices(),
                octave_denominator=microphone_octave_denominator(),
                average_by_prefix=bool(mic_average_var.get()),
            )
        )

    def update_summary() -> None:
        selected_count = len(selected_data_items())
        plots = "、".join(selected_plot_names()) or "未选图型"
        save_state = "保存" if bool(save_figures_var.get()) else "只显示"
        if current_workflow() == WORKFLOW_MICROPHONE:
            positions = ["主驾驶", "中排", "后排"]
            pos_text = "、".join(positions[idx] for idx in selected_microphone_indices()) or "未选麦克风位置"
            freq_text = f"FFT {mic_fft_min_var.get()}-{mic_fft_max_var.get()} Hz；总声压级/倍频程 {mic_total_min_var.get()}-{mic_total_max_var.get()} Hz"
            summary_var.set(f"{current_workflow()} | 已选 {selected_count} 个 | {plots} | {pos_text} | {freq_text} | {save_state}")
        elif current_workflow() == WORKFLOW_PAIRED:
            summary_var.set(f"{current_workflow()} | 已选 {selected_count} 组同名文件 | {plots} | {fmin_var.get()}-{fmax_var.get()} Hz | {save_state}")
        else:
            folder_text = folder_var.get() or "未选择日期"
            summary_var.set(f"{current_workflow()} | {folder_text} | 已选 {selected_count} 个 | {plots} | 输入基准 {input_mode_var.get()} | {fmin_var.get()}-{fmax_var.get()} Hz | {save_state}")

    def clear_content() -> None:
        for widget in content_frame.winfo_children():
            widget.destroy()

    def make_header(parent: Any, title: str, subtitle: str = "") -> None:
        ctk.CTkLabel(parent, text=title, font=("Microsoft YaHei UI", 24, "bold"), text_color=colors["text"]).grid(row=0, column=0, sticky="w", padx=4, pady=(0, 2))
        if subtitle:
            ctk.CTkLabel(parent, text=subtitle, font=("Microsoft YaHei UI", 13), text_color=colors["muted"]).grid(row=1, column=0, sticky="w", padx=4, pady=(0, 14))

    def add_label_entry(parent: Any, row: int, col: int, label: str, var: Any, width: int = 150, columnspan: int = 1) -> Any:
        ctk.CTkLabel(parent, text=label, text_color=colors["muted"], font=("Microsoft YaHei UI", 12)).grid(row=row, column=col, sticky="w", padx=8, pady=(8, 2))
        entry = ctk.CTkEntry(parent, textvariable=var, width=width, height=control_height, fg_color=colors["panel"], border_color=colors["line"], text_color=colors["text"], font=option_font)
        entry.grid(row=row + 1, column=col, columnspan=columnspan, sticky="ew", padx=8, pady=(0, 12))
        return entry

    def refresh_after_selection() -> None:
        update_summary()
        if step_var.get() in {"图形样式", "确认运行"}:
            render_current_step()

    def render_data_step() -> None:
        clear_content()
        content_frame.grid_columnconfigure(0, weight=1)
        make_header(content_frame, "选择数据", "默认不选择任何数据；请勾选本次要处理的文件或平均组。")

        controls = ctk.CTkFrame(content_frame, fg_color=colors["card"], border_color=colors["line"], border_width=1, corner_radius=14)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=2)

        if current_workflow() == WORKFLOW_VIBRATION:
            folders = data_folders()
            if folders and not folder_var.get():
                folder_var.set(folders[0])
            ctk.CTkLabel(controls, text="日期文件夹", text_color=colors["muted"]).grid(row=0, column=0, padx=12, pady=12)
            ctk.CTkOptionMenu(
                controls,
                variable=folder_var,
                values=folders or ["无数据"],
                command=lambda _value: (rebuild_data_items(True), render_current_step(), update_summary()),
                width=260,
                height=control_height,
                font=option_font,
                dropdown_font=option_font,
            ).grid(row=0, column=1, sticky="ew", padx=8, pady=12)
        else:
            fixed_text = "固定读取 5.22 与 5.23 同名文件" if current_workflow() == WORKFLOW_PAIRED else "固定读取 5.31 麦克风数据"
            ctk.CTkLabel(controls, text=fixed_text, text_color=colors["text"], font=("Microsoft YaHei UI", 13, "bold")).grid(row=0, column=0, columnspan=2, padx=12, pady=12, sticky="w")

        ctk.CTkLabel(controls, text="搜索", text_color=colors["muted"]).grid(row=0, column=2, padx=(20, 8), pady=12)
        search_entry = ctk.CTkEntry(controls, textvariable=search_var, height=control_height, fg_color=colors["panel"], border_color=colors["line"], text_color=colors["text"], font=option_font)
        search_entry.grid(row=0, column=3, sticky="ew", padx=8, pady=12)

        table = ctk.CTkScrollableFrame(content_frame, fg_color=colors["panel"], border_color=colors["line"], border_width=1, corner_radius=14)
        table.grid(row=3, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(3, weight=1)
        for index, weight in enumerate([0, 3, 2, 1, 2, 2, 1]):
            table.grid_columnconfigure(index, weight=weight)

        headers = ["选", "原始文件名", "图中显示名", "日期", "数据结构", "输入基准", "状态"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(table, text=header, text_color=colors["muted"], font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=col, sticky="ew", padx=6, pady=(8, 6))

        keyword = search_var.get().strip().lower()
        visible_items = [
            item for item in data_items.values()
            if not keyword or keyword in item.display_label.lower() or keyword in item.date_label.lower()
        ]

        def set_visible(value: bool) -> None:
            for item in visible_items:
                data_selected_vars[item.key].set(value)
            refresh_after_selection()

        def invert_visible() -> None:
            for item in visible_items:
                data_selected_vars[item.key].set(not bool(data_selected_vars[item.key].get()))
            refresh_after_selection()

        ctk.CTkButton(controls, text="全选当前显示", command=lambda: set_visible(True), fg_color=colors["accent"], hover_color=colors["accent_dark"]).grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")
        ctk.CTkButton(controls, text="清空", command=lambda: set_visible(False), fg_color=colors["card"], hover_color=colors["soft"], text_color=colors["text"]).grid(row=1, column=1, padx=8, pady=(0, 12), sticky="ew")
        ctk.CTkButton(controls, text="反选", command=invert_visible, fg_color=colors["card"], hover_color=colors["soft"], text_color=colors["text"]).grid(row=1, column=2, padx=8, pady=(0, 12), sticky="ew")

        for row, item in enumerate(visible_items, start=1):
            bg = colors["card"] if row % 2 else colors["panel"]
            cb = ctk.CTkCheckBox(table, text="", variable=data_selected_vars[item.key], command=refresh_after_selection, width=20)
            cb.grid(row=row, column=0, sticky="w", padx=6, pady=5)
            ctk.CTkLabel(table, text=item.display_label, text_color=colors["text"], anchor="w").grid(row=row, column=1, sticky="ew", padx=6, pady=5)
            ctk.CTkEntry(table, textvariable=data_display_name_vars[item.key], fg_color=bg, border_color=colors["line"], text_color=colors["text"], height=compact_height, font=compact_option_font).grid(row=row, column=2, sticky="ew", padx=6, pady=6)
            ctk.CTkLabel(table, text=item.date_label, text_color=colors["muted"]).grid(row=row, column=3, sticky="ew", padx=6, pady=5)
            ctk.CTkLabel(table, text=item.layout_desc, text_color=colors["muted"]).grid(row=row, column=4, sticky="ew", padx=6, pady=5)
            ctk.CTkLabel(table, text=item.input_desc, text_color=colors["muted"]).grid(row=row, column=5, sticky="ew", padx=6, pady=5)
            ctk.CTkLabel(table, text=item.status, text_color=colors["success"] if item.status == "有效" else colors["muted"]).grid(row=row, column=6, sticky="ew", padx=6, pady=5)

        update_summary()

    def render_plot_step() -> None:
        clear_content()
        make_header(content_frame, "选择图型", "只显示当前工作流可用的图型。")
        holder = ctk.CTkScrollableFrame(content_frame, fg_color="transparent")
        holder.grid(row=2, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(2, weight=1)
        holder.grid_columnconfigure(0, weight=1)

        if current_workflow() == WORKFLOW_MICROPHONE:
            card = ctk.CTkFrame(holder, fg_color=colors["card"], border_color=colors["line"], border_width=1, corner_radius=14)
            card.grid(row=0, column=0, sticky="ew", pady=8)
            card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(card, text="麦克风声学图", font=("Microsoft YaHei UI", 16, "bold"), text_color=colors["text"]).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))
            for idx, (label, key) in enumerate(MICROPHONE_PLOTS, start=1):
                ctk.CTkCheckBox(card, text=label, variable=mic_plot_vars[key], command=update_summary, text_color=colors["text"]).grid(row=idx, column=0, sticky="w", padx=18, pady=8)
            ctk.CTkCheckBox(card, text="按同名前缀平均后出图", variable=mic_average_var, command=lambda: (update_summary(), render_current_step()), text_color=colors["text"]).grid(row=4, column=0, sticky="w", padx=18, pady=(10, 14))
            pos = ctk.CTkFrame(holder, fg_color=colors["card"], border_color=colors["line"], border_width=1, corner_radius=14)
            pos.grid(row=1, column=0, sticky="ew", pady=8)
            ctk.CTkLabel(pos, text="麦克风位置", font=("Microsoft YaHei UI", 16, "bold"), text_color=colors["text"]).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))
            for idx, name in enumerate(["主驾驶", "中排", "后排"]):
                ctk.CTkCheckBox(pos, text=name, variable=mic_position_vars[idx], command=update_summary, text_color=colors["text"]).grid(row=1, column=idx, sticky="w", padx=18, pady=(0, 14))
        else:
            vars_to_use = current_plot_vars()
            row = 0
            for group_name, items in STRUCTURE_PLOT_GROUPS.items():
                card = ctk.CTkFrame(holder, fg_color=colors["card"], border_color=colors["line"], border_width=1, corner_radius=14)
                card.grid(row=row, column=0, sticky="ew", pady=8)
                card.grid_columnconfigure(0, weight=1)
                ctk.CTkLabel(card, text=group_name, font=("Microsoft YaHei UI", 16, "bold"), text_color=colors["text"]).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))
                for idx, (label, key) in enumerate(items, start=1):
                    ctk.CTkCheckBox(card, text=label, variable=vars_to_use[key], command=update_summary, text_color=colors["text"]).grid(row=idx, column=0, sticky="w", padx=18, pady=7)
                row += 1

    def render_params_step() -> None:
        clear_content()
        make_header(content_frame, "设置参数", "普通参数优先显示，高级参数保持可调但不干扰常用流程。")
        holder = ctk.CTkScrollableFrame(content_frame, fg_color="transparent")
        holder.grid(row=2, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(2, weight=1)
        holder.grid_columnconfigure((0, 1), weight=1)

        if current_workflow() == WORKFLOW_MICROPHONE:
            add_label_entry(holder, 0, 0, "FFT声压级下限 / Hz", mic_fft_min_var)
            add_label_entry(holder, 0, 1, "FFT声压级上限 / Hz", mic_fft_max_var)
            add_label_entry(holder, 2, 0, "总声压级/倍频程下限 / Hz", mic_total_min_var)
            add_label_entry(holder, 2, 1, "总声压级/倍频程上限 / Hz", mic_total_max_var)
            ctk.CTkLabel(holder, text="倍频程类型", text_color=colors["muted"]).grid(row=4, column=0, sticky="w", padx=8, pady=(10, 2))
            ctk.CTkOptionMenu(holder, variable=mic_octave_var, values=["1/3 倍频程", "1/6 倍频程", "1/12 倍频程", "1/24 倍频程"], command=lambda _v: update_summary(), height=control_height, font=option_font, dropdown_font=option_font).grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 12))
        else:
            add_label_entry(holder, 0, 0, "频率下限 / Hz", fmin_var)
            add_label_entry(holder, 0, 1, "频率上限 / Hz", fmax_var)
            ctk.CTkLabel(holder, text="输入基准", text_color=colors["muted"]).grid(row=2, column=0, sticky="w", padx=8, pady=(10, 2))
            ctk.CTkOptionMenu(holder, variable=input_mode_var, values=["自动", "Data第1行", "Data第2行", "Data第3行", "Data第4行"], command=lambda _v: update_summary(), height=control_height, font=option_font, dropdown_font=option_font).grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 12))
            ctk.CTkLabel(holder, text="处理模式", text_color=colors["muted"]).grid(row=2, column=1, sticky="w", padx=8, pady=(10, 2))
            ctk.CTkOptionMenu(holder, variable=mode_var, values=list(MODE_DISPLAY_TO_VALUE.keys()), command=lambda _v: update_summary(), height=control_height, font=option_font, dropdown_font=option_font).grid(row=3, column=1, sticky="ew", padx=8, pady=(0, 12))
            add_label_entry(holder, 4, 0, "目标频率分辨率 df / Hz", df_var)
            add_label_entry(holder, 4, 1, "分析时长 / s", duration_var)
            add_label_entry(holder, 6, 0, "起始时间 / s", start_var)
            add_label_entry(holder, 6, 1, "前裁时间 / s", trim_start_var)
            add_label_entry(holder, 8, 0, "后裁时间 / s", trim_end_var)

        save_card = ctk.CTkFrame(holder, fg_color=colors["card"], border_color=colors["line"], border_width=1, corner_radius=14)
        save_card.grid(row=12, column=0, columnspan=2, sticky="ew", padx=4, pady=18)
        save_card.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(save_card, text="输出设置", font=("Microsoft YaHei UI", 16, "bold"), text_color=colors["text"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 8))
        ctk.CTkCheckBox(save_card, text="出图后显示", variable=show_figures_var, command=update_summary, text_color=colors["text"]).grid(row=1, column=0, sticky="w", padx=16, pady=8)
        ctk.CTkCheckBox(save_card, text="保存本次生成图像", variable=save_figures_var, command=update_summary, text_color=colors["text"]).grid(row=1, column=1, sticky="w", padx=16, pady=8)
        if current_workflow() != WORKFLOW_MICROPHONE:
            ctk.CTkCheckBox(save_card, text="导出 Excel", variable=export_excel_var, command=update_summary, text_color=colors["text"]).grid(row=2, column=0, sticky="w", padx=16, pady=8)
        ctk.CTkLabel(save_card, text="格式", text_color=colors["muted"]).grid(row=3, column=0, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkOptionMenu(save_card, variable=format_var, values=["png", "pdf", "svg", "jpg"], command=lambda _v: update_summary(), height=control_height, font=option_font, dropdown_font=option_font).grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 12))
        add_label_entry(save_card, 3, 1, "DPI", dpi_var)
        ctk.CTkLabel(save_card, text="保存位置", text_color=colors["muted"]).grid(row=5, column=0, sticky="w", padx=16, pady=(12, 2))
        ctk.CTkEntry(save_card, textvariable=save_dir_var, height=control_height, fg_color=colors["panel"], border_color=colors["line"], text_color=colors["text"], font=option_font).grid(row=6, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 12))
        ctk.CTkButton(save_card, text="浏览保存位置", command=lambda: choose_save_dir(), fg_color=colors["card"], hover_color=colors["soft"], text_color=colors["text"]).grid(row=7, column=0, columnspan=2, sticky="ew", padx=16, pady=(4, 16))

    def render_style_step() -> None:
        clear_content()
        make_header(content_frame, "图形样式", "这里只列出本次实际参与出图的曲线或平均组。")
        holder = ctk.CTkScrollableFrame(content_frame, fg_color="transparent")
        holder.grid(row=2, column=0, sticky="nsew")
        content_frame.grid_rowconfigure(2, weight=1)
        holder.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(holder, text="样式预设", text_color=colors["muted"]).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        ctk.CTkOptionMenu(holder, variable=style_preset_var, values=list(STYLE_PRESETS.keys()), command=lambda _v: apply_style_preset(), height=control_height, font=option_font, dropdown_font=option_font).grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 12))
        add_label_entry(holder, 0, 1, "字体", font_var)
        add_label_entry(holder, 2, 0, "标题字号", title_size_var)
        add_label_entry(holder, 2, 1, "坐标轴字号", label_size_var)
        add_label_entry(holder, 4, 0, "刻度字号", tick_size_var)
        add_label_entry(holder, 4, 1, "图例字号", legend_size_var)
        add_label_entry(holder, 6, 0, "默认线宽", line_width_var)
        ctk.CTkLabel(holder, text="默认线形", text_color=colors["muted"]).grid(row=6, column=1, sticky="w", padx=8, pady=(8, 2))
        ctk.CTkOptionMenu(holder, variable=line_style_var, values=["实线", "虚线", "点划线", "点线"], height=control_height, font=option_font, dropdown_font=option_font).grid(row=7, column=1, sticky="ew", padx=8, pady=(0, 12))
        add_label_entry(holder, 8, 0, "透明度", line_alpha_var)
        add_label_entry(holder, 8, 1, "亮度", brightness_var)
        add_label_entry(holder, 10, 0, "网格透明度", grid_alpha_var)
        add_label_entry(holder, 10, 1, "图片宽度 inch，可空", figure_width_var)
        add_label_entry(holder, 12, 0, "图片高度 inch，可空", figure_height_var)
        add_label_entry(holder, 12, 1, "完整标题覆盖，可空", title_override_var)
        add_label_entry(holder, 14, 0, "标题前缀，可空", title_prefix_var)
        add_label_entry(holder, 14, 1, "标题后缀，可空", title_suffix_var)
        add_label_entry(holder, 16, 0, "X轴标签覆盖，可空", xlabel_var)
        add_label_entry(holder, 16, 1, "Y轴标签覆盖，可空", ylabel_var)

        table = ctk.CTkFrame(holder, fg_color=colors["panel"], border_color=colors["line"], border_width=1, corner_radius=14)
        table.grid(row=18, column=0, columnspan=2, sticky="ew", padx=4, pady=18)
        table.grid_columnconfigure(1, weight=2)
        table.grid_columnconfigure(2, weight=2)
        table.grid_columnconfigure((4, 5, 6), weight=1)
        headers = ["顺序", "曲线/平均组", "图中显示名", "颜色", "线宽", "线形", "亮度"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(table, text=header, text_color=colors["muted"], font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=col, sticky="ew", padx=6, pady=(12, 6))

        targets = selected_style_targets()
        for row, target in enumerate(targets, start=1):
            key = target["key"]
            style_display_name_vars.setdefault(key, ctk.StringVar(value=target["default_name"]))
            style_line_width_vars.setdefault(key, ctk.StringVar(value="默认"))
            style_line_style_vars.setdefault(key, ctk.StringVar(value="默认"))
            style_brightness_vars.setdefault(key, ctk.StringVar(value="默认"))
            style_color_values.setdefault(key, default_color_for_key(key, row - 1))

            order_frame = ctk.CTkFrame(table, fg_color="transparent")
            order_frame.grid(row=row, column=0, sticky="ew", padx=6, pady=5)
            ctk.CTkButton(order_frame, text="↑", width=34, command=lambda k=key: move_style_key(k, -1), fg_color=colors["card"], hover_color=colors["soft"], text_color=colors["text"]).pack(side="left", padx=2)
            ctk.CTkButton(order_frame, text="↓", width=34, command=lambda k=key: move_style_key(k, 1), fg_color=colors["card"], hover_color=colors["soft"], text_color=colors["text"]).pack(side="left", padx=2)
            ctk.CTkLabel(table, text=key, text_color=colors["text"], anchor="w").grid(row=row, column=1, sticky="ew", padx=6, pady=5)
            ctk.CTkEntry(table, textvariable=style_display_name_vars[key], fg_color=colors["card"], border_color=colors["line"], text_color=colors["text"], height=compact_height, font=compact_option_font).grid(row=row, column=2, sticky="ew", padx=6, pady=6)
            swatch = ctk.CTkButton(table, text="", width=58, height=compact_height, fg_color=style_color_values[key], hover_color=style_color_values[key], command=lambda k=key: choose_style_color(k))
            swatch.grid(row=row, column=3, sticky="w", padx=6, pady=5)
            style_swatch_widgets[key] = swatch
            ctk.CTkOptionMenu(table, variable=style_line_width_vars[key], values=LINE_WIDTH_OPTIONS, width=140, height=compact_height, font=compact_option_font, dropdown_font=compact_option_font).grid(row=row, column=4, sticky="ew", padx=6, pady=6)
            ctk.CTkOptionMenu(table, variable=style_line_style_vars[key], values=LINE_STYLE_OPTIONS, width=140, height=compact_height, font=compact_option_font, dropdown_font=compact_option_font).grid(row=row, column=5, sticky="ew", padx=6, pady=6)
            ctk.CTkOptionMenu(table, variable=style_brightness_vars[key], values=BRIGHTNESS_OPTIONS, width=140, height=compact_height, font=compact_option_font, dropdown_font=compact_option_font).grid(row=row, column=6, sticky="ew", padx=6, pady=6)

        if not targets:
            ctk.CTkLabel(table, text="尚未选择数据。请先到“选择数据”步骤勾选文件。", text_color=colors["muted"]).grid(row=1, column=0, columnspan=7, sticky="w", padx=14, pady=16)

    def render_confirm_step() -> None:
        clear_content()
        make_header(content_frame, "确认并运行", "运行前检查本次任务的关键设置。")
        card = ctk.CTkFrame(content_frame, fg_color=colors["card"], border_color=colors["line"], border_width=1, corner_radius=14)
        card.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        card.grid_columnconfigure(1, weight=1)

        rows = [
            ("工作流", current_workflow()),
            ("已选数据", f"{len(selected_data_items())} 个" if current_workflow() != WORKFLOW_PAIRED else f"{len(selected_data_items())} 组同名文件"),
            ("图型", "、".join(selected_plot_names()) or "未选择"),
            ("保存", "保存到文件" if save_figures_var.get() else "只显示不保存"),
            ("输出位置", save_dir_var.get()),
            ("格式 / DPI", f"{format_var.get()} / {dpi_var.get()}"),
        ]
        if current_workflow() == WORKFLOW_MICROPHONE:
            rows.extend([
                ("FFT频段", f"{mic_fft_min_var.get()}-{mic_fft_max_var.get()} Hz"),
                ("总声压级/倍频程频段", f"{mic_total_min_var.get()}-{mic_total_max_var.get()} Hz"),
                ("倍频程", mic_octave_var.get()),
                ("麦克风位置", "、".join(["主驾驶", "中排", "后排"][idx] for idx in selected_microphone_indices()) or "未选择"),
            ])
        else:
            rows.extend([
                ("频段", f"{fmin_var.get()}-{fmax_var.get()} Hz"),
                ("输入基准", input_mode_var.get()),
                ("统计测点", "自动：含左中/右中用六点，否则四点"),
                ("处理模式", mode_var.get()),
            ])

        for row, (label, value) in enumerate(rows):
            ctk.CTkLabel(card, text=label, text_color=colors["muted"], anchor="w").grid(row=row, column=0, sticky="w", padx=18, pady=8)
            ctk.CTkLabel(card, text=value, text_color=colors["text"], anchor="w", wraplength=760, justify="left").grid(row=row, column=1, sticky="ew", padx=18, pady=8)

        ctk.CTkButton(content_frame, text="开始处理并出图", height=46, command=run_current_task, fg_color=colors["accent"], hover_color=colors["accent_dark"], font=("Microsoft YaHei UI", 16, "bold")).grid(row=3, column=0, sticky="ew")

    def render_current_step() -> None:
        current = step_var.get()
        for name, button in step_buttons.items():
            button.configure(fg_color=colors["accent"] if name == current else colors["card"], text_color="#ffffff" if name == current else colors["text"])
        if current == "选择数据":
            render_data_step()
        elif current == "选择图型":
            render_plot_step()
        elif current == "设置参数":
            render_params_step()
        elif current == "图形样式":
            render_style_step()
        else:
            render_confirm_step()
        update_summary()

    def select_step(name: str) -> None:
        step_var.set(name)
        render_current_step()

    def choose_save_dir() -> None:
        selected = filedialog.askdirectory(initialdir=save_dir_var.get() or str(PROJECT_ROOT))
        if selected:
            save_dir_var.set(selected)
            update_summary()

    def apply_style_preset() -> None:
        preset = STYLE_PRESETS.get(style_preset_var.get(), STYLE_PRESETS["PPT汇报"])
        font_var.set(preset["font"])
        title_size_var.set(preset["title"])
        label_size_var.set(preset["label"])
        tick_size_var.set(preset["tick"])
        legend_size_var.set(preset["legend"])
        annotation_size_var.set(preset["annotation"])
        line_width_var.set(preset["line_width"])
        line_style_var.set(next((name for name, value in LINE_STYLE_VALUES.items() if value == preset.get("line_style", "-") and name != "默认"), "实线"))
        line_alpha_var.set(preset["alpha"])
        brightness_var.set(preset["brightness"])
        grid_alpha_var.set(preset["grid"])
        figure_width_var.set(preset["width"])
        figure_height_var.set(preset["height"])
        update_summary()

    def move_style_key(key: str, delta: int) -> None:
        selected_style_targets()
        if key not in style_order:
            return
        index = style_order.index(key)
        new_index = max(0, min(len(style_order) - 1, index + delta))
        if new_index == index:
            return
        style_order.pop(index)
        style_order.insert(new_index, key)
        render_style_step()

    def choose_style_color(key: str) -> None:
        current = style_color_values.get(key) or default_color_for_key(key)
        _rgb, selected = colorchooser.askcolor(color=current, title=f"选择 {key} 曲线颜色")
        if selected:
            style_color_values[key] = normalize_color(selected)
            swatch = style_swatch_widgets.get(key)
            if swatch is not None:
                swatch.configure(fg_color=style_color_values[key], hover_color=style_color_values[key])
            update_summary()

    def run_current_task() -> None:
        try:
            require_runtime_dependencies()
            status_var.set("正在处理，请等待...")
            app.update_idletasks()
            if current_workflow() == WORKFLOW_MICROPHONE:
                result = build_microphone_task().run()
            elif current_workflow() == WORKFLOW_PAIRED:
                result = build_paired_task().run()
            else:
                result = build_vibration_task().run()
            if save_figures_var.get() or export_excel_var.get():
                preview = "\n".join(str(path) for path in result.saved_paths[:12])
                if len(result.saved_paths) > 12:
                    preview += f"\n...共 {len(result.saved_paths)} 个文件"
                messagebox.showinfo("完成", f"{result.message}\n\n{preview}" if preview else result.message)
            else:
                messagebox.showinfo("完成", f"{result.message}\n图像已显示，未保存文件。")
            status_var.set(f"{result.message} 有效结果 {result.result_count} 个。")
        except Exception as exc:
            status_var.set("处理失败。")
            print(f"error: {exc}")
            messagebox.showerror("错误", str(exc))

    def select_workflow(name: str) -> None:
        workflow_var.set(name)
        for workflow_name, button in workflow_buttons.items():
            button.configure(fg_color=colors["accent"] if workflow_name == name else colors["card"], text_color="#ffffff" if workflow_name == name else colors["text"])
        if name == WORKFLOW_MICROPHONE:
            search_var.set("")
        elif name == WORKFLOW_PAIRED:
            search_var.set("")
        step_var.set("选择数据")
        rebuild_data_items(reset_selection=True)
        render_current_step()

    for row, workflow_name in enumerate(WORKFLOWS, start=2):
        btn = ctk.CTkButton(
            sidebar,
            text=workflow_name,
            height=54,
            corner_radius=14,
            fg_color=colors["accent"] if workflow_name == WORKFLOW_VIBRATION else colors["card"],
            hover_color=colors["accent_dark"] if workflow_name == WORKFLOW_VIBRATION else colors["soft"],
            text_color="#ffffff" if workflow_name == WORKFLOW_VIBRATION else colors["text"],
            command=lambda name=workflow_name: select_workflow(name),
            anchor="w",
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        btn.grid(row=row, column=0, sticky="ew", padx=18, pady=6)
        workflow_buttons[workflow_name] = btn

    ctk.CTkLabel(sidebar, textvariable=status_var, text_color=colors["muted"], wraplength=245, justify="left").grid(row=8, column=0, sticky="sw", padx=20, pady=(30, 8))

    ctk.CTkLabel(main, text="任务配置", font=("Microsoft YaHei UI", 24, "bold"), text_color=colors["text"]).grid(row=0, column=0, sticky="w", padx=22, pady=(20, 8))
    step_bar = ctk.CTkFrame(main, fg_color="transparent")
    step_bar.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 6))
    for col, name in enumerate(STEPS):
        step_bar.grid_columnconfigure(col, weight=1)
        btn = ctk.CTkButton(
            step_bar,
            text=name,
            height=38,
            fg_color=colors["accent"] if name == STEPS[0] else colors["card"],
            hover_color=colors["accent_dark"] if name == STEPS[0] else colors["soft"],
            text_color="#ffffff" if name == STEPS[0] else colors["text"],
            command=lambda step=name: select_step(step),
        )
        btn.grid(row=0, column=col, sticky="ew", padx=4)
        step_buttons[name] = btn

    ctk.CTkLabel(footer, textvariable=summary_var, text_color=colors["text"], anchor="w", justify="left", wraplength=980).grid(row=0, column=0, sticky="ew", padx=18, pady=(12, 4))
    ctk.CTkLabel(footer, textvariable=status_var, text_color=colors["muted"], anchor="w", justify="left").grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))

    for traced_var in [
        fmin_var,
        fmax_var,
        df_var,
        duration_var,
        mode_var,
        start_var,
        trim_start_var,
        trim_end_var,
        input_mode_var,
        save_figures_var,
        show_figures_var,
        export_excel_var,
        format_var,
        dpi_var,
        save_dir_var,
        mic_fft_min_var,
        mic_fft_max_var,
        mic_total_min_var,
        mic_total_max_var,
        mic_octave_var,
        mic_average_var,
    ]:
        traced_var.trace_add("write", lambda *_: update_summary())
    search_var.trace_add("write", lambda *_: render_current_step() if step_var.get() == "选择数据" else None)

    apply_style_preset()
    rebuild_data_items(reset_selection=True)
    render_current_step()
    app.mainloop()
