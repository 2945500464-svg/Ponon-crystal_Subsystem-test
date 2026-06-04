from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .analysis import analyze_selected_files
from .config import FIGURES_DIR, GUI_THEME_PATH, PROJECT_ROOT, RAW_DATA_DIR, RESULTS_DIR, analysis_duration_sec, analysis_mode, ensure_directories, figure_save_dpi, fmax, fmin, legend_fontsize, require_runtime_dependencies, show_figures, start_time_sec, target_df, trim_end_sec, trim_start_sec
from .data_format import load_data_format_config
from .exporters import save_selected_outputs
from .microphone_processing import MICROPHONE_FREQ_MAX, MICROPHONE_FREQ_MIN, analyze_microphone_files, save_microphone_outputs
from .plotting import build_average_results_between_dates
from .utils import get_condition_color, get_display_condition_name, normalize_condition_label

def launch_analysis_gui() -> None:
    """Launch the light, workflow-oriented CustomTkinter GUI."""
    try:
        import customtkinter as ctk
        import tkinter as tk
        from tkinter import colorchooser, filedialog, messagebox
    except ImportError as exc:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("缺少依赖", f"需要安装 customtkinter。\n请先运行: pip install -r requirements.txt\n\n{exc}")
        root.destroy()
        return

    ensure_directories()
    ctk.set_appearance_mode("light")
    if GUI_THEME_PATH.exists():
        ctk.set_default_color_theme(str(GUI_THEME_PATH))
    else:
        ctk.set_default_color_theme("blue")

    bg = "#f6f8fb"
    panel = "#ffffff"
    panel_alt = "#eef4fb"
    card = "#f8fafc"
    accent = "#2563eb"
    accent_hover = "#1d4ed8"
    text_main = "#1e293b"
    text_muted = "#64748b"
    border = "#d7dee8"
    label_text = "#334155"
    soft_blue = "#e0f2fe"
    row_alt = "#f1f5f9"

    app = ctk.CTk()
    app.title("副车架声子晶体试验数据处理")
    window_width = 1360
    window_height = 880
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()
    pos_x = max(0, int((screen_width - window_width) / 2))
    pos_y = max(0, int((screen_height - window_height) / 2))
    app.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
    app.minsize(1180, 760)
    app.configure(fg_color=bg)

    mode_display_to_value = {
        "单段分析": "single_segment",
        "分段平均": "segment_average",
    }
    mode_value_to_display = {value: label for label, value in mode_display_to_value.items()}
    WORKFLOW_VIBRATION = "副车架振动数据处理"
    WORKFLOW_PAIRED = "5.22 + 5.23 同名平均"
    WORKFLOW_MICROPHONE = "麦克风声学数据处理"
    workflow_options = [WORKFLOW_VIBRATION, WORKFLOW_PAIRED, WORKFLOW_MICROPHONE]

    workflow_var = ctk.StringVar(value=WORKFLOW_VIBRATION)
    folder_var = ctk.StringVar()
    save_dir_var = ctk.StringVar(value=str(RESULTS_DIR / "figures"))
    format_var = ctk.StringVar(value="png")
    dpi_var = ctk.StringVar(value=str(figure_save_dpi))
    fmin_var = ctk.StringVar(value=str(fmin))
    fmax_var = ctk.StringVar(value=str(fmax))
    df_var = ctk.StringVar(value=str(target_df))
    duration_var = ctk.StringVar(value=str(analysis_duration_sec))
    mode_var = ctk.StringVar(value=mode_value_to_display.get(analysis_mode, "单段分析"))
    start_var = ctk.StringVar(value=str(start_time_sec))
    trim_start_var = ctk.StringVar(value=str(trim_start_sec))
    trim_end_var = ctk.StringVar(value=str(trim_end_sec))
    input_mode_var = ctk.StringVar(value="自动")
    stat_point_mode_var = ctk.StringVar(value="auto")
    show_var = ctk.BooleanVar(value=True)
    save_figures_var = ctk.BooleanVar(value=False)
    export_excel_var = ctk.BooleanVar(value=False)
    status_var = ctk.StringVar(value="请选择数据文件夹和数据文件。")
    file_info_var = ctk.StringVar(value="未选择数据源")
    structure_info_var = ctk.StringVar(value="数据结构：等待选择")
    output_info_var = ctk.StringVar(value=f"输出目录：{save_dir_var.get()}")
    task_summary_var = ctk.StringVar(value="当前任务：请选择工作流和数据文件。")
    average_522_523_var = ctk.BooleanVar(value=False)

    plot_vars = {
        "single_outputs": ctk.BooleanVar(value=False),
        "four_average": ctk.BooleanVar(value=True),
        "six_average": ctk.BooleanVar(value=False),
        "input_psd": ctk.BooleanVar(value=False),
        "force_psd": ctk.BooleanVar(value=False),
        "frf": ctk.BooleanVar(value=False),
        "heatmap_no_load": ctk.BooleanVar(value=False),
        "heatmap_dva": ctk.BooleanVar(value=False),
        "total_level": ctk.BooleanVar(value=False),
        "damping_rate": ctk.BooleanVar(value=False),
    }
    mic_plot_vars = {
        "fft_spl": ctk.BooleanVar(value=True),
        "total_spl": ctk.BooleanVar(value=True),
        "third_octave": ctk.BooleanVar(value=True),
    }
    mic_position_vars = {
        0: ctk.BooleanVar(value=True),
        1: ctk.BooleanVar(value=True),
        2: ctk.BooleanVar(value=True),
    }
    mic_average_prefix_var = ctk.BooleanVar(value=False)
    mic_octave_fraction_var = ctk.StringVar(value="1/3 倍频程")
    mic_freq_min_var = ctk.StringVar(value=str(MICROPHONE_FREQ_MIN))
    mic_freq_max_var = ctk.StringVar(value=str(MICROPHONE_FREQ_MAX))
    font_family_var = ctk.StringVar(value="Microsoft YaHei UI")
    title_fontsize_var = ctk.StringVar(value="14")
    label_fontsize_var = ctk.StringVar(value="12")
    tick_fontsize_var = ctk.StringVar(value="10")
    legend_fontsize_var = ctk.StringVar(value=str(legend_fontsize))
    annotation_fontsize_var = ctk.StringVar(value="9")
    line_width_var = ctk.StringVar(value="1.6")
    line_alpha_var = ctk.StringVar(value="1.0")
    brightness_var = ctk.StringVar(value="1.0")
    grid_alpha_var = ctk.StringVar(value="0.45")
    figure_width_var = ctk.StringVar(value="")
    figure_height_var = ctk.StringVar(value="")
    title_prefix_var = ctk.StringVar(value="")
    title_suffix_var = ctk.StringVar(value="")
    xlabel_override_var = ctk.StringVar(value="")
    ylabel_override_var = ctk.StringVar(value="")
    file_search_var = ctk.StringVar(value="")
    current_files: List[Path] = []
    file_check_vars: Dict[str, Any] = {}
    curve_order: List[str] = []
    style_order: List[str] = []
    last_style_signature: List[str] = []
    dragging_condition_name: Optional[str] = None
    curve_color_values: Dict[str, str] = {}
    curve_display_name_vars: Dict[str, Any] = {}
    curve_line_width_vars: Dict[str, Any] = {}
    curve_brightness_vars: Dict[str, Any] = {}
    color_swatch_widgets: Dict[str, Any] = {}

    curve_default_palette = [
        "#1f77b4",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#d62728",
    ]
    curve_line_width_options = ["默认", "0.8", "1.0", "1.2", "1.5", "1.8", "2.0", "2.5", "3.0", "3.5", "4.0", "5.0", "6.0"]
    curve_brightness_options = ["默认", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.4", "1.6", "1.8", "2.0"]

    def update_output_info(*_: Any) -> None:
        output_info_var.set(f"输出目录：{save_dir_var.get()}")

    save_dir_var.trace_add("write", update_output_info)

    app.grid_columnconfigure(0, weight=1)
    app.grid_rowconfigure(1, weight=1)

    header = ctk.CTkFrame(app, fg_color=panel, border_color=border, border_width=1, corner_radius=18)
    header.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
    header.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        header,
        text="副车架声子晶体试验数据处理",
        font=("Microsoft YaHei UI", 26, "bold"),
        text_color=accent,
    ).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 4))
    ctk.CTkLabel(
        header,
        text="简洁出图界面 · 数据结构优先读取 Raw_data/数据格式.xlsx · 左侧激振=Data第1行，右侧激振=Data第2行",
        font=("Microsoft YaHei UI", 13),
        text_color=text_muted,
    ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 16))

    body = ctk.CTkFrame(app, fg_color="transparent")
    body.grid(row=1, column=0, sticky="nsew", padx=18, pady=8)
    body.grid_columnconfigure(0, weight=2, minsize=360)
    body.grid_columnconfigure(1, weight=4, minsize=640)
    body.grid_rowconfigure(0, weight=1)

    left = ctk.CTkFrame(body, fg_color=panel, border_color=border, border_width=1, corner_radius=18)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    left.grid_columnconfigure(0, weight=1)
    left.grid_rowconfigure(7, weight=1)

    right = ctk.CTkFrame(body, fg_color=panel, border_color=border, border_width=1, corner_radius=18)
    right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
    right.grid_columnconfigure(0, weight=1)
    right.grid_rowconfigure(0, weight=0)
    right.grid_rowconfigure(1, weight=1)

    ctk.CTkLabel(left, text="数据源", font=("Microsoft YaHei UI", 21, "bold"), text_color=accent).grid(
        row=0, column=0, sticky="w", padx=18, pady=(18, 6)
    )
    ctk.CTkLabel(left, textvariable=file_info_var, font=("Microsoft YaHei UI", 14), text_color=text_muted).grid(
        row=1, column=0, sticky="w", padx=18, pady=(0, 8)
    )
    folder_menu = ctk.CTkOptionMenu(
        left,
        variable=folder_var,
        values=[],
        fg_color=card,
        button_color=accent,
        button_hover_color=accent_hover,
        text_color=text_main,
        dropdown_fg_color=panel_alt,
        dropdown_hover_color=soft_blue,
        dropdown_text_color=text_main,
    )
    folder_menu.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))
    ctk.CTkLabel(left, textvariable=structure_info_var, text_color=label_text, wraplength=340, justify="left").grid(
        row=3, column=0, sticky="ew", padx=18, pady=(0, 10)
    )

    ctk.CTkEntry(
        left,
        textvariable=file_search_var,
        placeholder_text="搜索数据文件，例如 plan、DVA、no-load",
        fg_color=card,
        border_color=border,
        text_color=text_main,
    ).grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 8))

    file_actions = ctk.CTkFrame(left, fg_color="transparent")
    file_actions.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 8))
    file_actions.grid_columnconfigure((0, 1, 2), weight=1)

    list_frame = ctk.CTkFrame(left, fg_color=card, border_color=border, border_width=1, corner_radius=10)
    list_frame.grid(row=6, column=0, sticky="nsew", padx=18, pady=(0, 16))
    list_frame.grid_columnconfigure(0, weight=1)
    list_frame.grid_rowconfigure(0, weight=1)
    file_scroll_frame = ctk.CTkScrollableFrame(list_frame, fg_color="transparent")
    file_scroll_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    file_scroll_frame.grid_columnconfigure(0, weight=1)

    ctk.CTkButton(file_actions, text="全选", command=lambda: set_visible_file_selection(True), fg_color=panel_alt, hover_color=soft_blue, text_color=text_main).grid(row=0, column=0, sticky="ew", padx=(0, 6))
    ctk.CTkButton(file_actions, text="清空", command=lambda: clear_visible_file_selection(), fg_color=panel_alt, hover_color=soft_blue, text_color=text_main).grid(row=0, column=1, sticky="ew", padx=6)
    ctk.CTkButton(file_actions, text="反选", command=lambda: invert_visible_file_selection(), fg_color=panel_alt, hover_color=soft_blue, text_color=text_main).grid(row=0, column=2, sticky="ew", padx=(6, 0))

    workflow_card = ctk.CTkFrame(right, fg_color=card, border_color=border, border_width=1, corner_radius=14)
    workflow_card.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
    workflow_card.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(
        workflow_card,
        text="工作流",
        font=("Microsoft YaHei UI", 15, "bold"),
        text_color=accent,
    ).grid(row=0, column=0, sticky="w", padx=16, pady=12)
    workflow_segment = ctk.CTkSegmentedButton(
        workflow_card,
        variable=workflow_var,
        values=workflow_options,
        command=lambda value: on_workflow_changed(value),
        selected_color=soft_blue,
        selected_hover_color="#bfdbfe",
        unselected_color=panel_alt,
        unselected_hover_color=soft_blue,
        text_color=text_main,
        font=("Microsoft YaHei UI", 13, "bold"),
        dynamic_resizing=False,
    )
    workflow_segment.grid(row=0, column=1, sticky="ew", padx=12, pady=12)

    tabs = ctk.CTkTabview(
        right,
        fg_color=panel,
        segmented_button_selected_color=soft_blue,
        segmented_button_selected_hover_color="#bfdbfe",
        segmented_button_unselected_color=panel_alt,
        segmented_button_unselected_hover_color=soft_blue,
    )
    tabs.grid(row=1, column=0, sticky="nsew", padx=14, pady=(6, 14))
    tab_plots_raw = tabs.add("出图类型")
    tab_params_raw = tabs.add("分析参数")
    tab_save_raw = tabs.add("保存与统计")
    tab_style_raw = tabs.add("图形样式")
    tab_microphone_raw = tabs.add("麦克风信号处理")

    def make_scroll_tab(tab: Any) -> Any:
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure((0, 1), weight=1)
        return scroll

    tab_plots = make_scroll_tab(tab_plots_raw)
    tab_params = make_scroll_tab(tab_params_raw)
    tab_save = make_scroll_tab(tab_save_raw)
    tab_style = make_scroll_tab(tab_style_raw)
    tab_microphone = make_scroll_tab(tab_microphone_raw)

    def section_title(parent: Any, text: str, row: int) -> None:
        ctk.CTkLabel(parent, text=text, font=("Microsoft YaHei UI", 15, "bold"), text_color=accent).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=18, pady=(18, 8)
        )

    section_title(tab_plots, "选择要生成的图", 0)
    paired_average_checkbox = ctk.CTkCheckBox(
        tab_plots,
        text="使用 5.22 + 5.23 同名数据平均出图",
        variable=average_522_523_var,
        checkbox_width=22,
        checkbox_height=22,
        fg_color=accent,
        hover_color=accent_hover,
        text_color=text_main,
    )
    paired_average_checkbox.grid(row=1, column=0, columnspan=2, sticky="w", padx=18, pady=(8, 10))
    paired_average_note = ctk.CTkLabel(
        tab_plots,
        text="勾选后只允许在 5.22 或 5.23 文件夹中选择数据；程序会自动匹配另一日期同名文件，并只对完全对应的数据求平均。",
        text_color=text_muted,
        wraplength=620,
        justify="left",
    )
    paired_average_note.grid(row=2, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 8))
    plot_items = [
        ("单个输出通道", "single_outputs"),
        ("四点平均传递率", "four_average"),
        ("六点平均传递率", "six_average"),
        ("输入点PSD", "input_psd"),
        ("输入力PSD", "force_psd"),
        ("输入点加速度/力FRF", "frf"),
        ("归一化热力图：相对no-load", "heatmap_no_load"),
        ("归一化热力图：相对DVA", "heatmap_dva"),
        ("总振级图", "total_level"),
        ("平均减振率图", "damping_rate"),
    ]
    for idx, (text, key) in enumerate(plot_items, start=1):
        ctk.CTkCheckBox(
            tab_plots,
            text=text,
            variable=plot_vars[key],
            checkbox_width=22,
            checkbox_height=22,
            fg_color=accent,
            hover_color=accent_hover,
            text_color=text_main,
        ).grid(row=2 + (idx + 1) // 2, column=(idx - 1) % 2, sticky="w", padx=18, pady=10)

    ctk.CTkLabel(
        tab_plots,
        text="用途分组：基础曲线包含单输出、四点/六点平均、输入PSD、输入力PSD和FRF；对比评价包含热力图、总振级和平均减振率；5.22+5.23同名平均请在顶部工作流入口选择。",
        text_color=text_muted,
        wraplength=620,
        justify="left",
    ).grid(row=8, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))
    ctk.CTkButton(
        tab_plots,
        text="推荐组合：平均曲线 + 平均减振率",
        command=lambda: set_plot_option_combo("struct_common"),
        fg_color=panel_alt,
        hover_color=soft_blue,
        text_color=text_main,
    ).grid(row=9, column=0, sticky="ew", padx=18, pady=(8, 12))
    ctk.CTkButton(
        tab_plots,
        text="只看基础曲线",
        command=lambda: set_plot_option_combo("basic_curves"),
        fg_color=panel_alt,
        hover_color=soft_blue,
        text_color=text_main,
    ).grid(row=9, column=1, sticky="ew", padx=18, pady=(8, 12))

    section_title(tab_microphone, "麦克风信号处理", 0)
    ctk.CTkLabel(
        tab_microphone,
        text=(
            "独立处理 Raw_data/5.31 下的三行麦克风数据："
            "Data第1行为主驾驶，Data第2行为中排，Data第3行为后排。"
            "单位按 Pa 处理，参考声压 p0=20 μPa，频率范围可在下方调整。"
        ),
        text_color=text_muted,
        wraplength=620,
        justify="left",
    ).grid(row=1, column=0, columnspan=2, sticky="w", padx=18, pady=(8, 12))

    ctk.CTkLabel(
        tab_microphone,
        text="麦克风频率范围 / Hz",
        font=("Microsoft YaHei UI", 15, "bold"),
        text_color=accent,
    ).grid(row=2, column=0, columnspan=2, sticky="w", padx=18, pady=(10, 8))
    mic_freq_frame = ctk.CTkFrame(tab_microphone, fg_color="transparent")
    mic_freq_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 8))
    mic_freq_frame.grid_columnconfigure((1, 3), weight=1)
    ctk.CTkLabel(mic_freq_frame, text="下限", text_color=text_main).grid(row=0, column=0, sticky="w", padx=(0, 8))
    ctk.CTkEntry(
        mic_freq_frame,
        textvariable=mic_freq_min_var,
        fg_color=card,
        border_color=border,
        text_color=text_main,
        width=100,
    ).grid(row=0, column=1, sticky="ew", padx=(0, 14))
    ctk.CTkLabel(mic_freq_frame, text="上限", text_color=text_main).grid(row=0, column=2, sticky="w", padx=(0, 8))
    ctk.CTkEntry(
        mic_freq_frame,
        textvariable=mic_freq_max_var,
        fg_color=card,
        border_color=border,
        text_color=text_main,
        width=100,
    ).grid(row=0, column=3, sticky="ew")
    ctk.CTkLabel(
        tab_microphone,
        text="该范围同时用于 FFT声压级显示、总声压级积分和分数倍频程频带筛选。",
        text_color=text_muted,
        wraplength=620,
        justify="left",
    ).grid(row=4, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 10))

    mic_items = [
        ("FFT声压级图", "fft_spl"),
        ("总声压级图", "total_spl"),
        ("分数倍频程声压级图", "third_octave"),
    ]
    for idx, (text, key) in enumerate(mic_items):
        ctk.CTkCheckBox(
            tab_microphone,
            text=text,
            variable=mic_plot_vars[key],
            checkbox_width=22,
            checkbox_height=22,
            fg_color=accent,
            hover_color=accent_hover,
            text_color=text_main,
        ).grid(row=5 + idx, column=0, columnspan=2, sticky="w", padx=18, pady=10)

    ctk.CTkCheckBox(
        tab_microphone,
        text="按同名前缀平均后出图（忽略文件名最后的 _1、_2 等编号）",
        variable=mic_average_prefix_var,
        checkbox_width=22,
        checkbox_height=22,
        fg_color=accent,
        hover_color=accent_hover,
        text_color=text_main,
    ).grid(row=8, column=0, columnspan=2, sticky="w", padx=18, pady=(12, 8))

    ctk.CTkLabel(
        tab_microphone,
        text="倍频程类型",
        font=("Microsoft YaHei UI", 15, "bold"),
        text_color=accent,
    ).grid(row=9, column=0, sticky="w", padx=18, pady=(18, 8))
    ctk.CTkOptionMenu(
        tab_microphone,
        variable=mic_octave_fraction_var,
        values=["1/3 倍频程", "1/4 倍频程", "1/6 倍频程", "1/10 倍频程", "1/20 倍频程"],
        fg_color=card,
        button_color=accent,
        button_hover_color=accent_hover,
        text_color=text_main,
        dropdown_fg_color=panel_alt,
        dropdown_hover_color=soft_blue,
        dropdown_text_color=text_main,
    ).grid(row=9, column=1, sticky="ew", padx=18, pady=(18, 8))

    ctk.CTkLabel(
        tab_microphone,
        text="选择麦克风位置",
        font=("Microsoft YaHei UI", 15, "bold"),
        text_color=accent,
    ).grid(row=10, column=0, columnspan=2, sticky="w", padx=18, pady=(18, 8))
    for idx, text in enumerate(["主驾驶麦克风", "中排麦克风", "后排麦克风"]):
        ctk.CTkCheckBox(
            tab_microphone,
            text=text,
            variable=mic_position_vars[idx],
            checkbox_width=22,
            checkbox_height=22,
            fg_color=accent,
            hover_color=accent_hover,
            text_color=text_main,
        ).grid(row=11 + idx, column=0, columnspan=2, sticky="w", padx=18, pady=8)

    ctk.CTkButton(
        tab_microphone,
        text="全选麦克风图",
        command=lambda: [var.set(True) for var in mic_plot_vars.values()],
        fg_color=panel_alt,
        hover_color=soft_blue,
        text_color=text_main,
    ).grid(row=14, column=0, sticky="ew", padx=18, pady=(18, 8))
    ctk.CTkButton(
        tab_microphone,
        text="取消麦克风图",
        command=lambda: [var.set(False) for var in mic_plot_vars.values()],
        fg_color=panel_alt,
        hover_color=soft_blue,
        text_color=text_main,
    ).grid(row=14, column=1, sticky="ew", padx=18, pady=(18, 8))
    ctk.CTkButton(
        tab_microphone,
        text="全选麦克风位置",
        command=lambda: [var.set(True) for var in mic_position_vars.values()],
        fg_color=panel_alt,
        hover_color=soft_blue,
        text_color=text_main,
    ).grid(row=15, column=0, sticky="ew", padx=18, pady=(0, 8))
    ctk.CTkButton(
        tab_microphone,
        text="取消麦克风位置",
        command=lambda: [var.set(False) for var in mic_position_vars.values()],
        fg_color=panel_alt,
        hover_color=soft_blue,
        text_color=text_main,
    ).grid(row=15, column=1, sticky="ew", padx=18, pady=(0, 8))
    ctk.CTkButton(
        tab_microphone,
        text="推荐组合：三类图 + 三个麦克风位置",
        command=lambda: set_microphone_recommended_combo(),
        fg_color=panel_alt,
        hover_color=soft_blue,
        text_color=text_main,
    ).grid(row=16, column=0, columnspan=2, sticky="ew", padx=18, pady=(8, 8))
    ctk.CTkButton(
        tab_microphone,
        text="开始处理麦克风数据",
        command=lambda: run_microphone_clicked(),
        fg_color=accent,
        hover_color=accent_hover,
        text_color="#ffffff",
        font=("Microsoft YaHei UI", 15, "bold"),
        height=44,
    ).grid(row=17, column=0, columnspan=2, sticky="ew", padx=18, pady=(12, 8))
    ctk.CTkLabel(
        tab_microphone,
        text="保存、显示、DPI、图片格式和曲线样式沿用“保存与统计”“图形样式”页面的当前设置。",
        text_color=text_muted,
        wraplength=620,
        justify="left",
    ).grid(row=18, column=0, columnspan=2, sticky="w", padx=18, pady=(8, 12))

    def add_labeled_entry(parent: Any, row: int, col: int, label: str, variable: Any) -> None:
        ctk.CTkLabel(parent, text=label, text_color=label_text).grid(row=row, column=col, sticky="w", padx=18, pady=(12, 4))
        ctk.CTkEntry(parent, textvariable=variable, fg_color=card, border_color=border).grid(row=row + 1, column=col, sticky="ew", padx=18, pady=(0, 8))

    section_title(tab_params, "频域分析参数", 0)
    add_labeled_entry(tab_params, 1, 0, "频率下限 / Hz", fmin_var)
    add_labeled_entry(tab_params, 1, 1, "频率上限 / Hz", fmax_var)
    add_labeled_entry(tab_params, 3, 0, "目标频率分辨率 df / Hz", df_var)
    add_labeled_entry(tab_params, 3, 1, "分析时长 / s", duration_var)
    add_labeled_entry(tab_params, 5, 0, "起始时间 / s", start_var)
    add_labeled_entry(tab_params, 5, 1, "前裁时间 / s", trim_start_var)
    add_labeled_entry(tab_params, 7, 0, "后裁时间 / s", trim_end_var)

    ctk.CTkLabel(tab_params, text="处理模式", text_color=label_text).grid(row=7, column=1, sticky="w", padx=18, pady=(12, 4))
    ctk.CTkOptionMenu(tab_params, variable=mode_var, values=list(mode_display_to_value.keys()), fg_color=card, button_color=accent, button_hover_color=accent_hover, text_color=text_main, dropdown_fg_color=panel_alt, dropdown_hover_color=soft_blue, dropdown_text_color=text_main).grid(row=8, column=1, sticky="ew", padx=18, pady=(0, 8))
    ctk.CTkLabel(tab_params, text="输入基准", text_color=label_text).grid(row=9, column=0, sticky="w", padx=18, pady=(12, 4))
    ctk.CTkOptionMenu(tab_params, variable=input_mode_var, values=["自动", "Data第1行", "Data第2行", "Data第3行", "Data第4行"], fg_color=card, button_color=accent, button_hover_color=accent_hover, text_color=text_main, dropdown_fg_color=panel_alt, dropdown_hover_color=soft_blue, dropdown_text_color=text_main).grid(row=10, column=0, sticky="ew", padx=18, pady=(0, 8))
    ctk.CTkLabel(tab_params, text="自动输入基准和输出测点来自数据格式表；手动选择只覆盖输入行。", text_color=text_muted, wraplength=520, justify="left").grid(row=10, column=1, sticky="w", padx=18, pady=(0, 8))

    section_title(tab_save, "保存与统计设置", 0)
    ctk.CTkLabel(
        tab_save,
        text="总振级图和平均减振率图的统计频段统一使用“分析参数”里的频率下限/上限，避免重复设置。",
        text_color=text_muted,
        wraplength=620,
        justify="left",
    ).grid(row=1, column=0, columnspan=2, sticky="w", padx=18, pady=(8, 10))
    ctk.CTkLabel(tab_save, text="图片格式", text_color=label_text).grid(row=3, column=0, sticky="w", padx=18, pady=(12, 4))
    ctk.CTkOptionMenu(tab_save, variable=format_var, values=["png", "pdf", "svg", "jpg"], fg_color=card, button_color=accent, button_hover_color=accent_hover, text_color=text_main, dropdown_fg_color=panel_alt, dropdown_hover_color=soft_blue, dropdown_text_color=text_main).grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 8))
    add_labeled_entry(tab_save, 3, 1, "清晰度 DPI", dpi_var)
    ctk.CTkCheckBox(tab_save, text="出图后显示", variable=show_var, fg_color=accent, hover_color=accent_hover, text_color=text_main).grid(row=6, column=1, sticky="w", padx=18, pady=(12, 4))
    ctk.CTkCheckBox(tab_save, text="保存本次生成图像", variable=save_figures_var, fg_color=accent, hover_color=accent_hover, text_color=text_main).grid(row=7, column=0, sticky="w", padx=18, pady=(12, 4))
    ctk.CTkCheckBox(tab_save, text="导出 xlsx", variable=export_excel_var, fg_color=accent, hover_color=accent_hover, text_color=text_main).grid(row=7, column=1, sticky="w", padx=18, pady=(12, 4))
    ctk.CTkLabel(tab_save, text="保存位置", text_color=label_text).grid(row=8, column=0, sticky="w", padx=18, pady=(16, 4))
    ctk.CTkEntry(tab_save, textvariable=save_dir_var, fg_color=card, border_color=border).grid(row=9, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 8))

    def choose_save_dir() -> None:
        selected = filedialog.askdirectory(initialdir=save_dir_var.get() or str(PROJECT_ROOT))
        if selected:
            save_dir_var.set(selected)

    ctk.CTkButton(tab_save, text="浏览保存位置", command=choose_save_dir, fg_color=panel_alt, hover_color=soft_blue, text_color=text_main).grid(row=10, column=0, columnspan=2, sticky="ew", padx=18, pady=8)

    section_title(tab_style, "字体与字号", 0)
    ctk.CTkLabel(tab_style, text="字体", text_color=label_text).grid(row=1, column=0, sticky="w", padx=18, pady=(12, 4))
    ctk.CTkOptionMenu(
        tab_style,
        variable=font_family_var,
        values=["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "SimSun", "Arial", "Times New Roman"],
        fg_color=card,
        button_color=accent,
        button_hover_color=accent_hover,
        text_color=text_main,
        dropdown_fg_color=panel_alt,
        dropdown_hover_color=soft_blue,
        dropdown_text_color=text_main,
    ).grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 8))
    add_labeled_entry(tab_style, 1, 1, "标题字号", title_fontsize_var)
    add_labeled_entry(tab_style, 3, 0, "坐标轴标签字号", label_fontsize_var)
    add_labeled_entry(tab_style, 3, 1, "刻度字号", tick_fontsize_var)
    add_labeled_entry(tab_style, 5, 0, "图例字号", legend_fontsize_var)
    add_labeled_entry(tab_style, 5, 1, "标注字号", annotation_fontsize_var)

    section_title(tab_style, "线条与图面", 7)
    add_labeled_entry(tab_style, 8, 0, "曲线粗细", line_width_var)
    add_labeled_entry(tab_style, 8, 1, "曲线透明度 0-1", line_alpha_var)
    add_labeled_entry(tab_style, 10, 0, "曲线亮度系数", brightness_var)
    add_labeled_entry(tab_style, 10, 1, "网格透明度 0-1", grid_alpha_var)
    add_labeled_entry(tab_style, 12, 0, "图片宽度 inch，可留空", figure_width_var)
    add_labeled_entry(tab_style, 12, 1, "图片高度 inch，可留空", figure_height_var)

    section_title(tab_style, "标题与坐标轴标签", 14)
    add_labeled_entry(tab_style, 15, 0, "标题前缀，可留空", title_prefix_var)
    add_labeled_entry(tab_style, 15, 1, "标题后缀，可留空", title_suffix_var)
    add_labeled_entry(tab_style, 17, 0, "X轴标签覆盖，可留空", xlabel_override_var)
    add_labeled_entry(tab_style, 17, 1, "Y轴标签覆盖，可留空", ylabel_override_var)

    section_title(tab_style, "当前已选数据曲线颜色与图中名称", 19)
    ctk.CTkLabel(
        tab_style,
        text="先在左侧勾选数据文件，这里会自动列出每条曲线。可修改颜色和图中显示名称；原始数据文件名不会被改动。",
        text_color=text_muted,
        wraplength=620,
        justify="left",
    ).grid(row=20, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 8))
    ctk.CTkLabel(
        tab_style,
        text="单条曲线的“线宽”和“亮度”选择“默认”时使用上方全局曲线粗细和亮度系数。",
        text_color=text_muted,
        wraplength=620,
        justify="left",
    ).grid(row=21, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 8))
    color_rows_frame = ctk.CTkFrame(tab_style, fg_color=card, border_color=border, border_width=1, corner_radius=10)
    color_rows_frame.grid(row=22, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 18))
    color_rows_frame.grid_columnconfigure(0, weight=1)

    footer = ctk.CTkFrame(app, fg_color=panel, border_color=border, border_width=1, corner_radius=16)
    footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(8, 18))
    footer.grid_columnconfigure(0, weight=1)
    footer.grid_columnconfigure(1, weight=0)
    ctk.CTkLabel(footer, textvariable=status_var, text_color=text_main, anchor="w", justify="left").grid(row=0, column=0, sticky="ew", padx=18, pady=(12, 2))
    ctk.CTkLabel(footer, textvariable=output_info_var, text_color=text_muted, anchor="w", justify="left").grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 2))
    ctk.CTkLabel(footer, textvariable=task_summary_var, text_color=label_text, anchor="w", justify="left", wraplength=980).grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))

    def normalize_gui_color(color: str) -> str:
        try:
            import matplotlib.colors as mcolors

            return mcolors.to_hex(mcolors.to_rgba(color), keep_alpha=False)
        except Exception:
            return str(color or "#1f77b4")

    def default_curve_color_for_condition(condition_name: str) -> str:
        fixed_color = get_condition_color(condition_name)
        if fixed_color:
            return normalize_gui_color(fixed_color)
        label = normalize_condition_label(condition_name)
        palette_index = sum(ord(char) for char in label) % len(curve_default_palette)
        return curve_default_palette[palette_index]

    def get_curve_color(condition_name: str) -> str:
        if condition_name not in curve_color_values:
            curve_color_values[condition_name] = default_curve_color_for_condition(condition_name)
        return curve_color_values[condition_name]

    def get_curve_display_name_var(condition_name: str) -> Any:
        if condition_name not in curve_display_name_vars:
            default_name = condition_name[3:] if condition_name.startswith("平均 ") else condition_name
            curve_display_name_vars[condition_name] = ctk.StringVar(value=default_name)
        return curve_display_name_vars[condition_name]

    def get_curve_display_name(condition_name: str) -> str:
        var = get_curve_display_name_var(condition_name)
        value = str(var.get() or "").strip()
        return value or condition_name

    def get_curve_line_width_var(condition_name: str) -> Any:
        if condition_name not in curve_line_width_vars:
            curve_line_width_vars[condition_name] = ctk.StringVar(value="默认")
        return curve_line_width_vars[condition_name]

    def get_curve_brightness_var(condition_name: str) -> Any:
        if condition_name not in curve_brightness_vars:
            curve_brightness_vars[condition_name] = ctk.StringVar(value="默认")
        return curve_brightness_vars[condition_name]

    def get_filtered_files() -> List[Path]:
        keyword = file_search_var.get().strip().lower()
        if not keyword:
            return list(current_files)
        filtered_files: List[Path] = []
        for path in current_files:
            searchable_text = " ".join(
                [
                    path.name,
                    path.stem,
                    normalize_condition_label(path.stem),
                    get_display_condition_name(path.stem),
                ]
            ).lower()
            if keyword in searchable_text:
                filtered_files.append(path)
        return filtered_files

    def get_selected_files() -> List[Path]:
        selected: List[Path] = []
        for path in current_files:
            var = file_check_vars.get(path.name)
            if var is not None and bool(var.get()):
                selected.append(path)
        return selected

    def sync_curve_order_with_selection() -> None:
        selected_names = [path.stem for path in get_selected_files()]
        selected_set = set(selected_names)
        curve_order[:] = [name for name in curve_order if name in selected_set]
        for name in selected_names:
            if name not in curve_order:
                curve_order.append(name)

    def get_ordered_selected_files() -> List[Path]:
        sync_curve_order_with_selection()
        selected_by_name = {path.stem: path for path in get_selected_files()}
        ordered_files = [selected_by_name[name] for name in curve_order if name in selected_by_name]
        ordered_files.extend(path for path in get_selected_files() if path.stem not in set(curve_order))
        return ordered_files

    def current_style_mode() -> str:
        if (workflow_var.get() or WORKFLOW_VIBRATION) == WORKFLOW_MICROPHONE and bool(mic_average_prefix_var.get()):
            return "mic_prefix"
        if bool(average_522_523_var.get()):
            return "paired"
        return "raw"

    def build_style_targets() -> List[Dict[str, Any]]:
        mode = current_style_mode()
        targets: Dict[str, Dict[str, Any]] = {}
        for path in get_ordered_selected_files():
            stem = path.stem
            if mode == "mic_prefix":
                key = average_prefix_from_stem(stem)
            elif mode == "paired":
                key = f"平均 {stem}"
            else:
                key = stem
            if key not in targets:
                targets[key] = {"key": key, "sources": [], "first_source": stem}
                if key != stem:
                    if key not in curve_color_values and stem in curve_color_values:
                        curve_color_values[key] = curve_color_values[stem]
                    if key not in curve_display_name_vars and stem in curve_display_name_vars:
                        source_name = str(curve_display_name_vars[stem].get() or "").strip()
                        if source_name:
                            curve_display_name_vars[key] = ctk.StringVar(value=source_name)
                    if key not in curve_line_width_vars and stem in curve_line_width_vars:
                        curve_line_width_vars[key] = ctk.StringVar(value=str(curve_line_width_vars[stem].get() or "默认"))
                    if key not in curve_brightness_vars and stem in curve_brightness_vars:
                        curve_brightness_vars[key] = ctk.StringVar(value=str(curve_brightness_vars[stem].get() or "默认"))
            targets[key]["sources"].append(stem)

        target_list = list(targets.values())
        target_keys = {target["key"] for target in target_list}
        style_order[:] = [key for key in style_order if key in target_keys]
        for target in target_list:
            if target["key"] not in style_order:
                style_order.append(target["key"])
        ordered = []
        by_key = {target["key"]: target for target in target_list}
        for key in style_order:
            if key in by_key:
                ordered.append(by_key[key])
        return ordered

    def get_ordered_style_keys() -> List[str]:
        return [target["key"] for target in build_style_targets()]

    def regrid_color_rows_from_order() -> None:
        row_by_name = {
            getattr(child, "_condition_name", None): child
            for child in color_rows_frame.winfo_children()
            if getattr(child, "_condition_name", None)
        }
        for row_index, condition_name in enumerate(style_order):
            row_frame = row_by_name.get(condition_name)
            if row_frame is not None:
                row_frame.grid_configure(
                    row=row_index,
                    column=0,
                    sticky="ew",
                    padx=12,
                    pady=(10 if row_index == 0 else 8, 8),
                )

    def move_curve_order_item(condition_name: str, target_index: int) -> None:
        build_style_targets()
        if condition_name not in style_order:
            return
        old_index = style_order.index(condition_name)
        style_order.pop(old_index)
        target_index = max(0, min(target_index, len(style_order)))
        style_order.insert(target_index, condition_name)
        regrid_color_rows_from_order()

    def begin_curve_drag(condition_name: str) -> None:
        nonlocal dragging_condition_name
        dragging_condition_name = condition_name

    def end_curve_drag(event: Any) -> None:
        nonlocal dragging_condition_name
        if not dragging_condition_name:
            return
        row_widgets = []
        for child in color_rows_frame.winfo_children():
            row_name = getattr(child, "_condition_name", None)
            if row_name:
                row_widgets.append((row_name, child))
        if not row_widgets:
            dragging_condition_name = None
            return
        target_index = len(row_widgets) - 1
        for idx, (_name, widget) in enumerate(row_widgets):
            midpoint = widget.winfo_rooty() + widget.winfo_height() / 2
            if event.y_root < midpoint:
                target_index = idx
                break
        move_curve_order_item(dragging_condition_name, target_index)
        dragging_condition_name = None

    app.bind("<ButtonRelease-1>", end_curve_drag)

    def average_prefix_from_stem(stem: str) -> str:
        parts = str(stem).rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return str(stem)

    def get_microphone_octave_denominator() -> int:
        text = str(mic_octave_fraction_var.get() or "1/3")
        if "1/20" in text or "二十分之一" in text:
            return 20
        if "1/10" in text or "十分之一" in text:
            return 10
        if "1/4" in text or "四分之一" in text:
            return 4
        if "1/6" in text or "六分之一" in text:
            return 6
        return 3

    def get_microphone_octave_label() -> str:
        denominator = get_microphone_octave_denominator()
        if denominator == 20:
            return "二十分之一倍频程声压级"
        if denominator == 10:
            return "十分之一倍频程声压级"
        if denominator == 6:
            return "六分之一倍频程声压级"
        if denominator == 4:
            return "四分之一倍频程声压级"
        return "三分之一倍频程声压级"

    def get_microphone_frequency_range() -> tuple[float, float]:
        try:
            freq_min_value = float(mic_freq_min_var.get())
            freq_max_value = float(mic_freq_max_var.get())
        except ValueError as exc:
            raise ValueError("麦克风频率上下限必须填写为数字。") from exc
        if freq_min_value < 0:
            raise ValueError("麦克风频率下限不能小于 0 Hz。")
        if freq_max_value <= freq_min_value:
            raise ValueError("麦克风频率上限必须大于下限。")
        return freq_min_value, freq_max_value

    def selected_plot_type_names() -> List[str]:
        return [text for text, key in plot_items if key in plot_vars and bool(plot_vars[key].get())]

    def selected_microphone_plot_type_names() -> List[str]:
        mic_labels = {
            "fft_spl": "FFT声压级",
            "total_spl": "总声压级",
            "third_octave": get_microphone_octave_label(),
        }
        return [mic_labels[key] for key, var in mic_plot_vars.items() if bool(var.get())]

    def selected_microphone_position_names() -> List[str]:
        mic_labels = {0: "主驾驶", 1: "中排", 2: "后排"}
        return [mic_labels[idx] for idx, var in mic_position_vars.items() if bool(var.get())]

    def update_task_summary() -> None:
        selected_folder = folder_var.get() or "未选择"
        selected_count = len(get_selected_files())
        workflow = workflow_var.get() or WORKFLOW_VIBRATION
        save_state = "保存" if bool(save_figures_var.get()) else "只显示"
        freq_desc = f"{fmin_var.get()}-{fmax_var.get()} Hz"
        if workflow == WORKFLOW_MICROPHONE:
            plot_desc = "、".join(selected_microphone_plot_type_names()) or "未选麦克风图"
            mic_desc = "、".join(selected_microphone_position_names()) or "未选麦克风位置"
            avg_desc = "；按同名前缀平均" if bool(mic_average_prefix_var.get()) else ""
            mic_freq_desc = f"{mic_freq_min_var.get()}-{mic_freq_max_var.get()} Hz"
            task_summary_var.set(
                f"当前任务：{workflow} | 文件夹：{selected_folder} | 已选 {selected_count} 个 | 图：{plot_desc} | 位置：{mic_desc}{avg_desc} | 频段：{mic_freq_desc} | {save_state} | 输出：{save_dir_var.get()}"
            )
        else:
            plot_desc = "、".join(selected_plot_type_names()) or "未选结构振动图"
            avg_desc = "；5.22+5.23同名平均" if bool(average_522_523_var.get()) else ""
            input_desc = input_mode_var.get()
            task_summary_var.set(
                f"当前任务：{workflow}{avg_desc} | 文件夹：{selected_folder} | 已选 {selected_count} 个 | 输入基准：{input_desc} | 图：{plot_desc} | 频段：{freq_desc} | {save_state} | 输出：{save_dir_var.get()}"
            )

    def set_plot_option_combo(combo_name: str) -> None:
        for var in plot_vars.values():
            var.set(False)
        if combo_name == "struct_common":
            plot_vars["four_average"].set(True)
            plot_vars["six_average"].set(True)
            plot_vars["damping_rate"].set(True)
        elif combo_name == "basic_curves":
            plot_vars["single_outputs"].set(False)
            plot_vars["four_average"].set(True)
            plot_vars["six_average"].set(True)
        update_task_summary()

    def set_microphone_recommended_combo() -> None:
        for var in mic_plot_vars.values():
            var.set(True)
        for var in mic_position_vars.values():
            var.set(True)
        update_task_summary()

    def set_visible_tabs(tab_names: List[str], default_tab: str) -> None:
        try:
            tabs._segmented_button.configure(values=tab_names)
        except Exception:
            pass
        try:
            tabs.set(default_tab)
        except Exception:
            pass

    def apply_workflow_defaults() -> None:
        workflow = workflow_var.get() or WORKFLOW_VIBRATION
        if workflow == WORKFLOW_PAIRED:
            if not bool(average_522_523_var.get()):
                average_522_523_var.set(True)
            paired_average_checkbox.grid()
            paired_average_note.grid()
            target_folder = resolve_average_folder(folder_var.get())
            if target_folder:
                folder_var.set(target_folder)
            set_visible_tabs(["出图类型", "分析参数", "保存与统计", "图形样式"], "出图类型")
        elif workflow == WORKFLOW_MICROPHONE:
            if bool(average_522_523_var.get()):
                average_522_523_var.set(False)
            if (RAW_DATA_DIR / "5.31").is_dir():
                folder_var.set("5.31")
            paired_average_checkbox.grid_remove()
            paired_average_note.grid_remove()
            set_visible_tabs(["麦克风信号处理", "保存与统计", "图形样式"], "麦克风信号处理")
        else:
            if bool(average_522_523_var.get()):
                average_522_523_var.set(False)
            paired_average_checkbox.grid_remove()
            paired_average_note.grid_remove()
            set_visible_tabs(["出图类型", "分析参数", "保存与统计", "图形样式"], "出图类型")
        refresh_files(folder_var.get())
        update_task_summary()

    def on_workflow_changed(value: Optional[str] = None) -> None:
        if value:
            workflow_var.set(value)
        apply_workflow_defaults()

    def update_file_info() -> None:
        selected_folder = folder_var.get() or "未选择文件夹"
        total_count = len(current_files)
        visible_count = len(get_filtered_files())
        selected_count = len(get_selected_files())
        file_info_var.set(f"{selected_folder} · 共 {total_count} 个 .mat，当前显示 {visible_count} 个，已选 {selected_count} 个")
        status_var.set(f"已选择 {selected_count} 个数据文件。")

        update_task_summary()

    def refresh_color_rows() -> None:
        sync_curve_order_with_selection()
        style_targets = build_style_targets()
        color_swatch_widgets.clear()
        for child in color_rows_frame.winfo_children():
            child.destroy()

        if not style_targets:
            ctk.CTkLabel(
                color_rows_frame,
                text="左侧勾选数据后，这里会显示每条曲线的颜色、名称、线宽和亮度设置。",
                text_color=text_muted,
                font=("Microsoft YaHei UI", 13),
            ).grid(row=0, column=0, sticky="w", padx=14, pady=14)
            return

        for row_index, target in enumerate(style_targets):
            condition_name = str(target["key"])
            sources = [str(source) for source in target.get("sources", [])]
            current_color = get_curve_color(condition_name)
            display_name_var = get_curve_display_name_var(condition_name)
            line_width_item_var = get_curve_line_width_var(condition_name)
            brightness_item_var = get_curve_brightness_var(condition_name)

            row_frame = ctk.CTkFrame(color_rows_frame, fg_color="transparent")
            row_frame._condition_name = condition_name
            row_frame.grid(row=row_index, column=0, sticky="ew", padx=12, pady=(10 if row_index == 0 else 8, 8))
            row_frame.grid_columnconfigure(2, weight=2)
            row_frame.grid_columnconfigure(3, weight=1)
            row_frame.grid_columnconfigure(4, weight=1)

            drag_handle = ctk.CTkLabel(
                row_frame,
                text="☰",
                width=28,
                text_color=accent,
                font=("Microsoft YaHei UI", 18, "bold"),
            )
            drag_handle.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 8))
            drag_handle.bind("<ButtonPress-1>", lambda _event, name=condition_name: begin_curve_drag(name))
            drag_handle.bind("<ButtonRelease-1>", end_curve_drag)

            swatch = ctk.CTkFrame(row_frame, fg_color=current_color, width=46, height=28, corner_radius=6)
            swatch.grid(row=0, column=1, rowspan=2, sticky="w", padx=(0, 12))
            swatch.grid_propagate(False)
            swatch.configure(cursor="hand2")
            swatch.bind("<Button-1>", lambda _event, name=condition_name: choose_curve_color(name))
            color_swatch_widgets[condition_name] = swatch

            ctk.CTkLabel(
                row_frame,
                text=f"曲线：{condition_name}" if len(sources) <= 1 else f"平均组：{condition_name}  <-  {', '.join(sources)}",
                text_color=text_main,
                font=("Microsoft YaHei UI", 13),
                anchor="w",
            ).grid(row=0, column=2, columnspan=3, sticky="ew", padx=(0, 10), pady=(0, 4))

            ctk.CTkEntry(
                row_frame,
                textvariable=display_name_var,
                placeholder_text="图中显示名称",
                fg_color=panel,
                border_color=border,
                text_color=text_main,
                width=180,
            ).grid(row=1, column=2, sticky="ew", padx=(0, 8))

            ctk.CTkOptionMenu(
                row_frame,
                variable=line_width_item_var,
                values=curve_line_width_options,
                fg_color=panel,
                button_color=accent,
                button_hover_color=accent_hover,
                text_color=text_main,
                dropdown_fg_color=panel_alt,
                dropdown_hover_color=soft_blue,
                dropdown_text_color=text_main,
                width=100,
            ).grid(row=1, column=3, sticky="ew", padx=(0, 8))

            ctk.CTkOptionMenu(
                row_frame,
                variable=brightness_item_var,
                values=curve_brightness_options,
                fg_color=panel,
                button_color=accent,
                button_hover_color=accent_hover,
                text_color=text_main,
                dropdown_fg_color=panel_alt,
                dropdown_hover_color=soft_blue,
                dropdown_text_color=text_main,
                width=110,
            ).grid(row=1, column=4, sticky="ew", padx=(0, 8))

    def on_file_selection_changed() -> None:
        update_file_info()
        refresh_color_rows()

    def refresh_file_rows() -> None:
        for child in file_scroll_frame.winfo_children():
            child.destroy()

        filtered_files = get_filtered_files()
        if not filtered_files:
            ctk.CTkLabel(
                file_scroll_frame,
                text="没有匹配的数据文件。",
                text_color=text_muted,
                font=("Microsoft YaHei UI", 14),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=12)
            update_file_info()
            refresh_color_rows()
            return

        ctk.CTkLabel(
            file_scroll_frame,
            text="选择 | 文件名 | 识别工况 | 日期 | 结构 | 状态",
            text_color=accent,
            font=("Microsoft YaHei UI", 13, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 8))
        data_format_config = load_data_format_config()
        selected_folder = folder_var.get()
        selected_structure = data_format_config.get(selected_folder, {})
        structure_text = str(selected_structure.get("layout") or selected_structure.get("sensor_count") or "auto")
        side_text = str(selected_structure.get("side") or "auto")
        paired_mode = bool(average_522_523_var.get())
        other_folder = paired_average_other_folder(selected_folder) if paired_mode and selected_folder in {"5.22", "5.23"} else ""

        for row_index, path in enumerate(filtered_files):
            if path.name not in file_check_vars:
                file_check_vars[path.name] = ctk.BooleanVar(value=False)
            row_frame = ctk.CTkFrame(file_scroll_frame, fg_color=panel if row_index % 2 else card, border_color=border, border_width=1, corner_radius=8)
            row_frame.grid(row=row_index + 1, column=0, sticky="ew", padx=8, pady=5)
            row_frame.grid_columnconfigure(0, weight=1)
            status_text = "有效"
            if paired_mode and other_folder:
                status_text = "同名OK" if (RAW_DATA_DIR / other_folder / path.name).exists() else "缺少同名"
            checkbox = ctk.CTkCheckBox(
                row_frame,
                text=path.name,
                variable=file_check_vars[path.name],
                command=on_file_selection_changed,
                checkbox_width=24,
                checkbox_height=24,
                fg_color=accent,
                hover_color=accent_hover,
                text_color=text_main,
                font=("Microsoft YaHei UI", 15),
            )
            checkbox.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
            detail_text = (
                f"工况：{get_display_condition_name(path.stem)} | 日期：{path.parent.name} | "
                f"结构：{structure_text} | 激振器位置：{side_text} | 状态：{status_text}"
            )
            ctk.CTkLabel(
                row_frame,
                text=detail_text,
                text_color=text_muted,
                font=("Microsoft YaHei UI", 12),
                anchor="w",
                justify="left",
                wraplength=330,
            ).grid(row=1, column=0, sticky="ew", padx=38, pady=(0, 8))

        if workflow_var.get() == WORKFLOW_MICROPHONE and bool(mic_average_prefix_var.get()):
            selected_for_preview = get_selected_files() or filtered_files
            groups: Dict[str, List[str]] = {}
            for path in selected_for_preview:
                groups.setdefault(average_prefix_from_stem(path.stem), []).append(path.stem)
            preview_lines = [
                f"{prefix}  <-  {', '.join(names)}"
                for prefix, names in groups.items()
                if len(names) >= 1
            ]
            if preview_lines:
                ctk.CTkLabel(
                    file_scroll_frame,
                    text="麦克风同名前缀平均预览：\n" + "\n".join(preview_lines[:12]),
                    text_color=label_text,
                    font=("Microsoft YaHei UI", 12),
                    justify="left",
                    anchor="w",
                    wraplength=340,
                ).grid(row=len(filtered_files) + 2, column=0, sticky="ew", padx=10, pady=(10, 12))
        update_file_info()
        refresh_color_rows()

    def set_visible_file_selection(selected: bool) -> None:
        for path in get_filtered_files():
            if path.name in file_check_vars:
                file_check_vars[path.name].set(selected)
        on_file_selection_changed()

    def clear_visible_file_selection() -> None:
        set_visible_file_selection(False)

    def invert_visible_file_selection() -> None:
        for path in get_filtered_files():
            if path.name in file_check_vars:
                file_check_vars[path.name].set(not bool(file_check_vars[path.name].get()))
        on_file_selection_changed()

    def choose_curve_color(condition_name: str) -> None:
        current_color = get_curve_color(condition_name)
        _, selected_color = colorchooser.askcolor(color=current_color, title=f"选择 {condition_name} 曲线颜色")
        if selected_color:
            curve_color_values[condition_name] = normalize_gui_color(selected_color)
            swatch = color_swatch_widgets.get(condition_name)
            if swatch is not None:
                swatch.configure(fg_color=curve_color_values[condition_name])
            update_task_summary()

    def build_selected_color_mapping_text(average_mode: bool = False) -> str:
        lines: List[str] = []
        for condition_name in get_ordered_style_keys():
            color = get_curve_color(condition_name)
            if condition_name:
                lines.append(f"{condition_name}={color}")
        return "\n".join(lines)

    def build_selected_display_name_mapping_text(average_mode: bool = False) -> str:
        lines: List[str] = []
        for condition_name in get_ordered_style_keys():
            display_name = get_curve_display_name(condition_name)
            if display_name != condition_name:
                lines.append(f"{condition_name}={display_name}")
        return "\n".join(lines)

    def build_curve_float_mapping_text(var_getter: Any, average_mode: bool = False) -> str:
        lines: List[str] = []
        for condition_name in get_ordered_style_keys():
            value = str(var_getter(condition_name).get() or "").strip()
            if value and value != "默认":
                lines.append(f"{condition_name}={value}")
        return "\n".join(lines)

    def build_selected_condition_order(average_mode: bool = False) -> List[str]:
        return get_ordered_style_keys()

    def build_gui_plot_style(average_mode: bool = False) -> Dict[str, Any]:
        return {
            "font_family": font_family_var.get(),
            "title_fontsize": title_fontsize_var.get(),
            "label_fontsize": label_fontsize_var.get(),
            "tick_fontsize": tick_fontsize_var.get(),
            "legend_fontsize": legend_fontsize_var.get(),
            "annotation_fontsize": annotation_fontsize_var.get(),
            "line_width": line_width_var.get(),
            "line_alpha": line_alpha_var.get(),
            "brightness": brightness_var.get(),
            "grid_alpha": grid_alpha_var.get(),
            "figure_width": figure_width_var.get(),
            "figure_height": figure_height_var.get(),
            "title_prefix": title_prefix_var.get(),
            "title_suffix": title_suffix_var.get(),
            "xlabel_override": xlabel_override_var.get(),
            "ylabel_override": ylabel_override_var.get(),
            "color_mapping": build_selected_color_mapping_text(average_mode),
            "display_name_mapping": build_selected_display_name_mapping_text(average_mode),
            "line_width_mapping": build_curve_float_mapping_text(get_curve_line_width_var, average_mode),
            "brightness_mapping": build_curve_float_mapping_text(get_curve_brightness_var, average_mode),
            "condition_order": build_selected_condition_order(average_mode),
        }

    def list_data_folders() -> List[str]:
        if not RAW_DATA_DIR.exists():
            return []
        return sorted([p.name for p in RAW_DATA_DIR.iterdir() if p.is_dir()])

    def paired_average_other_folder(folder_name: str) -> str:
        return "5.23" if folder_name == "5.22" else "5.22"

    def resolve_average_folder(folder_name: str) -> str:
        if folder_name in {"5.22", "5.23"}:
            return folder_name
        for candidate in ("5.22", "5.23"):
            if (RAW_DATA_DIR / candidate).is_dir():
                folder_var.set(candidate)
                return candidate
        return folder_name

    def refresh_files(folder: Optional[str] = None) -> None:
        nonlocal current_files, file_check_vars
        selected_folder = folder or folder_var.get()
        average_mode = bool(average_522_523_var.get())
        if average_mode:
            selected_folder = resolve_average_folder(selected_folder)
        data_dir = RAW_DATA_DIR / selected_folder
        if average_mode:
            other_folder = paired_average_other_folder(selected_folder)
            other_dir = RAW_DATA_DIR / other_folder
            files = sorted(
                [
                    path
                    for path in data_dir.glob("*.mat")
                    if (other_dir / path.name).exists()
                ]
            ) if data_dir.exists() and other_dir.exists() else []
        else:
            other_folder = ""
            files = sorted(data_dir.glob("*.mat")) if data_dir.exists() else []
        current_files = files
        file_check_vars = {path.name: ctk.BooleanVar(value=False) for path in current_files}
        curve_order[:] = [path.stem for path in current_files]
        date_config = load_data_format_config().get(selected_folder, {})
        layout = date_config.get("layout", "auto")
        side = date_config.get("side", "auto")
        input_desc = "Data第2行" if side == "right" else "Data第1行" if side == "left" else "自动推断"
        average_desc = f" | 平均模式：仅显示 {selected_folder}/{other_folder} 同名文件" if average_mode else ""
        structure_info_var.set(f"数据结构：{layout} | 激振器位置：{side} | 自动输入基准：{input_desc}{average_desc}")
        file_search_var.set("")
        refresh_file_rows()
        if files:
            if average_mode:
                status_var.set(f"{selected_folder}+{other_folder}: {len(files)} 个同名 .mat 文件，默认未选择。")
            else:
                status_var.set(f"{selected_folder}: {len(files)} 个 .mat 文件，默认未选择。")
        else:
            if average_mode:
                status_var.set(f"{selected_folder}+{other_folder}: 未找到两边同名的 .mat 文件。")
            else:
                status_var.set(f"{selected_folder}: 未找到 .mat 文件。")

    def run_clicked() -> None:
        try:
            selected_folder = folder_var.get()
            selected_files = get_selected_files()
            if not selected_folder:
                messagebox.showwarning("提示", "请先选择数据文件夹。")
                return
            if not selected_files:
                messagebox.showwarning("提示", "请至少选择一个 .mat 文件。")
                return
            selected_plot_options = {key: var.get() for key, var in plot_vars.items()}
            if not any(selected_plot_options.values()):
                messagebox.showwarning("提示", "请至少选择一种出图类型。")
                return

            freq_min_value = float(fmin_var.get())
            freq_max_value = float(fmax_var.get())
            stat_min_value = freq_min_value
            stat_max_value = freq_max_value
            dpi_value = int(float(dpi_var.get()))
            if freq_max_value <= freq_min_value:
                raise ValueError("频率上限必须大于频率下限。")
            if dpi_value <= 0:
                raise ValueError("DPI 必须大于 0。")

            status_var.set("正在处理数据，请等待...")
            app.update_idletasks()

            require_runtime_dependencies()
            use_paired_average = bool(average_522_523_var.get())
            analysis_files = selected_files
            selected_names = [path.name for path in get_ordered_selected_files()]
            if use_paired_average:
                selected_folder = resolve_average_folder(selected_folder)
                if selected_folder not in {"5.22", "5.23"}:
                    messagebox.showwarning("提示", "5.22+5.23 平均模式只能使用 5.22 或 5.23 文件夹。")
                    return
                other_folder = paired_average_other_folder(selected_folder)
                missing = [
                    name
                    for name in selected_names
                    if not (RAW_DATA_DIR / "5.22" / name).exists() or not (RAW_DATA_DIR / "5.23" / name).exists()
                ]
                if missing:
                    messagebox.showwarning(
                        "提示",
                        "以下文件没有在 5.22 和 5.23 中同时存在，不能做同名平均：\n" + "\n".join(missing[:20]),
                    )
                    return
                analysis_files = [RAW_DATA_DIR / "5.22" / name for name in selected_names] + [
                    RAW_DATA_DIR / "5.23" / name for name in selected_names
                ]

            all_results, layout_info = analyze_selected_files(
                mat_files=analysis_files,
                input_mode=input_mode_var.get(),
                freq_min=freq_min_value,
                freq_max=freq_max_value,
                desired_df=float(df_var.get()),
                duration_sec=float(duration_var.get()),
                mode=mode_display_to_value.get(mode_var.get(), mode_var.get()),
                start_sec=float(start_var.get()),
                trim_start=float(trim_start_var.get()),
                trim_end=float(trim_end_var.get()),
                prefix_date=use_paired_average,
            )
            if not all_results:
                status_var.set("处理失败：没有有效结果。")
                messagebox.showwarning("提示", "没有成功处理任何数据文件，请查看终端 warning 信息。")
                return

            if use_paired_average:
                averaged_results = build_average_results_between_dates(all_results, "5.22", "5.23")
                ordered_average_names = [f"平均 {Path(name).stem}" for name in selected_names]
                averaged_results = {
                    name: averaged_results[name]
                    for name in ordered_average_names
                    if name in averaged_results
                }
                if not averaged_results:
                    status_var.set("处理失败：没有可平均的 5.22/5.23 同名结果。")
                    messagebox.showwarning("提示", "没有生成任何 5.22/5.23 同名平均结果，请检查两边数据结构和文件名。")
                    return
                missing_average = [name for name in ordered_average_names if name not in averaged_results]
                if missing_average:
                    print("warning: 以下同名数据未生成平均结果，可能是输出测点不一致：")
                    for name in missing_average:
                        print(f"  {name}")
                all_results = averaged_results

            plot_style = build_gui_plot_style(use_paired_average)

            saved_paths = save_selected_outputs(
                all_results=all_results,
                layout_info=layout_info,
                plot_options=selected_plot_options,
                save_dir=Path(save_dir_var.get()).expanduser(),
                image_format=format_var.get().lower(),
                dpi_value=dpi_value,
                freq_min=freq_min_value,
                freq_max=freq_max_value,
                show_after=show_var.get(),
                save_figures_flag=save_figures_var.get(),
                export_excel_flag=export_excel_var.get(),
                stat_point_mode=stat_point_mode_var.get(),
                stat_band_min=stat_min_value,
                stat_band_max=stat_max_value,
                plot_style=plot_style,
            )
            if save_figures_var.get() or export_excel_var.get():
                status_var.set(f"处理完成：输出 {len(saved_paths)} 个文件。")
            else:
                status_var.set("处理完成：图像已显示，未保存文件。")
            preview = "\n".join(str(path) for path in saved_paths[:12])
            if len(saved_paths) > 12:
                preview += f"\n...共 {len(saved_paths)} 个文件"
            if preview:
                messagebox.showinfo("完成", "处理完成。\n\n" + preview)
            else:
                messagebox.showinfo("完成", "处理完成：图像已显示，未保存文件。")
        except Exception as exc:
            status_var.set("处理出错。")
            print(f"error: {exc}")
            messagebox.showerror("错误", str(exc))

    def run_microphone_clicked() -> None:
        try:
            microphone_dir = RAW_DATA_DIR / "5.31"
            if bool(average_522_523_var.get()):
                average_522_523_var.set(False)
            if microphone_dir.is_dir() and folder_var.get() != "5.31":
                folder_var.set("5.31")
                refresh_files("5.31")

            selected_files = get_selected_files()
            if not selected_files:
                messagebox.showwarning("提示", "请至少选择一个麦克风 .mat 文件。")
                return

            selected_mic_options = {key: bool(var.get()) for key, var in mic_plot_vars.items()}
            if not any(selected_mic_options.values()):
                messagebox.showwarning("提示", "请至少选择一种麦克风出图类型。")
                return
            selected_mic_indices = [idx for idx, var in mic_position_vars.items() if bool(var.get())]
            if not selected_mic_indices:
                messagebox.showwarning("提示", "请至少选择一个麦克风位置。")
                return

            mic_freq_min_value, mic_freq_max_value = get_microphone_frequency_range()
            dpi_value = int(float(dpi_var.get()))
            if dpi_value <= 0:
                raise ValueError("DPI 必须大于 0。")

            status_var.set("正在处理麦克风数据，请等待...")
            app.update_idletasks()

            require_runtime_dependencies()
            microphone_results = analyze_microphone_files(
                mat_files=get_ordered_selected_files(),
                freq_min=mic_freq_min_value,
                freq_max=mic_freq_max_value,
                average_by_prefix=bool(mic_average_prefix_var.get()),
                octave_denominator=get_microphone_octave_denominator(),
            )
            if not microphone_results:
                status_var.set("麦克风处理失败：没有有效结果。")
                messagebox.showwarning("提示", "没有成功处理任何麦克风数据文件，请查看终端 warning 信息。")
                return

            saved_paths = save_microphone_outputs(
                results=microphone_results,
                plot_options=selected_mic_options,
                save_dir=Path(save_dir_var.get()).expanduser(),
                image_format=format_var.get().lower(),
                dpi_value=dpi_value,
                show_after=show_var.get(),
                save_figures_flag=save_figures_var.get(),
                plot_style=build_gui_plot_style(bool(mic_average_prefix_var.get())),
                freq_min=mic_freq_min_value,
                freq_max=mic_freq_max_value,
                mic_indices=selected_mic_indices,
                octave_denominator=get_microphone_octave_denominator(),
            )
            if save_figures_var.get():
                status_var.set(f"麦克风处理完成：输出 {len(saved_paths)} 个文件。")
            else:
                status_var.set("麦克风处理完成：图像已显示，未保存文件。")

            preview = "\n".join(str(path) for path in saved_paths[:12])
            if len(saved_paths) > 12:
                preview += f"\n...共 {len(saved_paths)} 个文件"
            if preview:
                messagebox.showinfo("完成", "麦克风处理完成。\n\n" + preview)
            else:
                messagebox.showinfo("完成", "麦克风处理完成：图像已显示，未保存文件。")
        except Exception as exc:
            status_var.set("麦克风处理出错。")
            print(f"error: {exc}")
            messagebox.showerror("错误", str(exc))

    def run_current_workflow() -> None:
        if (workflow_var.get() or WORKFLOW_VIBRATION) == WORKFLOW_MICROPHONE:
            run_microphone_clicked()
        else:
            run_clicked()

    ctk.CTkButton(
        footer,
        text="开始处理并出图",
        command=run_current_workflow,
        fg_color=accent,
        hover_color=accent_hover,
        text_color="#ffffff",
        font=("Microsoft YaHei UI", 15, "bold"),
        height=46,
    ).grid(row=0, column=1, rowspan=3, sticky="e", padx=18, pady=12)

    file_search_var.trace_add("write", lambda *_: refresh_file_rows())
    average_522_523_var.trace_add("write", lambda *_: refresh_files(folder_var.get()))
    mic_average_prefix_var.trace_add("write", lambda *_: refresh_file_rows())
    workflow_var.trace_add("write", lambda *_: update_task_summary())
    for traced_var in list(plot_vars.values()) + list(mic_plot_vars.values()) + list(mic_position_vars.values()):
        traced_var.trace_add("write", lambda *_: update_task_summary())
    for traced_var in [save_figures_var, export_excel_var, show_var, fmin_var, fmax_var, input_mode_var, format_var, dpi_var, save_dir_var, mic_octave_fraction_var, mic_freq_min_var, mic_freq_max_var]:
        traced_var.trace_add("write", lambda *_: update_task_summary())

    folders = list_data_folders()
    folder_menu.configure(values=folders, command=refresh_files)
    if folders:
        folder_var.set(folders[0])
        refresh_files(folders[0])
    apply_workflow_defaults()
    app.mainloop()
