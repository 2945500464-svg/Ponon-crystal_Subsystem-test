from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# 固定采样频率 / Hz
fs = 48000

# 频率分析范围 / Hz
fmin = 70
fmax = 140

# 目标 FFT 频率分辨率 / Hz
target_df = 1 / 20

# 单次分析时长 / s
analysis_duration_sec = 20.0

# 处理模式："single_segment" 或 "segment_average"
analysis_mode = "single_segment"

# 分析起始时间 / s
start_time_sec = 0.0

# 原始数据预裁剪：截去前后指定时长后，再对中间有效数据做分析
trim_start_sec = 0.0
trim_end_sec = 0.0

# 是否保存图片
save_figures = False

# 图片保存分辨率
figure_save_dpi = 600

# 图例字体大小
legend_fontsize = 12

# 是否导出 Excel
save_excel = False

# 是否显示图片
show_figures = True

# 通道名称：第 1 个名称对应 Data 第 1 行，即输入通道
channel_names = [
    "左输入",
    "右输入",
    "输入力",
    "激振器加速度",
    "左前",
    "左后",
    "左动力吸振器",
    "右前",
    "右后",
    "右动力吸振器",
]

# 输入点 FRF：Data 第 1 行为输入点加速度，Data 第 3 行为新增力传感器
input_accel_channel_index = 1
right_input_channel_index = 1
force_channel_index = 2
exciter_accel_channel_index = 3
save_frf_figure = False

# 六测点加权传递率图：左前、左后、左中、右前、右后、右中，默认等权
weighted_transfer_sensor_names = ["左前", "左后", "左中", "右前", "右后", "右中"]
weighted_transfer_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

channel_names_58 = [
    "左输入",
    "右输入",
    "输入力",
    "左前",
    "左后",
    "左动力吸振器",
    "右前",
    "右后",
    "右动力吸振器",
]

channel_names_510 = channel_names

channel_names_511 = [
    "左输入",
    "右输入",
    "输入力",
    "激振器加速度",
    "左前",
    "左后",
    "左中",
    "右前",
    "右后",
    "右中",
]

OLD_8_CHANNEL_NAMES = [
    "左输入",
    "右输入",
    "左前",
    "左后",
    "左动力吸振器",
    "右前",
    "右后",
    "右动力吸振器",
]

FORCE_9_CHANNEL_NAMES = [
    "左输入",
    "右输入",
    "力传感器",
    "左前",
    "左后",
    "左动力吸振器",
    "右前",
    "右后",
    "右动力吸振器",
]

NEW_10_CHANNEL_NAMES = [
    "左输入",
    "右输入",
    "力传感器",
    "激振器加速度",
    "左前",
    "左后",
    "左中",
    "右前",
    "右后",
    "右中",
]

VEHICLE_17_CHANNEL_NAMES = [
    "前排麦克风",
    "中排麦克风",
    "后排麦克风",
    "左半轴X",
    "左半轴Y",
    "左半轴Z",
    "右半轴X",
    "右半轴Y",
    "右半轴Z",
    "左前",
    "左中",
    "左后",
    "右前",
    "右中",
    "右后",
    "左动力吸振器",
    "右动力吸振器",
]

FORCE_INPUT_MID_9_CHANNEL_NAMES = [
    "力传感器",
    "左输入",
    "右动力吸振器",
    "左前",
    "左中",
    "左后",
    "右前",
    "右中",
    "右后",
]

CHANNEL_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "old8": {
        "names": OLD_8_CHANNEL_NAMES,
        "output_indices": [2, 3, 4, 5, 6, 7],
        "four_indices": [2, 3, 5, 6],
        "six_indices": [2, 3, 4, 5, 6, 7],
        "force_index": None,
        "exciter_index": None,
    },
    "force9": {
        "names": FORCE_9_CHANNEL_NAMES,
        "output_indices": [3, 4, 5, 6, 7, 8],
        "four_indices": [3, 4, 6, 7],
        "six_indices": [3, 4, 5, 6, 7, 8],
        "force_index": 2,
        "exciter_index": None,
    },
    "new10": {
        "names": NEW_10_CHANNEL_NAMES,
        "output_indices": [4, 5, 6, 7, 8, 9],
        "four_indices": [4, 5, 7, 8],
        "six_indices": [4, 5, 6, 7, 8, 9],
        "force_index": 2,
        "exciter_index": 3,
    },
    "vehicle17": {
        "names": VEHICLE_17_CHANNEL_NAMES,
        "output_indices": [9, 10, 11, 12, 13, 14, 15, 16],
        "four_indices": [9, 11, 12, 14],
        "six_indices": [9, 10, 11, 12, 13, 14],
        "force_index": None,
        "exciter_index": None,
    },
    "force_input_mid9": {
        "names": FORCE_INPUT_MID_9_CHANNEL_NAMES,
        "output_indices": [2, 3, 4, 5, 6, 7, 8],
        "four_indices": [3, 5, 6, 8],
        "six_indices": [3, 4, 5, 6, 7, 8],
        "force_index": 0,
        "exciter_index": None,
    },
}

DEFAULT_DATE_DATA_FORMATS: Dict[str, Dict[str, str]] = {
    "4.24": {"side": "left", "layout": "old8"},
    "5.8": {"side": "left", "layout": "force9"},
    "5.10": {"side": "left", "layout": "force9"},
    "5.11": {"side": "left", "layout": "new10"},
    "5.12": {"side": "right", "layout": "new10"},
    "5.22": {"side": "right", "layout": "new10"},
    "5.23": {"side": "left", "layout": "new10"},
    "6.11": {"side": "none", "layout": "vehicle17"},
    "6.23": {"side": "left", "layout": "force_input_mid9", "input_index": "1"},
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "Raw_data"
RESULTS_DIR = PROJECT_ROOT / "output"
XLSX_DIR = RESULTS_DIR / "xlsx"
FIGURES_DIR = RESULTS_DIR / "figures"
GUI_THEME_PATH = PROJECT_ROOT / "themes" / "clean_breeze.json"


def require_runtime_dependencies() -> None:
    """检查运行所需第三方依赖。"""
    import importlib.util

    required = ["matplotlib", "numpy", "pandas", "scipy"]
    if save_excel:
        required.append("openpyxl")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        missing_text = ", ".join(sorted(set(missing)))
        raise ImportError(f"缺少 Python 依赖：{missing_text}。请先执行：pip install -r requirements.txt")


def ensure_directories() -> None:
    """创建项目所需目录。"""
    for folder in [RAW_DATA_DIR, RESULTS_DIR, XLSX_DIR, FIGURES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
