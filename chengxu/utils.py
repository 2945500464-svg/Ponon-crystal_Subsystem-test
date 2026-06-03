from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import RAW_DATA_DIR

def sanitize_filename(name: str, fallback: str = "output") -> str:
    """将通道名转换为安全文件名。"""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip().strip(".")
    return safe or fallback


def sanitize_sheet_name(name: str) -> str:
    """将通道名转换为 Excel 工作表名。"""
    safe = re.sub(r"[\[\]:*?/\\]", "_", str(name)).strip()
    return (safe or "TL")[:31]


def build_channel_names(n_channels: int, configured_names: List[str]) -> List[str]:
    """根据实际通道数生成通道名称，缺失或多余通道自动兜底。"""
    names: List[str] = []
    for idx in range(n_channels):
        if idx < len(configured_names) and str(configured_names[idx]).strip():
            names.append(str(configured_names[idx]).strip())
        else:
            names.append(f"Ch_{idx + 1}")
    return names


def get_date_tag(mat_file: Path) -> str:
    """根据 .mat 所在子文件夹生成日期前缀。"""
    return mat_file.parent.name if mat_file.parent != RAW_DATA_DIR else ""


def get_condition_plot_order(all_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """绘图顺序：no-load 第一，DVA 第二，plan 按数字顺序，其余放最后。"""
    def sort_key(condition_name: str) -> Tuple[int, int, str]:
        label = normalize_condition_label(condition_name)
        if label == "no-load":
            return (0, 0, condition_name)
        if label == "DVA":
            return (1, 0, condition_name)

        match = re.search(r"plan(\d+)", label, flags=re.IGNORECASE)
        if match:
            return (2, int(match.group(1)), condition_name)

        return (3, 0, condition_name)

    return sorted(all_results.keys(), key=sort_key)


def normalize_condition_label(condition_name: str) -> str:
    """将文件名形式的工况名归一到 no-load / DVA / planN，便于热力图排序。"""
    lower_name = condition_name.lower().replace("_", "-").replace(" ", "-")
    if "no-load" in lower_name or "noload" in lower_name:
        return "no-load"
    if "dva" in lower_name:
        return "DVA"

    match = re.search(r"plan[-_ ]*(\d+)", condition_name, flags=re.IGNORECASE)
    if match:
        return f"plan{int(match.group(1))}"

    return condition_name


def get_display_condition_name(condition_name: str) -> str:
    """图例显示名统一使用数据文件原始名称。"""
    return str(condition_name)


def get_condition_color(condition_name: str) -> Optional[str]:
    """固定关键工况颜色，其余工况交给 matplotlib 自动分配不同颜色。"""
    if condition_name.startswith("5.8-"):
        date_colors = {
            "no-load": "#000000",
            "DVA": "#d62728",
            "plan1": "#1f77b4",
            "plan2": "#2ca02c",
            "plan3": "#9467bd",
            "plan4": "#ff7f0e",
            "plan5": "#17becf",
            "plan6": "#8c564b",
        }
        return date_colors.get(normalize_condition_label(condition_name))
    if condition_name.startswith("5.10-"):
        date_colors = {
            "no-load": "#555555",
            "DVA": "#ff9896",
            "plan1": "#aec7e8",
            "plan2": "#98df8a",
            "plan3": "#c5b0d5",
            "plan4": "#ffbb78",
            "plan5": "#9edae5",
            "plan6": "#c49c94",
        }
        return date_colors.get(normalize_condition_label(condition_name))
    if condition_name.startswith("5.11-"):
        date_colors = {
            "no-load": "#222222",
            "DVA": "#8c1d18",
            "plan1": "#174a7e",
            "plan2": "#1b6e1f",
            "plan3": "#6d3f99",
            "plan4": "#b85f00",
            "plan5": "#0f7f8f",
            "plan6": "#5f3b32",
        }
        return date_colors.get(normalize_condition_label(condition_name))

    label = normalize_condition_label(condition_name)
    fixed_colors = {
        "no-load": "black",
        "DVA": "red",
        "plan1": "#1f77b4",
        "plan2": "#2ca02c",
        "plan3": "#9467bd",
        "plan4": "#ff7f0e",
        "plan5": "#17becf",
        "plan6": "#8c564b",
    }
    return fixed_colors.get(label)
