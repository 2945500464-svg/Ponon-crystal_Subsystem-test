from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import CHANNEL_LAYOUTS, DEFAULT_DATE_DATA_FORMATS, RAW_DATA_DIR
from .utils import build_channel_names, get_date_tag

_DATA_FORMAT_TABLE_CACHE: Optional[Dict[str, Dict[str, str]]] = None

def _parse_side(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if "右" in text or "right" in text:
        return "right"
    if "左" in text or "left" in text:
        return "left"
    return None


def _parse_layout_key(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if "力传感器" in text and "右输入" in text and "声子晶体" in text and "10" in text:
        return "force_right_input_pc10"
    if "力传感器" in text and "左中" in text and "右中" in text and "9" in text:
        return "force_input_mid9"
    if "17" in text or "实车" in text or "半轴" in text:
        return "vehicle17"
    if "10个" in text or "10" in text:
        return "new10"
    if "9个" in text or "9" in text:
        return "force9"
    if "8个" in text or "8" in text:
        return "old8"
    return None


def _parse_input_index(value: Any) -> Optional[int]:
    """从数据格式表的通道描述中识别输入基准行号。"""
    text = str(value or "").strip()
    if not text:
        return None

    explicit_match = re.search(r"Data\s*第?\s*(\d+)\s*行", text, flags=re.IGNORECASE)
    if explicit_match:
        return max(int(explicit_match.group(1)) - 1, 0)

    compact = re.sub(r"\s+", "", text)
    for match in re.finditer(r"(\d+)[：:、，,;；-]*([^0-9]+)", compact):
        row_number = int(match.group(1))
        channel_text = match.group(2)
        if "左输入" in channel_text or "右输入" in channel_text or "输入加速度" in channel_text or "激振器加速度" in channel_text:
            return max(row_number - 1, 0)
    return None


def load_data_format_config() -> Dict[str, Dict[str, str]]:
    """Read Raw_data/数据格式.xlsx and merge with built-in defaults."""
    global _DATA_FORMAT_TABLE_CACHE
    if _DATA_FORMAT_TABLE_CACHE is not None:
        return _DATA_FORMAT_TABLE_CACHE

    config = {key: value.copy() for key, value in DEFAULT_DATE_DATA_FORMATS.items()}
    table_files = sorted(
        path for path in RAW_DATA_DIR.glob("*.xlsx")
        if not path.name.startswith("~$")
    )
    if not table_files:
        _DATA_FORMAT_TABLE_CACHE = config
        return config

    try:
        import openpyxl

        workbook = openpyxl.load_workbook(table_files[0], data_only=True)
        worksheet = workbook.active
        rows = list(worksheet.iter_rows(values_only=True))
        if not rows:
            _DATA_FORMAT_TABLE_CACHE = config
            return config

        headers = [str(cell or "").strip() for cell in rows[0]]
        header_index = {name: idx for idx, name in enumerate(headers)}
        folder_idx = header_index.get("文件夹命名", 0)
        side_idx = header_index.get("激振器位置", 1)
        sensor_idx = header_index.get("传感器数量", 2)

        last_layout: Optional[str] = None
        for row in rows[1:]:
            if not row:
                continue
            folder = str(row[folder_idx] or "").strip()
            if not folder:
                continue

            side = _parse_side(row[side_idx] if side_idx < len(row) else None)
            sensor_value = row[sensor_idx] if sensor_idx < len(row) else None
            layout = _parse_layout_key(sensor_value)
            input_index = _parse_input_index(sensor_value)
            if layout is not None:
                last_layout = layout

            current = config.get(folder, {}).copy()
            if side is not None:
                current["side"] = side
            if layout is not None:
                current["layout"] = layout
            elif "layout" not in current and last_layout is not None:
                current["layout"] = last_layout
            if input_index is not None:
                current["input_index"] = str(input_index)
            config[folder] = current
    except Exception as exc:
        print(f"  warning: 数据格式表读取失败，使用内置日期结构配置。原因: {exc}")

    _DATA_FORMAT_TABLE_CACHE = config
    return config


def infer_layout_key(date_tag: str, n_channels: Optional[int] = None) -> str:
    date_config = load_data_format_config().get(date_tag, {})
    layout = date_config.get("layout")
    if layout in CHANNEL_LAYOUTS:
        return layout
    if n_channels is not None:
        if n_channels >= 17:
            return "vehicle17"
        if n_channels >= 10:
            return "new10"
        if n_channels == 9:
            return "force9"
        return "old8"
    return "new10"


def get_channel_layout(mat_file: Path, n_channels: Optional[int] = None) -> Dict[str, Any]:
    return CHANNEL_LAYOUTS[infer_layout_key(get_date_tag(mat_file), n_channels=n_channels)]


def get_input_channel_index_for_file(mat_file: Path) -> int:
    """Automatic input reference: left exciter -> Data row 1, right exciter -> Data row 2."""
    date_config = load_data_format_config().get(get_date_tag(mat_file), {})
    if "input_index" in date_config:
        try:
            return int(date_config["input_index"])
        except (TypeError, ValueError):
            pass
    side = date_config.get("side")
    return 1 if side == "right" else 0


def get_channel_config_for_file(mat_file: Path) -> Tuple[List[str], List[int], List[int], Optional[int]]:
    layout = get_channel_layout(mat_file)
    return (
        list(layout["names"]),
        list(layout["output_indices"]),
        list(layout["four_indices"]),
        layout.get("exciter_index"),
    )


def get_gui_channel_config(mat_file: Path, n_channels: int) -> Dict[str, Any]:
    layout = get_channel_layout(mat_file, n_channels=n_channels)
    force_idx = layout.get("force_index")
    exciter_idx = layout.get("exciter_index")
    return {
        "layout_key": infer_layout_key(get_date_tag(mat_file), n_channels=n_channels),
        "names": build_channel_names(n_channels, list(layout["names"])),
        "output_indices": [idx for idx in layout["output_indices"] if idx < n_channels],
        "four_indices": [idx for idx in layout["four_indices"] if idx < n_channels],
        "six_indices": [idx for idx in layout["six_indices"] if idx < n_channels],
        "force_index": force_idx if force_idx is not None and force_idx < n_channels else None,
        "exciter_index": exciter_idx if exciter_idx is not None and exciter_idx < n_channels else None,
    }


def get_heatmap_sensor_indices(layout: Dict[str, Any]) -> List[int]:
    """热力图测点选择：含左中/右中时用六点，否则用四点。"""
    names = [str(name) for name in layout.get("names", [])]
    if "左中" in names and "右中" in names:
        return list(layout.get("six_indices") or layout.get("output_indices") or [])
    return list(layout.get("four_indices") or layout.get("output_indices") or [])


def resolve_gui_input_index(mat_file: Path, input_mode: str) -> int:
    explicit_map = {
        "自动": -1,
        "Data第1行（力传感器）": 0,
        "Data第2行（左输入）": 1,
        "Data第2行（右输入）": 1,
        "Data第2行（输入）": 1,
        "第一行力传感器": 0,
        "第二行左输入": 1,
        "第二行右输入": 1,
        "Data第1行": 0,
        "Data第2行": 1,
        "Data第3行": 2,
        "Data第4行": 3,
        "Data row 1": 0,
        "Data row 2": 1,
        "Data row 3": 2,
        "Data row 4": 3,
    }
    if input_mode in explicit_map and explicit_map[input_mode] >= 0:
        return explicit_map[input_mode]
    return get_input_channel_index_for_file(mat_file)
