from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager as mpl_font_manager

from .config import legend_fontsize
from .utils import get_display_condition_name, normalize_condition_label

_FONT_REGISTRATION_DONE = False

def _safe_style_float(value: Any, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        result = default
    if min_value is not None:
        result = max(result, min_value)
    if max_value is not None:
        result = min(result, max_value)
    return result


def _safe_style_int(value: Any, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    result = int(round(_safe_style_float(value, float(default), None, None)))
    if min_value is not None:
        result = max(result, min_value)
    if max_value is not None:
        result = min(result, max_value)
    return result


def parse_color_mapping(mapping_text: str) -> Dict[str, str]:
    """Parse GUI color mapping lines like no-load=#000000."""
    color_map: Dict[str, str] = {}
    try:
        import matplotlib.colors as mcolors
    except Exception:
        return color_map

    for raw_line in str(mapping_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if not key or not value:
            continue
        try:
            normalized_color = mcolors.to_hex(mcolors.to_rgba(value), keep_alpha=False)
        except ValueError:
            print(f"  warning: 颜色配置无效，已忽略: {line}")
            continue

        candidate_keys = {
            key,
            key.lower(),
            normalize_condition_label(key),
            normalize_condition_label(key).lower(),
            get_display_condition_name(key),
            get_display_condition_name(normalize_condition_label(key)),
        }
        for candidate in candidate_keys:
            if candidate:
                color_map[str(candidate)] = normalized_color
                color_map[str(candidate).lower()] = normalized_color
    return color_map


def parse_display_name_mapping(mapping_text: str) -> Dict[str, str]:
    """Parse GUI display-name mapping lines like plan1=方案1."""
    display_map: Dict[str, str] = {}
    for raw_line in str(mapping_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if not key or not value:
            continue
        display_map[key] = value
        display_map[key.lower()] = value
    return display_map


def parse_float_mapping(mapping_text: str, min_value: float, max_value: float) -> Dict[str, float]:
    """Parse style mapping lines like plan1=2.4."""
    value_map: Dict[str, float] = {}
    for raw_line in str(mapping_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if not key or not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            print(f"  warning: 曲线样式配置无效，已忽略: {line}")
            continue
        parsed = max(min_value, min(max_value, parsed))
        value_map[key] = parsed
        value_map[key.lower()] = parsed
    return value_map


def _adjust_color_brightness(color: Any, brightness: float) -> Any:
    if color is None:
        return color
    try:
        import matplotlib.colors as mcolors

        rgba = mcolors.to_rgba(color)
        rgb = tuple(min(1.0, max(0.0, channel * brightness)) for channel in rgba[:3])
        return (*rgb, rgba[3])
    except Exception:
        return color


def _lookup_style_color(label: str, color_map: Dict[str, str]) -> Optional[str]:
    candidates = [
        str(label),
        str(label).lower(),
        normalize_condition_label(str(label)),
        normalize_condition_label(str(label)).lower(),
        get_display_condition_name(str(label)),
        get_display_condition_name(normalize_condition_label(str(label))),
    ]
    for candidate in candidates:
        if candidate in color_map:
            return color_map[candidate]
        if str(candidate).lower() in color_map:
            return color_map[str(candidate).lower()]
    return None


def _lookup_style_float(label: str, value_map: Dict[str, float]) -> Optional[float]:
    label_text = str(label)
    candidates = [
        label_text,
        label_text.lower(),
        normalize_condition_label(label_text),
        normalize_condition_label(label_text).lower(),
        get_display_condition_name(label_text),
        get_display_condition_name(normalize_condition_label(label_text)),
    ]
    for candidate in candidates:
        if candidate in value_map:
            return value_map[candidate]
        if str(candidate).lower() in value_map:
            return value_map[str(candidate).lower()]
    return None


def _lookup_style_display_name(label: str, display_map: Dict[str, str]) -> Optional[str]:
    label_text = str(label)
    candidates = [label_text, label_text.lower()]
    for candidate in candidates:
        if candidate in display_map:
            return display_map[candidate]
    return None


def register_matplotlib_chinese_fonts() -> None:
    """Register common Windows Chinese fonts so matplotlib output does not show tofu/garbled glyphs."""
    global _FONT_REGISTRATION_DONE
    if _FONT_REGISTRATION_DONE or mpl_font_manager is None:
        return

    font_files = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/Deng.ttf"),
    ]
    for font_file in font_files:
        if not font_file.exists():
            continue
        try:
            mpl_font_manager.fontManager.addfont(str(font_file))
        except Exception:
            pass
    _FONT_REGISTRATION_DONE = True


def matplotlib_font_exists(font_family: str) -> bool:
    if not font_family or mpl_font_manager is None:
        return False
    try:
        font_path = mpl_font_manager.findfont(
            mpl_font_manager.FontProperties(family=[font_family]),
            fallback_to_default=False,
        )
        return bool(font_path and Path(font_path).exists())
    except Exception:
        return False


def resolve_plot_font_family(preferred_font: str) -> Tuple[str, List[str]]:
    """Return an available font family and a fallback list for Chinese plots."""
    register_matplotlib_chinese_fonts()
    candidates: List[str] = []
    for font_name in [
        preferred_font,
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "SimSun",
        "DengXian",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]:
        font_name = str(font_name or "").strip()
        if font_name and font_name not in candidates:
            candidates.append(font_name)

    available = [font_name for font_name in candidates if matplotlib_font_exists(font_name)]
    if not available:
        available = ["DejaVu Sans"]

    fallback_list = available + [font_name for font_name in candidates if font_name not in available]
    return available[0], fallback_list


def normalize_plot_style(plot_style: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict(plot_style or {})
    width_text = str(raw.get("figure_width", "")).strip()
    height_text = str(raw.get("figure_height", "")).strip()
    if isinstance(raw.get("color_map"), dict):
        color_map = dict(raw["color_map"])
    else:
        color_map = parse_color_mapping(str(raw.get("color_mapping") or ""))
    if isinstance(raw.get("display_name_map"), dict):
        display_name_map = dict(raw["display_name_map"])
    else:
        display_name_map = parse_display_name_mapping(str(raw.get("display_name_mapping") or ""))
    if isinstance(raw.get("line_width_map"), dict):
        line_width_map = dict(raw["line_width_map"])
    else:
        line_width_map = parse_float_mapping(str(raw.get("line_width_mapping") or ""), 0.1, 10.0)
    if isinstance(raw.get("brightness_map"), dict):
        brightness_map = dict(raw["brightness_map"])
    else:
        brightness_map = parse_float_mapping(str(raw.get("brightness_mapping") or ""), 0.1, 3.0)
    requested_font_family = str(raw.get("font_family") or "Microsoft YaHei UI").strip() or "Microsoft YaHei UI"
    resolved_font_family, font_family_list = resolve_plot_font_family(requested_font_family)
    raw_condition_order = raw.get("condition_order") or []
    if isinstance(raw_condition_order, (list, tuple)):
        condition_order = [str(item) for item in raw_condition_order if str(item).strip()]
    else:
        condition_order = [line.strip() for line in str(raw_condition_order).splitlines() if line.strip()]
    return {
        "font_family": resolved_font_family,
        "font_family_list": font_family_list,
        "title_fontsize": _safe_style_int(raw.get("title_fontsize"), 14, 6, 60),
        "label_fontsize": _safe_style_int(raw.get("label_fontsize"), 12, 6, 60),
        "tick_fontsize": _safe_style_int(raw.get("tick_fontsize"), 10, 5, 50),
        "legend_fontsize": _safe_style_int(raw.get("legend_fontsize"), legend_fontsize, 5, 50),
        "annotation_fontsize": _safe_style_int(raw.get("annotation_fontsize"), 9, 5, 50),
        "line_width": _safe_style_float(raw.get("line_width"), 1.6, 0.1, 10.0),
        "line_alpha": _safe_style_float(raw.get("line_alpha"), 1.0, 0.05, 1.0),
        "brightness": _safe_style_float(raw.get("brightness"), 1.0, 0.1, 3.0),
        "grid_alpha": _safe_style_float(raw.get("grid_alpha"), 0.45, 0.0, 1.0),
        "figure_width": _safe_style_float(width_text, 0.0, 0.0, 40.0) if width_text else 0.0,
        "figure_height": _safe_style_float(height_text, 0.0, 0.0, 30.0) if height_text else 0.0,
        "title_prefix": str(raw.get("title_prefix") or ""),
        "title_suffix": str(raw.get("title_suffix") or ""),
        "xlabel_override": str(raw.get("xlabel_override") or "").strip(),
        "ylabel_override": str(raw.get("ylabel_override") or "").strip(),
        "color_map": color_map,
        "display_name_map": display_name_map,
        "line_width_map": line_width_map,
        "brightness_map": brightness_map,
        "condition_order": condition_order,
    }


def apply_plot_style(fig: plt.Figure, plot_style: Optional[Dict[str, Any]]) -> None:
    """Apply user-selected visual style to a generated matplotlib figure."""
    if fig is None:
        return
    style = normalize_plot_style(plot_style)

    width = style["figure_width"]
    height = style["figure_height"]
    if width > 0 and height > 0:
        fig.set_size_inches(width, height, forward=True)

    color_map: Dict[str, str] = style["color_map"]
    display_name_map: Dict[str, str] = style["display_name_map"]
    line_width_map: Dict[str, float] = style["line_width_map"]
    brightness_map: Dict[str, float] = style["brightness_map"]
    brightness = style["brightness"]
    line_alpha = style["line_alpha"]
    line_width = style["line_width"]

    for ax in fig.axes:
        is_colorbar = str(ax.get_label()).startswith("<colorbar")
        title = ax.get_title()
        if title and not is_colorbar:
            ax.set_title(f"{style['title_prefix']}{title}{style['title_suffix']}", fontsize=style["title_fontsize"], fontfamily=style["font_family"])
        elif title:
            ax.title.set_fontsize(style["title_fontsize"])

        if not is_colorbar and style["xlabel_override"]:
            ax.set_xlabel(style["xlabel_override"])
        if not is_colorbar and style["ylabel_override"]:
            ax.set_ylabel(style["ylabel_override"])

        ax.xaxis.label.set_fontsize(style["label_fontsize"])
        ax.yaxis.label.set_fontsize(style["label_fontsize"])
        ax.xaxis.label.set_fontfamily(style["font_family"])
        ax.yaxis.label.set_fontfamily(style["font_family"])
        ax.tick_params(axis="both", labelsize=style["tick_fontsize"])
        for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick_label.set_fontfamily(style["font_family"])

        x_tick_labels = ax.get_xticklabels()
        mapped_tick_texts: List[str] = []
        has_mapped_tick = False
        for tick_label in x_tick_labels:
            original_text = tick_label.get_text()
            mapped_text = _lookup_style_display_name(original_text, display_name_map)
            if mapped_text:
                mapped_tick_texts.append(mapped_text)
                has_mapped_tick = True
            else:
                mapped_tick_texts.append(original_text)
        if has_mapped_tick and len(mapped_tick_texts) == len(ax.get_xticks()):
            rotation = x_tick_labels[0].get_rotation() if x_tick_labels else 0
            ha = x_tick_labels[0].get_ha() if x_tick_labels else "center"
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(mapped_tick_texts, rotation=rotation, ha=ha, fontfamily=style["font_family"], fontsize=style["tick_fontsize"])

        for line in ax.get_lines():
            label = line.get_label()
            mapped_color = _lookup_style_color(label, color_map)
            if mapped_color:
                line.set_color(mapped_color)
            mapped_label = _lookup_style_display_name(label, display_name_map)
            if mapped_label and not str(label).startswith("_"):
                line.set_label(mapped_label)
            line_brightness = _lookup_style_float(label, brightness_map)
            line_width_value = _lookup_style_float(label, line_width_map)
            line.set_color(_adjust_color_brightness(line.get_color(), line_brightness if line_brightness is not None else brightness))
            line.set_linewidth(line_width_value if line_width_value is not None else line_width)
            line.set_alpha(line_alpha)

        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        for patch_index, patch in enumerate(ax.patches):
            if patch_index < len(xtick_labels):
                mapped_color = _lookup_style_color(xtick_labels[patch_index], color_map)
                if mapped_color:
                    patch.set_facecolor(mapped_color)
            patch.set_facecolor(_adjust_color_brightness(patch.get_facecolor(), brightness))
            patch.set_alpha(line_alpha)

        for axis_name, tick_getter, tick_setter, position_getter in [
            ("x", ax.get_xticklabels, ax.set_xticklabels, ax.get_xticks),
            ("y", ax.get_yticklabels, ax.set_yticklabels, ax.get_yticks),
        ]:
            tick_labels = list(tick_getter())
            original_texts = [tick.get_text() for tick in tick_labels]
            mapped_texts = [
                _lookup_style_display_name(text, display_name_map) or text
                for text in original_texts
            ]
            if mapped_texts != original_texts:
                if axis_name == "x":
                    ax.set_xticks(position_getter())
                    tick_setter(mapped_texts)
                else:
                    ax.set_yticks(position_getter())
                    tick_setter(mapped_texts)

        for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick_label.set_fontsize(style["tick_fontsize"])
            tick_label.set_fontfamily(style["font_family"])

        for text in ax.texts:
            text.set_fontsize(style["annotation_fontsize"])
            text.set_fontfamily(style["font_family"])

        legend = ax.get_legend()
        if legend is not None:
            legend_handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
            for legend_text, legend_handle in zip(legend.get_texts(), legend_handles):
                original_legend_label = legend_text.get_text()
                legend_text.set_fontsize(style["legend_fontsize"])
                legend_text.set_fontfamily(style["font_family"])
                mapped_color = _lookup_style_color(original_legend_label, color_map)
                if mapped_color:
                    try:
                        legend_handle.set_color(mapped_color)
                    except Exception:
                        try:
                            legend_handle.set_facecolor(mapped_color)
                        except Exception:
                            pass
                if hasattr(legend_handle, "set_linewidth"):
                    legend_width_value = _lookup_style_float(original_legend_label, line_width_map)
                    legend_handle.set_linewidth(legend_width_value if legend_width_value is not None else line_width)
                if hasattr(legend_handle, "set_alpha"):
                    legend_handle.set_alpha(line_alpha)
                mapped_legend_label = _lookup_style_display_name(original_legend_label, display_name_map)
                if mapped_legend_label:
                    legend_text.set_text(mapped_legend_label)

        ax.grid(True, alpha=style["grid_alpha"])

    fig.tight_layout()
