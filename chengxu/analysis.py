from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import RAW_DATA_DIR, fs
from .data_format import get_gui_channel_config, get_heatmap_sensor_indices, resolve_gui_input_index
from .mat_io import load_mat_file
from .signal_processing import compute_accel_force_frf, compute_channel_fft_amplitude_1hz, compute_channel_psd, compute_force_psd, compute_normalized_psd_ratio_average, compute_transfer_loss, get_analysis_segments, trim_data_edges

def analyze_selected_files(
    mat_files: List[Path],
    input_mode: str,
    freq_min: float,
    freq_max: float,
    desired_df: float,
    duration_sec: float,
    mode: str,
    start_sec: float,
    trim_start: float,
    trim_end: float,
    prefix_date: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load selected mat files and compute transfer/PSD data for plotting."""
    all_results: Dict[str, Dict[str, Any]] = {}
    layout_info: Dict[str, Any] = {"four_names": [], "six_names": [], "auto_stat_names": [], "auto_stat_mode": "four"}

    for mat_file in mat_files:
        print("-" * 72)
        try:
            display_path = mat_file.relative_to(RAW_DATA_DIR)
        except ValueError:
            display_path = mat_file
        print(f"正在处理: {display_path}")
        try:
            _, data = load_mat_file(mat_file)
            n_channels, n_samples = data.shape
            layout = get_gui_channel_config(mat_file, n_channels)
            names = layout["names"]
            input_index = resolve_gui_input_index(mat_file, input_mode)
            if input_index >= n_channels:
                print(f"  warning: 输入通道超出数据行数，跳过: {mat_file.name}")
                continue

            output_indices = [idx for idx in layout["output_indices"] if idx < n_channels and idx != input_index]
            if not output_indices:
                print(f"  warning: 没有可用输出通道，跳过: {mat_file.name}")
                continue

            trimmed_data, _ = trim_data_edges(
                data=data,
                sample_rate=fs,
                trim_start=trim_start,
                trim_end=trim_end,
            )
            segments, segment_info = get_analysis_segments(
                data=trimmed_data,
                sample_rate=fs,
                duration_sec=duration_sec,
                mode=mode,
                start_sec=start_sec,
            )

            freqs, transfer_loss_db, actual_df = compute_transfer_loss(
                segments=segments[:, [input_index] + output_indices, :],
                sample_rate=fs,
                desired_df=desired_df,
                freq_min=freq_min,
                freq_max=freq_max,
            )

            condition_name = f"{mat_file.parent.name} {mat_file.stem}" if prefix_date else mat_file.stem
            output_names = [names[idx] if idx < len(names) else f"Ch_{idx + 1}" for idx in output_indices]
            result: Dict[str, Any] = {
                "freqs": freqs,
                "transfer_loss_db": transfer_loss_db,
                "output_names": output_names,
                "channel_names": names,
                "source_file": mat_file.name,
                "input_index": input_index,
            }

            try:
                fft_freqs, fft_amplitude = compute_channel_fft_amplitude_1hz(
                    data=trimmed_data,
                    sample_rate=fs,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    channel_indices=output_indices,
                )
                result["accel_fft_freqs"] = fft_freqs
                result["accel_fft_amplitude"] = fft_amplitude
                result["accel_fft_names"] = output_names
            except Exception as exc:
                print(f"  warning: 加速度FFT计算失败: {exc}")

            try:
                input_psd_freqs, input_psd_data = compute_channel_psd(
                    segments=segments,
                    sample_rate=fs,
                    desired_df=desired_df,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    channel_indices=[input_index],
                )
                result["input_accel_psd_freqs"] = input_psd_freqs
                result["input_accel_psd"] = input_psd_data[0, :]
            except Exception as exc:
                print(f"  warning: 输入PSD计算失败: {exc}")

            force_index = layout.get("force_index")
            if force_index is not None and force_index < n_channels:
                try:
                    force_psd_freqs, force_psd = compute_force_psd(
                        segments=segments,
                        sample_rate=fs,
                        desired_df=desired_df,
                        freq_min=freq_min,
                        freq_max=freq_max,
                        force_index=force_index,
                    )
                    result["force_psd_freqs"] = force_psd_freqs
                    result["force_psd"] = force_psd
                except Exception as exc:
                    print(f"  warning: 输入力PSD计算失败: {exc}")

                try:
                    frf_freqs, frf_db = compute_accel_force_frf(
                        segments=segments,
                        sample_rate=fs,
                        desired_df=desired_df,
                        freq_min=freq_min,
                        freq_max=freq_max,
                        accel_index=input_index,
                        force_index=force_index,
                    )
                    result["frf_freqs"] = frf_freqs
                    result["accel_force_frf_db"] = frf_db
                except Exception as exc:
                    print(f"  warning: FRF计算失败: {exc}")

            normalized_indices = get_heatmap_sensor_indices(layout)
            auto_stat_names = [names[idx] for idx in normalized_indices if idx < len(names)]
            auto_stat_mode = "six" if ("左中" in auto_stat_names and "右中" in auto_stat_names) else "four"
            try:
                norm_freqs, norm_ratio = compute_normalized_psd_ratio_average(
                    segments=segments,
                    sample_rate=fs,
                    desired_df=desired_df,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    input_index=input_index,
                    sensor_indices=normalized_indices,
                )
                result["normalized_psd_freqs"] = norm_freqs
                result["normalized_psd_ratio_avg"] = norm_ratio
            except Exception as exc:
                print(f"  warning: 归一化PSD比值计算失败: {exc}")

            all_results[condition_name] = result
            layout_info["four_names"] = [names[idx] for idx in layout["four_indices"] if idx < len(names)]
            layout_info["six_names"] = [names[idx] for idx in layout["six_indices"] if idx < len(names)]
            layout_info["auto_stat_names"] = auto_stat_names
            layout_info["auto_stat_mode"] = auto_stat_mode

            print(
                f"  完成: {n_channels}通道 x {n_samples}样本, "
                f"输入=Data第{input_index + 1}行, 分析段={int(segment_info['n_segments'])}, df={actual_df:g} Hz"
            )
        except Exception as exc:
            print(f"  warning: {mat_file.name} 处理失败，已跳过。原因: {exc}")

    return all_results, layout_info
