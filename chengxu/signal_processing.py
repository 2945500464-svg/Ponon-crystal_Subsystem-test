from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .config import exciter_accel_channel_index, force_channel_index

def trim_data_edges(
    data: np.ndarray,
    sample_rate: float,
    trim_start: float,
    trim_end: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    裁掉数据前后指定时长，仅保留中间有效时段。

    返回：
        trimmed_data: 裁剪后的通道数据
        info: 裁剪信息
    """
    if trim_start < 0 or trim_end < 0:
        raise ValueError("trim_start_sec 和 trim_end_sec 不能为负数")
    if sample_rate <= 0:
        raise ValueError("fs 必须大于 0")

    n_samples = data.shape[1]
    trim_start_samples = max(0, int(round(trim_start * sample_rate)))
    trim_end_samples = max(0, int(round(trim_end * sample_rate)))

    start_sample = min(trim_start_samples, n_samples)
    end_sample = max(start_sample, n_samples - trim_end_samples)

    if start_sample >= end_sample:
        raise ValueError(
            "前后裁剪时长过大，裁剪后没有剩余有效数据。"
            f"当前样本数={n_samples}，前裁样本={trim_start_samples}，后裁样本={trim_end_samples}"
        )

    trimmed_data = data[:, start_sample:end_sample]
    info = {
        "trim_start_samples": float(trim_start_samples),
        "trim_end_samples": float(trim_end_samples),
        "effective_samples": float(trimmed_data.shape[1]),
        "effective_duration_sec": float(trimmed_data.shape[1] / sample_rate),
    }
    return trimmed_data, info


def get_analysis_segments(
    data: np.ndarray,
    sample_rate: float,
    duration_sec: float,
    mode: str,
    start_sec: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    根据处理模式生成分析片段。

    返回：
        segments: 形状为 段数 x 通道数 x 每段样本数
        info: 分段信息
    """
    if duration_sec <= 0:
        raise ValueError("analysis_duration_sec 必须大于 0")
    if sample_rate <= 0:
        raise ValueError("fs 必须大于 0")

    n_channels, n_samples = data.shape
    segment_samples = max(1, int(round(duration_sec * sample_rate)))
    start_sample = max(0, int(round(start_sec * sample_rate)))
    available_samples = max(0, n_samples - start_sample)

    info: Dict[str, float] = {
        "segment_samples": float(segment_samples),
        "start_sample": float(start_sample),
        "available_samples": float(available_samples),
        "used_samples": 0.0,
        "n_segments": 0.0,
        "zero_padded_samples": 0.0,
    }

    if mode == "single_segment":
        segment = np.zeros((n_channels, segment_samples), dtype=float)
        end_sample = min(n_samples, start_sample + segment_samples)
        if start_sample < n_samples:
            chunk = data[:, start_sample:end_sample]
            segment[:, : chunk.shape[1]] = chunk
            info["used_samples"] = float(chunk.shape[1])
            info["zero_padded_samples"] = float(segment_samples - chunk.shape[1])
        else:
            info["zero_padded_samples"] = float(segment_samples)

        info["n_segments"] = 1.0
        return segment[np.newaxis, :, :], info

    if mode == "segment_average":
        if available_samples >= segment_samples:
            n_full_segments = available_samples // segment_samples
            used_samples = n_full_segments * segment_samples
            start = start_sample
            end = start + used_samples
            usable = data[:, start:end]
            segments = usable.reshape(n_channels, n_full_segments, segment_samples).transpose(1, 0, 2)

            info["used_samples"] = float(used_samples)
            info["n_segments"] = float(n_full_segments)
            info["discarded_tail_samples"] = float(available_samples - used_samples)
            return segments, info

        # 数据不足一段时，保留已有数据并零填充，避免样本少 1 点时处理失败。
        segment = np.zeros((n_channels, segment_samples), dtype=float)
        if start_sample < n_samples:
            chunk = data[:, start_sample:n_samples]
            segment[:, : chunk.shape[1]] = chunk
            info["used_samples"] = float(chunk.shape[1])
            info["zero_padded_samples"] = float(segment_samples - chunk.shape[1])
        else:
            info["zero_padded_samples"] = float(segment_samples)

        info["n_segments"] = 1.0
        return segment[np.newaxis, :, :], info

    raise ValueError('analysis_mode 只能是 "single_segment" 或 "segment_average"')


def choose_nfft(segment_samples: int, sample_rate: float, desired_df: float) -> int:
    """根据目标频率分辨率和段长确定 nfft。"""
    if desired_df <= 0:
        raise ValueError("target_df 必须大于 0")

    target_nfft = max(1, int(round(sample_rate / desired_df)))
    return max(segment_samples, target_nfft)


def remove_channel_mean(segment_data: np.ndarray) -> np.ndarray:
    """按通道去均值，保持与 MATLAB 程序 data = data - mean(data, 2) 一致。"""
    return segment_data - np.mean(segment_data, axis=1, keepdims=True)


def apply_hann_and_fft(
    segment_data: np.ndarray,
    sample_rate: float,
    desired_df: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单段多通道数据加 Hann 窗并计算单边 FFT。

    参数：
        segment_data: 形状为 通道数 x 样本数

    返回：
        freqs: 单边频率轴
        spectrum: 单边复数频谱，形状为 通道数 x 频点数
    """
    if segment_data.ndim != 2:
        raise ValueError("segment_data 必须是二维数组：通道数 x 样本数")

    n_samples = segment_data.shape[1]
    nfft = choose_nfft(n_samples, sample_rate, desired_df)
    window = np.hanning(n_samples)
    windowed = remove_channel_mean(segment_data) * window[np.newaxis, :]

    spectrum = np.fft.rfft(windowed, n=nfft, axis=1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    return freqs, spectrum


def compute_transfer_loss(
    segments: np.ndarray,
    sample_rate: float,
    desired_df: float,
    freq_min: float,
    freq_max: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    基于功率谱计算传递比：TL(f) = 10 * log10(S_yy(f) / S_xx(f))。

    segment_average 模式下，先对线性 PSD 比值求平均，再转换为 dB。
    """
    if segments.ndim != 3:
        raise ValueError("segments 必须是三维数组：段数 x 通道数 x 每段样本数")

    if segments.shape[1] < 2:
        raise ValueError("Data 至少需要 1 个输入通道和 1 个输出通道")

    ratio_list: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None

    tiny = np.finfo(float).tiny

    for segment in segments:
        segment_freqs, spectrum = apply_hann_and_fft(segment, sample_rate, desired_df)
        power_spectrum = np.abs(spectrum) ** 2

        input_psd = np.maximum(power_spectrum[0:1, :], tiny)
        output_psd = power_spectrum[1:, :]
        ratio = output_psd / input_psd
        ratio_list.append(ratio)

        if freqs is None:
            freqs = segment_freqs

    if freqs is None:
        raise ValueError("没有可用于分析的数据段")

    mean_ratio = np.mean(np.stack(ratio_list, axis=0), axis=0)
    transfer_loss = 10.0 * np.log10(np.maximum(mean_ratio, tiny))

    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    actual_df = float(freqs[1] - freqs[0]) if freqs.size > 1 else math.nan
    return freqs[freq_mask], transfer_loss[:, freq_mask], actual_df


def get_transfer_loss_output_indices(n_channels: int) -> List[int]:
    """生成 PSD 传递比输出通道索引，跳过参考输入、力传感器和激振器加速度通道。"""
    skipped_indices = {1, force_channel_index, exciter_accel_channel_index}
    return [idx for idx in range(1, n_channels) if idx not in skipped_indices]


def compute_accel_force_frf(
    segments: np.ndarray,
    sample_rate: float,
    desired_df: float,
    freq_min: float,
    freq_max: float,
    accel_index: int,
    force_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """基于 PSD 计算输入点响应比：10log10(S_aa/S_ff)。"""
    if segments.ndim != 3:
        raise ValueError("segments 必须是三维数组：段数 x 通道数 x 每段样本数")

    if segments.shape[1] <= max(accel_index, force_index):
        raise ValueError("通道数不足，无法计算输入点加速度/力 FRF")

    frf_list: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None
    tiny = np.finfo(float).tiny

    for segment in segments:
        segment_freqs, spectrum = apply_hann_and_fft(segment, sample_rate, desired_df)
        accel_psd = np.abs(spectrum[accel_index, :]) ** 2
        force_psd = np.maximum(np.abs(spectrum[force_index, :]) ** 2, tiny)
        frf_list.append(accel_psd / force_psd)

        if freqs is None:
            freqs = segment_freqs

    if freqs is None:
        raise ValueError("没有可用于计算 FRF 的数据段")

    mean_frf = np.mean(np.stack(frf_list, axis=0), axis=0)
    frf_db = 10.0 * np.log10(np.maximum(mean_frf, tiny))
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    return freqs[freq_mask], frf_db[freq_mask]


def compute_force_psd(
    segments: np.ndarray,
    sample_rate: float,
    desired_df: float,
    freq_min: float,
    freq_max: float,
    force_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算输入力通道的单边功率谱密度。"""
    if segments.ndim != 3:
        raise ValueError("segments 必须是三维数组：段数 x 通道数 x 每段样本数")

    if segments.shape[1] <= force_index:
        raise ValueError("通道数不足，无法计算输入力 PSD")

    psd_list: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None

    for segment in segments:
        force_signal = segment[force_index, :]
        force_signal = force_signal - np.mean(force_signal)
        n_samples = force_signal.size
        nfft = choose_nfft(n_samples, sample_rate, desired_df)
        window = np.hanning(n_samples)
        window_power = np.sum(window ** 2)
        spectrum = np.fft.rfft(force_signal * window, n=nfft)
        segment_freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        psd = (np.abs(spectrum) ** 2) / (sample_rate * window_power)
        if psd.size > 2:
            psd[1:-1] *= 2.0
        psd_list.append(psd)

        if freqs is None:
            freqs = segment_freqs

    if freqs is None:
        raise ValueError("没有可用于计算输入力 PSD 的数据段")

    mean_psd = np.mean(np.stack(psd_list, axis=0), axis=0)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    return freqs[freq_mask], mean_psd[freq_mask]


def compute_channel_psd(
    segments: np.ndarray,
    sample_rate: float,
    desired_df: float,
    freq_min: float,
    freq_max: float,
    channel_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """计算指定通道的平均单边 PSD。"""
    if segments.ndim != 3:
        raise ValueError("segments 必须是三维数组：段数 x 通道数 x 每段样本数")
    if not channel_indices:
        raise ValueError("channel_indices 不能为空")
    if segments.shape[1] <= max(channel_indices):
        raise ValueError("通道数不足，无法计算指定通道 PSD")

    psd_list: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None

    for segment in segments:
        n_samples = segment.shape[1]
        nfft = choose_nfft(n_samples, sample_rate, desired_df)
        window = np.hanning(n_samples)
        window_power = np.sum(window ** 2)
        selected = segment[channel_indices, :]
        selected = selected - np.mean(selected, axis=1, keepdims=True)
        spectrum = np.fft.rfft(selected * window[np.newaxis, :], n=nfft, axis=1)
        segment_freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        psd = (np.abs(spectrum) ** 2) / (sample_rate * window_power)
        if psd.shape[1] > 2:
            psd[:, 1:-1] *= 2.0
        psd_list.append(psd)

        if freqs is None:
            freqs = segment_freqs

    if freqs is None:
        raise ValueError("没有可用于计算 PSD 的数据段")

    mean_psd = np.mean(np.stack(psd_list, axis=0), axis=0)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    return freqs[freq_mask], mean_psd[:, freq_mask]


def compute_channel_fft_amplitude(
    segments: np.ndarray,
    sample_rate: float,
    desired_df: float,
    freq_min: float,
    freq_max: float,
    channel_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """计算指定加速度通道的单边 FFT 幅值谱。

    频率分辨率由 desired_df 控制；多段数据先对单边幅值谱在线性域平均。
    """
    if segments.ndim != 3:
        raise ValueError("segments 必须是三维数组：段数 x 通道数 x 每段样本数")
    if not channel_indices:
        raise ValueError("channel_indices 不能为空")
    if segments.shape[1] <= max(channel_indices):
        raise ValueError("通道数不足，无法计算指定通道 FFT")
    if sample_rate <= 0:
        raise ValueError("fs 必须大于 0")
    if desired_df <= 0:
        raise ValueError("FFT 频率分辨率必须大于 0")

    nfft = max(1, int(round(sample_rate / desired_df)))
    block_samples = nfft

    window = np.hanning(block_samples)
    coherent_gain = max(float(np.sum(window)), np.finfo(float).tiny)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    amplitude_list: List[np.ndarray] = []

    for segment in segments:
        n_samples = segment.shape[1]
        n_blocks = n_samples // block_samples
        if n_blocks == 0:
            block = np.zeros((segment.shape[0], block_samples), dtype=float)
            block[:, :n_samples] = segment
            block_list = [block]
        else:
            block_list = [
                segment[:, start : start + block_samples]
                for start in range(0, n_blocks * block_samples, block_samples)
            ]

        for block in block_list:
            selected = block[channel_indices, :]
            selected = selected - np.mean(selected, axis=1, keepdims=True)
            spectrum = np.fft.rfft(selected * window[np.newaxis, :], n=nfft, axis=1)
            amplitude = 2.0 * np.abs(spectrum) / coherent_gain
            if amplitude.shape[1] > 0:
                amplitude[:, 0] *= 0.5
            if nfft % 2 == 0 and amplitude.shape[1] > 1:
                amplitude[:, -1] *= 0.5
            amplitude_list.append(amplitude)

    mean_amplitude = np.mean(np.stack(amplitude_list, axis=0), axis=0)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    return freqs[freq_mask], mean_amplitude[:, freq_mask]


def compute_normalized_psd_ratio_average(
    segments: np.ndarray,
    sample_rate: float,
    desired_df: float,
    freq_min: float,
    freq_max: float,
    input_index: int,
    sensor_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """计算指定输出测点平均输入归一化 PSD 比值：mean(S_yy,i / S_xx)。"""
    valid_sensor_indices = [idx for idx in sensor_indices if idx < segments.shape[1]]
    if not valid_sensor_indices:
        raise ValueError("没有可用于归一化 PSD 比值计算的输出测点")

    freqs, psd = compute_channel_psd(
        segments=segments,
        sample_rate=sample_rate,
        desired_df=desired_df,
        freq_min=freq_min,
        freq_max=freq_max,
        channel_indices=[input_index] + valid_sensor_indices,
    )
    tiny = np.finfo(float).tiny
    input_psd = np.maximum(psd[0, :], tiny)
    ratio_avg = np.mean(psd[1:, :] / input_psd[np.newaxis, :], axis=0)
    return freqs, ratio_avg
