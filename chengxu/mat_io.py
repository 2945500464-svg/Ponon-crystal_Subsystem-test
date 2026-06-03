from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import h5py
import numpy as np
import scipy.io

def _unwrap_singleton(value: Any) -> Any:
    """展开 MATLAB 结构中常见的单元素 object 数组。"""
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _extract_mat_field(obj: Any, field_name: str) -> Any:
    """从 scipy.loadmat 读出的 MATLAB 结构中提取字段。"""
    obj = _unwrap_singleton(obj)

    if hasattr(obj, field_name):
        return getattr(obj, field_name)

    if isinstance(obj, np.ndarray):
        if obj.dtype.names and field_name in obj.dtype.names:
            return obj[field_name]
        if obj.size == 1:
            return _extract_mat_field(obj.item(), field_name)

    if isinstance(obj, np.void) and obj.dtype.names and field_name in obj.dtype.names:
        return obj[field_name]

    raise KeyError(f"缺少字段：{field_name}")


def _read_h5_node(h5_file: h5py.File, node: Any) -> Any:
    """读取 MATLAB v7.3 HDF5 节点，兼容常见引用形式。"""
    if h5py is None:
        raise ImportError("读取 MATLAB v7.3 文件需要安装 h5py：pip install h5py")

    if isinstance(node, h5py.Dataset):
        data = node[()]
        dtype = getattr(data, "dtype", None)
        if dtype is not None and h5py.check_dtype(ref=dtype) is not None:
            refs = np.asarray(data).ravel()
            refs = [ref for ref in refs if ref]
            if not refs:
                raise ValueError("HDF5 引用为空")
            if len(refs) == 1:
                return _read_h5_node(h5_file, h5_file[refs[0]])
            return np.array([_read_h5_node(h5_file, h5_file[ref]) for ref in refs], dtype=object)
        return data

    if isinstance(node, h5py.Group):
        raise ValueError(f"HDF5 节点 {node.name} 是 Group，无法直接转为数组")

    raise TypeError(f"不支持的 HDF5 节点类型：{type(node)}")


def _load_mat_with_scipy(mat_path: Path) -> Tuple[Any, Any]:
    mat_data = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    if "shdf" not in mat_data:
        raise KeyError("缺少结构体 shdf")

    shdf = mat_data["shdf"]
    time_data = _extract_mat_field(shdf, "Absc1Data")
    data = _extract_mat_field(shdf, "Data")
    return time_data, data


def _load_mat_with_h5py(mat_path: Path) -> Tuple[Any, Any]:
    if h5py is None:
        raise ImportError("读取 MATLAB v7.3 文件需要安装 h5py：pip install h5py")

    with h5py.File(mat_path, "r") as h5_file:
        if "shdf" not in h5_file:
            raise KeyError("缺少结构体 shdf")

        shdf = h5_file["shdf"]
        if not isinstance(shdf, h5py.Group):
            raise ValueError("v7.3 MAT 文件中的 shdf 不是可识别的 Group")

        if "Absc1Data" not in shdf or "Data" not in shdf:
            raise KeyError("shdf 中缺少 Absc1Data 或 Data")

        time_data = _read_h5_node(h5_file, shdf["Absc1Data"])
        data = _read_h5_node(h5_file, shdf["Data"])
        return time_data, data


def _to_float_array(value: Any, name: str, mat_path: Path) -> np.ndarray:
    """将读取出的数据转换为 float 数组。"""
    value = _unwrap_singleton(value)
    array = np.asarray(value)

    # MATLAB v7.3 复杂数可能以 real/imag 复合类型保存。
    if array.dtype.names and {"real", "imag"}.issubset(set(array.dtype.names)):
        array = array["real"] + 1j * array["imag"]

    array = np.squeeze(array)

    if array.dtype == object:
        array = _unwrap_singleton(array)
        array = np.asarray(array)
        array = np.squeeze(array)

    if np.iscomplexobj(array):
        imag_max = float(np.max(np.abs(np.imag(array)))) if array.size else 0.0
        if imag_max > 1e-10:
            print(f"  warning: {mat_path.name} 的 {name} 含复数，已取实部处理")
        array = np.real(array)

    try:
        return array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 无法转换为数值数组") from exc


def _normalize_loaded_arrays(time_raw: Any, data_raw: Any, mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """整理 Absc1Data 和 Data 的维度。"""
    time_data = _to_float_array(time_raw, "Absc1Data", mat_path).reshape(-1)
    data = _to_float_array(data_raw, "Data", mat_path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.ndim != 2:
        raise ValueError(f"Data 维度异常：{data.shape}")

    # 若 v7.3 文件被 h5py 读成 N x M，且时间长度匹配第 1 维，则转置为 M x N。
    if time_data.size > 0 and data.shape[1] != time_data.size and data.shape[0] == time_data.size:
        data = data.T
        print("  warning: 检测到 Data 维度与时间轴相反，已自动转置为 通道 x 样本")

    if data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError(f"Data 为空或维度异常：{data.shape}")

    if time_data.size > 0 and data.shape[1] != time_data.size:
        print(
            f"  warning: 时间点数({time_data.size})与 Data 样本数({data.shape[1]})不一致，"
            "后续按 Data 实际样本数处理"
        )

    return time_data, data


def load_mat_file(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取单个 .mat 文件。

    返回：
        time_data: 时间数据，一维数组
        data: 通道数据，形状为 通道数 x 样本数
    """
    try:
        time_raw, data_raw = _load_mat_with_scipy(mat_path)
    except NotImplementedError:
        time_raw, data_raw = _load_mat_with_h5py(mat_path)
    except ValueError as scipy_error:
        # scipy 对 v7.3 文件常抛 ValueError，这里尝试 h5py 兼容读取。
        try:
            time_raw, data_raw = _load_mat_with_h5py(mat_path)
        except Exception:
            raise scipy_error

    return _normalize_loaded_arrays(time_raw, data_raw, mat_path)
