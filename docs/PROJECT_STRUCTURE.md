# 程序结构与维护说明

项目已经从单一 `main.py` 拆分为 `chengxu` 包，便于人工检查和后续维护。

## 1. 入口

`main.py` 只负责启动 GUI，并保留对常用函数的兼容导出。

```text
python main.py
```

## 2. 模块边界

| 模块 | 边界 |
|---|---|
| `config.py` | 路径、默认参数、通道结构常量 |
| `utils.py` | 文件名清理、工况排序、名称规范化、颜色规则 |
| `data_format.py` | 日期数据结构解析，不做计算和绘图 |
| `mat_io.py` | `.mat` 文件读取，不参与图形样式 |
| `signal_processing.py` | FFT、PSD、传递率、FRF 等纯计算 |
| `analysis.py` | 批量组织分析结果 |
| `plotting.py` | 结构振动类图形 |
| `microphone_processing.py` | 麦克风声学计算和图形 |
| `plot_style.py` | 图形样式统一应用 |
| `exporters.py` | 保存图片和表格 |
| `gui.py` | 界面状态、用户选择、任务调度 |

## 3. 维护原则

- 数据读取模块不依赖 GUI。
- 计算模块不直接保存图片。
- 绘图模块接收分析结果，不直接读取 `.mat`。
- GUI 负责收集用户输入和调用分析/导出函数，不写复杂公式。
- 新增数据结构优先在 `config.py` 和 `data_format.py` 中维护。
- 新增图形优先在 `plotting.py` 或 `microphone_processing.py` 中实现，再由 `exporters.py` 和 GUI 调用。

## 4. 本地文件管理

`Raw_data` 和 `output` 是运行时目录：

- `Raw_data` 放原始数据；
- `output` 放生成结果。

两者中的实际数据和结果默认不进入 Git。
