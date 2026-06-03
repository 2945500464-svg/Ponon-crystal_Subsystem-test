# Ponon crystal_Subsystem test

副车架子系统声子晶体减振试验数据处理与出图程序。项目当前版本已经从早期脚本整理为模块化 Python 程序，并提供图形化交互界面，用于读取不同日期、不同结构的 MATLAB `.mat` 试验数据，完成输入归一化 PSD 传递率、平均传递率、频段统计、减振率、热力图、FRF 以及麦克风声学结果出图。

> 仓库只保存程序、主题、图标和说明文档；原始 `.mat` 数据、`数据格式.xlsx`、生成图片、报告和 PPT 不纳入 Git 管理。

## 1. 当前能力

- 图形化选择 `Raw_data` 下的日期文件夹和 `.mat` 数据文件。
- 自动读取 `Raw_data/数据格式.xlsx`，识别不同日期的输入基准、传感器数量和通道结构。
- 支持副车架振动数据处理：
  - 单输出通道 PSD 传递率曲线；
  - 四点等权平均、六点等权平均 PSD 传递率曲线；
  - 输入点 PSD、输入力 PSD；
  - 力-加速度 FRF；
  - 输入归一化 PSD 比值热力图；
  - 频段总振级图；
  - 相对 no-load 的平均减振率图。
- 支持 `5.22 + 5.23` 同名工况平均出图，用于左右激振器位置重复试验的综合比较。
- 支持 `5.31` 麦克风声学数据处理：
  - 70-140 Hz FFT 声压级图；
  - 70-140 Hz 总声压级图；
  - 1/3、1/6、1/10 倍频程声压级图；
  - 同名前缀数据平均后出图。
- 支持 GUI 内调整图形样式：
  - 字体、字号、图片尺寸；
  - 曲线颜色、线宽、亮度；
  - 图中显示名称；
  - 图例顺序；
  - 保存格式、DPI、保存目录。

## 2. 项目结构

```text
.
├─ main.py                         程序入口
├─ requirements.txt                Python 依赖
├─ launch_AITO_Data_processing.vbs  Windows 无控制台启动脚本
├─ launch_debug.bat                调试启动脚本
├─ assets/                         程序图标
├─ chengxu/                        核心程序模块
├─ themes/                         CustomTkinter 界面主题
├─ docs/                           数据结构、算法和界面说明
├─ Raw_data/                       本地原始数据目录，Git 默认忽略数据文件
└─ output/                         默认输出目录，Git 默认忽略生成文件
```

核心模块职责：

| 模块 | 作用 |
|---|---|
| `chengxu/config.py` | 全局参数、路径、默认通道结构 |
| `chengxu/data_format.py` | 读取 `数据格式.xlsx`，解析日期数据结构 |
| `chengxu/mat_io.py` | 读取 MATLAB `.mat` / v7.3 `.mat` 数据 |
| `chengxu/signal_processing.py` | 分段、Hann 窗、FFT、PSD、PSD 传递率、FRF |
| `chengxu/analysis.py` | 批量分析所选振动试验数据 |
| `chengxu/plotting.py` | 副车架振动类图表 |
| `chengxu/microphone_processing.py` | 麦克风声压级分析与出图 |
| `chengxu/plot_style.py` | 图形样式、颜色、图例名称和顺序 |
| `chengxu/exporters.py` | 图片和 Excel 导出 |
| `chengxu/gui.py` | CustomTkinter 交互界面 |

## 3. 安装与运行

建议使用 Python 3.9 及以上版本。

```bash
pip install -r requirements.txt
python main.py
```

Windows 下也可以双击：

```text
launch_AITO_Data_processing.vbs
```

如果程序窗口无法打开，使用：

```text
launch_debug.bat
```

该脚本会保留终端窗口，便于查看缺失依赖或数据读取错误。

## 4. 数据放置规则

所有试验数据放在 `Raw_data` 下。程序会扫描该目录中的日期文件夹。

```text
Raw_data/
├─ 数据格式.xlsx
├─ 5.8/
│  ├─ no-load.mat
│  ├─ DVA.mat
│  └─ plan1.mat
├─ 5.12/
└─ 5.31/
```

`.mat` 文件要求：

```text
shdf
├─ Absc1Data   时间数据，通常为 1 × N
└─ Data        通道数据，通常为 M × N
```

程序按行号解释通道，不依赖 `.mat` 文件内部单位元数据自动判断输入输出。

## 5. 工作流说明

### 5.1 副车架振动数据处理

用于处理加速度、力传感器等结构振动数据。该流程根据 `数据格式.xlsx` 和内置兜底规则决定输入基准、输出测点、四点/六点统计测点。

常用图：

- 四点或六点 PSD 传递率曲线；
- 归一化热力图；
- 总振级柱状图；
- 平均减振率柱状图。

### 5.2 5.22 + 5.23 同名平均

用于把 `5.22` 和 `5.23` 中同名数据进行线性域平均后出图。该流程只允许同名工况参与，避免混入不对应的数据。

### 5.3 麦克风声学数据处理

用于处理 `Raw_data/5.31` 下三通道麦克风数据：

```text
Data 第 1 行：主驾驶麦克风，单位 Pa
Data 第 2 行：中排麦克风，单位 Pa
Data 第 3 行：后排麦克风，单位 Pa
```

输出声学结果时使用参考声压 `p0 = 20 μPa`，结果单位为 `dB SPL`。

## 6. 输出规则

未勾选“保存本次生成图像”时，程序只弹出显示图片。

勾选后，默认保存到：

```text
output/figures/
```

Excel 表格默认保存到：

```text
output/xlsx/
```

支持 `png`、`pdf`、`svg`、`jpg` 等格式，DPI 可在界面中设置。

## 7. 关键口径

本项目优先使用输入归一化 PSD 传递率评价结构传递能力，而不是直接比较输出 PSD。原因是不同工况下激振输入能量可能存在差异，直接比较输出 PSD 容易把输入差异误判为方案差异。

多测点平均必须先在线性域平均，再转换为 dB。不要先把每个测点转为 dB 后直接算术平均。

详细公式见 [docs/PROCESSING_METHODS.md](docs/PROCESSING_METHODS.md)。

## 8. Git 管理原则

- 上传程序、主题、图标和说明文档。
- 不上传 `.mat` 原始数据。
- 不上传本地 `数据格式.xlsx`。
- 不上传生成图、Excel、Word 报告、PPT。
- 若确实需要共享大文件，应单独约定数据脱敏和存储方式，不直接提交到本仓库。
