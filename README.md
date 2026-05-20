# Ponon crystal_Subsystem test

副车架子系统声子晶体减振试验数据处理与出图程序。

本仓库用于批量读取不同工况下的 MATLAB `.mat` 试验数据，计算输入归一化 PSD 传递率、频段均值、相对 no-load 减振率，并输出对比曲线、柱状图和 Excel 数据表。

> 仓库只保存程序和说明，不上传原始 `.mat` 数据、PPT、Word 报告和已生成图片。

## 1. 目录结构

```text
.
├── main.py
├── requirements.txt
├── README.md
├── docs/
│   ├── DATA_STRUCTURE.md
│   └── PROCESSING_METHODS.md
├── yuanshishuju/
│   ├── 5.8/
│   ├── 5.10/
│   ├── 5.11/
│   └── 5.12/
└── results/
    ├── figures/
    └── xlsx/
```

GitHub 不保存空文件夹，因此仓库中通过 `.gitkeep` 保留目录结构。

## 2. 安装依赖

建议 Python 3.9 及以上版本。

```bash
pip install -r requirements.txt
```

## 3. 原始数据要求

每个 `.mat` 文件对应一个工况，文件中应包含：

```text
shdf.Absc1Data   # 时间数据
shdf.Data        # 通道数据，M x N
```

文件命名示例：

```text
no_load_1.mat
DVA_3.mat
plan1.mat
plan2_1.mat
plan3_1.mat
plan6_1.mat
plan7_1.mat
plan8_1.mat
plan9_1.mat
```

程序按文件名自动识别工况：

- `no_load` / `noload` -> `no-load`
- `DVA` -> `DVA`
- `plan1`、`plan2` ... -> 对应 plan 工况

## 4. 数据放置方式

不同日期批次的数据结构不同，建议按日期放入不同文件夹：

```text
yuanshishuju/
├── 5.8/
├── 5.10/
├── 5.11/
└── 5.12/
```

当前重点支持：

- 5.8：旧数据结构，四测点平均。
- 5.12：新数据结构，六测点平均。

详细通道行号见：[`docs/DATA_STRUCTURE.md`](docs/DATA_STRUCTURE.md)。

## 5. 运行方式

在项目根目录运行：

```bash
python main.py
```

默认会处理：

- `yuanshishuju/5.8/`
- `yuanshishuju/5.12/`

输出到：

```text
results/figures/
results/xlsx/
```

## 6. 主要输出

程序会按日期批次输出：

1. 多测点平均输入归一化 PSD 传递率曲线。
2. 频段均值 `L_B` 对比柱状图。
3. 相对 no-load 的减振率 `DR_B` 对比柱状图。
4. 对应 Excel 数据表。

默认统计频段：

- `80-100 Hz`
- `90-110 Hz`
- `70-140 Hz`

## 7. 核心计算公式

输入归一化 PSD 比值：

```text
R_j(f) = S_yy,j(f) / S_xx(f)
```

多测点线性域加权平均：

```text
R_avg(f) = sum(w_j * R_j(f)) / sum(w_j)
```

频段均值：

```text
L_B = -10 * log10( mean_{f_i in B}(R_avg(f_i)) )
```

相对 no-load 减振率：

```text
Delta L_B = L_B,scheme - L_B,no-load
DR_B = [1 - 10^(-Delta L_B / 10)] * 100%
```

解释：

- `L_B` 越大，表示该频段输入归一化 PSD 传递率越低。
- `DR_B > 0` 表示相对 no-load 降低。
- `DR_B < 0` 表示相对 no-load 放大。

## 8. 重要说明

1. 多测点平均必须在线性 PSD 比值域进行，不能先转 dB 再平均。
2. 当不同工况输入 PSD 差异明显时，优先使用输入归一化 PSD 传递率，不建议直接比较输出 PSD。
3. “总振级”和“输入归一化 PSD 传递率”是两个不同概念，报告和图题中应分开表述。
4. `.mat` 原始数据默认被 `.gitignore` 排除，避免误传试验数据。
