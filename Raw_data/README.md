# Raw_data

Put local test data here. Raw `.mat` files and `数据格式.xlsx` are ignored by Git by default.

Recommended structure:

```text
Raw_data/
├─ 数据格式.xlsx
├─ 5.8/
│  ├─ no-load.mat
│  └─ plan1.mat
├─ 5.12/
└─ 5.31/
```

The program reads date folders from this directory. Keep experimental data local unless a separate data-sharing rule is agreed.
