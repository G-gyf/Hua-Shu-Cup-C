# 数据预处理与分析管道

运行命令：

```powershell
python analysis_pipeline/run_pipeline.py
```

管道只读取 `附件数据/` 中的 Excel 文件，不会修改原始附件。每次运行会重建：

- `processed/`：标准化维表、事实表、任务逐小时重叠剖面、时延可行集、预测面板、情景数据、清单和质量报告；
- `reports/`：数据分析报告、统计表和图表。

任务开工时间按整数小时处理；任务分钟级时长保留，并由 `task_hour_profile.csv` 给出相对小时的实际重叠量。问题四情景参数在 `scenario_config.json` 中维护。

代码运行环境：Python 3.10+，`pandas`、`openpyxl`、`matplotlib`。若未安装 `matplotlib`，其余数据和报告仍可生成，只会跳过 PNG 图表。
