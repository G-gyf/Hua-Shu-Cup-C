# 步骤 0：数据预处理与分析

本目录将官方附件 Excel 转换为后续建模可直接读取的标准化数据表，并生成数据质量检查、统计汇总和探索性图表。它是问题一至问题四的共同数据入口。

赛题本身的事实、题目要求和约束以根目录 [C题_赛题与数据记忆.md](../C题_赛题与数据记忆.md) 为准；本目录仅记录已实施的数据处理口径和由数据直接得到的分析结果。

## 输入、脚本与输出

| 项目 | 位置 | 说明 |
| --- | --- | --- |
| 官方输入 | `../附件数据/*.xlsx` | GPU、网络时延、电力映射、区域时序、存储和工作负载轨迹 |
| 主脚本 | `run_pipeline.py` | 读取官方数据，完成规范化、校验、特征/情景表构造与报告输出 |
| 情景配置 | `scenario_config.json` | 问题四所需情景参数；运行时会展开为有效配置表 |
| 处理数据 | `processed/` | 供预测、调度和评价模型使用的维度表、事实表及派生面板 |
| 分析报告 | `reports/` | 数据质量检查、汇总表、Markdown 报告和 PNG 图表 |
| 实施记忆 | `步骤0_数据预处理与分析_记忆.md` | 已执行步骤、口径、质量结论与注意事项 |

## 运行方法

请从本目录运行：

```powershell
python -m pip install pandas openpyxl matplotlib
python run_pipeline.py
```

脚本会先清理并重建 `processed/` 和 `reports/` 中的生成内容。它不会修改 `../附件数据/` 的任何官方原始文件。若未安装 `matplotlib`，数据表与文字报告仍可生成，但 PNG 图表会被跳过。

运行完成后，控制台会输出生成文件位置；建议检查 `processed/data_quality_checks.csv` 和 `reports/data_analysis_report.md`。

## 处理口径

### 时间与任务重叠

- 任务开工时间按整点小时归属，用于建立调度时刻和时序索引。
- 任务时长仍保留原始分钟信息；`task_hour_profile.csv` 按任务与小时的实际重叠时长给出 GPU 占用，不把跨小时任务简单截断。
- `fact_tasks.csv` 保留逐任务属性，是问题一预测和后续调度的基础；`forecast_panel.csv` 提供逐小时预测面板。

### 原始字段与单位

- 原始字段名、单位和表间关联方式均以官方附件为准，未用主观规则改写数值。
- 规范化输出将区域、任务类型、时刻及资源属性拆分为维度表与事实表，具体字段字典见 `processed/data_dictionary.md`。
- 问题四的碳预算、电价扩散和可再生能源缩放仅以配置表显式管理；有效组合输出到 `processed/scenario_config_effective.csv`。

### 数据质量

质量检查结果保存在 `processed/data_quality_checks.csv` 与 `reports/tables/data_quality_checks.csv`。当前检查覆盖源表行数、主键/时间完整性、维表关联、任务小时展开和基础电力/SOC 关系。已知边界和不应擅自修正的记录详见本目录记忆文件。

## 关键输出速查

### 建模主表

| 文件 | 粒度 | 用途 |
| --- | --- | --- |
| `fact_tasks.csv` | 单个任务 | 任务到达、GPU 需求、时长、SLA 等原始任务属性；问题一直接输入 |
| `task_hour_profile.csv` | 任务 × 相对小时 | 处理跨小时重叠后的 GPU 占用，用于资源调度 |
| `forecast_panel.csv` | 来源区域 × 任务类型 × 小时 | 预测任务 GPU 需求的面板数据 |
| `fact_region_hour_input.csv` | 区域 × 小时 | 区域级价格、碳强度、可再生出力等时序输入 |
| `fact_region_hour_baseline.csv` | 区域 × 小时 | 基线电力与储能相关数据 |
| `scenario_region_hour.csv` | 情景 × 区域 × 小时 | 问题四的区域时序情景数据 |

### 维度与辅助表

| 文件 | 内容 |
| --- | --- |
| `dim_region.csv` | 区域基础信息 |
| `dim_network_latency.csv` | 区域间网络时延 |
| `dim_task_power.csv` | 任务类型与功率映射 |
| `dim_storage.csv` | 储能设备信息 |
| `task_region_eligibility.csv` | 按时延要求得到的任务可调度区域集合 |
| `manifest.json` | 本次输出文件与来源信息 |

## 继续开发时的注意事项

- 不要用预测窗口中的真实任务记录构造问题一的预测特征或进行调参。
- 调度模型应区分“任务在来源区域到达”和“任务被分配到执行区域”，不得把二者混为同一字段。
- 每次新增派生变量、修正清洗逻辑或修改情景参数时，应同步更新脚本、记忆文件和数据字典，并重新运行管道。
