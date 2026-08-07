# C 题数据预处理与探索分析报告

## 数据范围

- 任务记录：50,000 条；区域—小时记录：14,442 条。
- 任务逐小时重叠剖面：195,047 条；任务—目标区域时延记录：300,000 条。
- 原始 Excel 仅作读取；基准字段与输入字段已分表保存。

## 数据质量

- PASS：25 项；FLAG：1 项。
- 质量校验仅标记附件数据与附件公式间的差异，不会修改源数据。

| check | status | failed_count | maximum_error | detail |
| --- | --- | --- | --- | --- |
| TaskID uniqueness | PASS | 0.000000 |  | TaskID must be unique. |
| Task missing values | PASS | 0.000000 |  | Task table has no missing cells. |
| Task arrival range | PASS | 0.000000 |  | ArrivalHour must be 0-2399. |
| Earliest start equals arrival | PASS | 0.000000 |  | Attachment task records use arrival as earliest start. |
| Task immediate completion boundary | PASS | 0.000000 |  | Immediate execution must finish no later than the 2406 boundary. |
| Region-hour key uniqueness | PASS | 0.000000 |  | (Region, Hour) must be unique. |
| Region-hour coverage | PASS | 0.000000 |  | Expected 14442 records for 6 regions and hours 0-2406. |
| Region-time missing values | PASS | 0.000000 |  | Region-time table has no missing cells. |
| Latency matrix completeness | PASS | 0.000000 |  | Expected a 6x6 directed matrix. |
| Latency key uniqueness | PASS | 0.000000 |  | Each directed pair must appear once. |
| Task overlap-duration conservation | PASS | 0.000000 | 0.000000 | Sum of relative-hour overlaps must equal task duration in hours. |
| Task latency feasibility | PASS | 0.000000 |  | Each task must have at least one destination satisfying its MaxLatency_ms. |
| Baseline IT split | PASS | 0.000000 | 0.000000 | Verified against the formula in Attachment 1. |
| Baseline PUE conversion | PASS | 0.000000 | 0.000050 | Verified against the formula in Attachment 1. |
| Baseline charge split | PASS | 0.000000 | 0.000100 | Verified against the formula in Attachment 1. |
| Baseline net import | PASS | 0.000000 | 0.000100 | Verified against the formula in Attachment 1. |
| Baseline carbon | PASS | 0.000000 | 0.000080 | Verified against the formula in Attachment 1. |
| Baseline renewable allocation | PASS | 0.000000 | 0.000200 | Verified against the formula in Attachment 1. |
| Baseline power balance | PASS | 0.000000 | 0.000200 | Verified against the formula in Attachment 1. |
| Baseline SOC minimum | PASS | 0.000000 |  | Checked against storage_information.xlsx. |
| Baseline SOC capacity | PASS | 0.000000 |  | Checked against storage_information.xlsx. |
| Baseline charge power limit | PASS | 0.000000 |  | Checked against storage_information.xlsx. |
| Baseline discharge power limit | PASS | 0.000000 |  | Checked against storage_information.xlsx. |
| Baseline grid import limit | PASS | 0.000000 |  | Checked against storage_information.xlsx. |
| Baseline grid export limit | PASS | 0.000000 |  | Checked against storage_information.xlsx. |
| Baseline SOC recurrence | FLAG | 1.000000 | 0.999944 | Verified against Attachment 1 SOC formula. |

## 任务类型汇总

| TaskType | tasks | gpu_demand_min | gpu_demand_mean | gpu_demand_max | duration_min_mean | duration_min_max | total_gpu_hours | task_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AITraining | 16,559.000 | 16.000 | 71.281 | 127.000 | 204.506 | 399.000 | 4,023,538.717 | 0.331 |
| BatchInference | 16,717.000 | 4.000 | 13.523 | 23.000 | 204.208 | 399.000 | 769,097.800 | 0.334 |
| RealTimeInference | 16,724.000 | 1.000 | 4.019 | 7.000 | 203.928 | 399.000 | 228,150.133 | 0.334 |

## 任务来源区域汇总

| SourceRegion | tasks | total_gpu_hours | gpu_demand_mean | duration_min_mean |
| --- | --- | --- | --- | --- |
| RegionA | 10,062.000 | 448,452.450 | 13.091 | 204.755 |
| RegionB | 9,200.000 | 430,758.150 | 13.784 | 203.419 |
| RegionC | 7,559.000 | 374,934.283 | 14.635 | 205.005 |
| RegionD | 9,560.000 | 1,548,858.433 | 47.591 | 203.992 |
| RegionE | 6,912.000 | 1,087,748.217 | 46.900 | 202.729 |
| RegionF | 6,707.000 | 1,130,035.117 | 48.503 | 205.440 |

## 时延阈值可行区域数

| TaskType | feasible_region_min | feasible_region_mean | feasible_region_max |
| --- | --- | --- | --- |
| AITraining | 6.000 | 6.000 | 6.000 |
| BatchInference | 5.000 | 5.679 | 6.000 |
| RealTimeInference | 1.000 | 2.873 | 3.000 |

## 区域逐时能源数据汇总

| Region | ElectricityPrice_CNY_per_MWh_min | ElectricityPrice_CNY_per_MWh_mean | ElectricityPrice_CNY_per_MWh_max | CarbonIntensity_tCO2_per_MWh_min | CarbonIntensity_tCO2_per_MWh_mean | CarbonIntensity_tCO2_per_MWh_max | AvailableRenewable_MW_min | AvailableRenewable_MW_mean | AvailableRenewable_MW_max | Baseline_AI_IT_Load_MW_min | Baseline_AI_IT_Load_MW_mean | Baseline_AI_IT_Load_MW_max | NonAI_IT_Load_MW_min | NonAI_IT_Load_MW_mean | NonAI_IT_Load_MW_max | NetGridImport_MW_min | NetGridImport_MW_mean | NetGridImport_MW_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RegionA | 444.600 | 707.454 | 1,095.840 | 0.552 | 0.620 | 0.688 | 500.000 | 799.804 | 1,100.000 | 0.000 | 23.311 | 90.436 | 155.675 | 314.160 | 401.020 | 0.000 | 388.065 | 497.002 |
| RegionB | 426.080 | 677.976 | 1,050.180 | 0.517 | 0.580 | 0.643 | 500.000 | 799.804 | 1,100.000 | 0.000 | 22.630 | 87.167 | 149.185 | 311.066 | 391.787 | 0.000 | 370.944 | 467.856 |
| RegionC | 413.730 | 658.325 | 1,019.740 | 0.490 | 0.550 | 0.610 | 500.000 | 799.804 | 1,100.000 | 0.000 | 20.033 | 90.759 | 150.666 | 312.144 | 387.628 | 0.000 | 364.246 | 455.266 |
| RegionD | 265.530 | 422.507 | 654.460 | 0.374 | 0.420 | 0.466 | 500.000 | 799.804 | 1,100.000 | 0.000 | 98.871 | 232.068 | 37.524 | 254.181 | 459.020 | -180.000 | 79.981 | 282.582 |
| RegionE | 234.650 | 373.378 | 578.360 | 0.196 | 0.220 | 0.244 | 500.000 | 799.804 | 1,100.000 | 0.000 | 69.508 | 215.984 | 4.566 | 283.566 | 473.415 | -220.000 | -42.611 | 166.060 |
| RegionF | 247.000 | 393.030 | 608.800 | 0.232 | 0.260 | 0.288 | 500.000 | 799.804 | 1,100.000 | 0.000 | 72.639 | 206.915 | 68.870 | 280.700 | 473.417 | -220.000 | -14.339 | 152.497 |

## 预测划分汇总

| split | records | task_count | gpu_demand_sum | gpu_hours_sum |
| --- | --- | --- | --- | --- |
| test | 432.000 | 538.000 | 17,449.000 | 59,986.767 |
| train | 42,336.000 | 48,963.000 | 1,442,031.000 | 4,913,871.783 |
| validation | 432.000 | 499.000 | 14,134.000 | 46,928.100 |

## 图表

- `figures/task_counts_by_type.png`
- `figures/hourly_task_arrivals.png`
- `figures/gpu_hours_by_source_region.png`
- `figures/price_carbon_scatter.png`

## 口径说明

- 问题 1 的预测面板按照题目给定的训练、验证和测试窗口标记。
- `task_hour_profile.csv` 采用整数小时开工、分钟级时长与实际重叠量；执行实际时点为开工时段加相对小时。
- 问题 2/4 应以任务调度重新形成 AI IT 负荷，并叠加 `NonAI_IT_Load_MW`；问题 3 使用附件给定的 `Baseline_AI_IT_Load_MW` 与 `NonAI_IT_Load_MW`。
- `scenario_region_hour.csv` 为问题 4 参数化情景，不属于附件原始数据。