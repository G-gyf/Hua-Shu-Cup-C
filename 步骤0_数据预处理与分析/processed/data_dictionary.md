# 处理后数据字典

所有 CSV 使用 UTF-8 with BOM；原始附件不作修改。

## `dim_region.csv`

区域 GPU、IT/设施功率、PUE 和区域角色。

字段：

| field | dtype |
| --- | --- |
| Region | object |
| RegionRole | object |
| Total_GPU | int64 |
| Max_IT_Power_MW | int64 |
| PUE | float64 |
| Max_Facility_Power_MW | float64 |
| Reserved_GPU_Ratio | float64 |
| Available_GPU | int64 |
| Max_Workload_GPUh_per_h | int64 |
| CapacityLevel | object |
| Remarks | object |

## `dim_storage.csv`

区域储能参数、SOC 初值、充放电与购售电上限。

字段：

| field | dtype |
| --- | --- |
| Region | object |
| StorageCapacity_MWh | int64 |
| MinSOC_MWh | int64 |
| InitialSOC_MWh | float64 |
| MaxChargePower_MW | int64 |
| MaxDischargePower_MW | int64 |
| ChargeEfficiency | float64 |
| DischargeEfficiency | float64 |
| SellLimit_MW | int64 |
| Remarks | object |
| MaxGridImport_MW | int64 |
| MaxGridExport_MW | int64 |
| SOC_State_Convention | object |

## `dim_task_power.csv`

三类任务的等效 GPU IT 功率映射。

字段：

| field | dtype |
| --- | --- |
| TaskType | object |
| GPU_Power_MW_per_EquivalentGPU | float64 |
| Remarks | object |

## `dim_network_latency.csv`

源区域到目标区域的单向网络时延。

字段：

| field | dtype |
| --- | --- |
| FromRegion | object |
| ToRegion | object |
| NetworkLatency_ms | int64 |
| LatencyClass | object |

## `fact_tasks.csv`

原始任务字段及分钟时长换算的派生字段。

字段：

| field | dtype |
| --- | --- |
| TaskID | int64 |
| TaskType | object |
| ArrivalHour | int64 |
| GPU_Demand | int64 |
| EstimatedDuration_min | int64 |
| DelaySensitivity | object |
| SourceRegion | object |
| MaxLatency_ms | int64 |
| LatestFinishHour | int64 |
| EarliestStartHour | int64 |
| ExecutionMode | object |
| duration_h | float64 |
| duration_ceil_h | int64 |
| immediate_finish_hour | float64 |
| immediate_finish_boundary | int64 |
| gpu_hours | float64 |

## `task_hour_profile.csv`

任务在相对执行小时内的精确重叠、GPUh 与 AI IT 能量。

字段：

| field | dtype |
| --- | --- |
| TaskID | int64 |
| TaskType | object |
| GPU_Demand | int64 |
| relative_hour | int64 |
| overlap_h | float64 |
| gpu_hour | float64 |
| gpu_power_mw | float64 |
| ai_it_energy_mwh | float64 |

## `task_region_eligibility.csv`

每项任务到各区域的单向时延及阈值可行标记。

字段：

| field | dtype |
| --- | --- |
| TaskID | int64 |
| TaskType | object |
| SourceRegion | object |
| MaxLatency_ms | int64 |
| FromRegion | object |
| ToRegion | object |
| NetworkLatency_ms | int64 |
| LatencyClass | object |
| is_latency_feasible | bool |

## `forecast_panel.csv`

问题 1 的 18 条小时序列及无未来信息的时序特征。

字段：

| field | dtype |
| --- | --- |
| Hour | int64 |
| SourceRegion | object |
| TaskType | object |
| task_count | int64 |
| gpu_demand_sum | float64 |
| gpu_hours_sum | float64 |
| hour_of_day | int64 |
| day_index | int64 |
| day_of_week | int64 |
| is_weekend | int64 |
| task_count_lag_1 | float64 |
| task_count_lag_24 | float64 |
| task_count_lag_168 | float64 |
| task_count_roll_mean_24 | float64 |
| task_count_roll_mean_168 | float64 |
| gpu_demand_sum_lag_1 | float64 |
| gpu_demand_sum_lag_24 | float64 |
| gpu_demand_sum_lag_168 | float64 |
| gpu_demand_sum_roll_mean_24 | float64 |
| gpu_demand_sum_roll_mean_168 | float64 |
| gpu_hours_sum_lag_1 | float64 |
| gpu_hours_sum_lag_24 | float64 |
| gpu_hours_sum_lag_168 | float64 |
| gpu_hours_sum_roll_mean_24 | float64 |
| gpu_hours_sum_roll_mean_168 | float64 |
| split | object |

## `scenario_region_hour.csv`

问题 4 的参数化价格/可再生能源情景数据。

字段：

| field | dtype |
| --- | --- |
| Hour | int64 |
| Region | object |
| scenario_id | object |
| carbon_budget_ratio | float64 |
| price_spread_factor | float64 |
| renewable_scale_factor | float64 |
| ElectricityPrice_CNY_per_MWh_scenario | float64 |
| SellPrice_CNY_per_MWh_scenario | float64 |
| AvailableRenewable_MW_scenario | float64 |
