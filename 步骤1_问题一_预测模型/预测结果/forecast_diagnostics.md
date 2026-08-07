# 问题一短期 GPU 需求预测诊断

## 预测口径

- 目标：来源区域 × 任务类型 × 小时的总 GPU 需求。
- 预测协议：每个验证/测试窗口均在预测原点一次性预测未来 24 小时；窗口内真实任务不参与任何特征或模型更新。
- 官方验证集选择的点预测模型：`direct_mean`。
- 区间：从预测原点之前的完整历史小时任务包进行 10,000 次 Bootstrap；每个任务包保留任务数、GPU 需求、时长和 SLA 属性的原始联合关系。
- 区域×类型经验校准区间：80% 为 P17–P83；95% 为 P5–P95。
- 区域/系统汇总经验校准区间：80% 为 P9.5–P90.5；95% 为 P3–P97。校准依据：48 rolling 24-hour origins within the training period; target empirical coverage 80% and 95%.

## 官方验证集候选模型比较

| Model | OriginHour | n | ActualSum | ForecastSum | MAE | RMSE | WAPE | Bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_mean | 2,351.0000 | 432.0000 | 14,134.0000 | 14,714.6020 | 23.8110 | 45.8519 | 0.7278 | 1.3440 |
| rolling_336 | 2,351.0000 | 432.0000 | 14,134.0000 | 14,788.1429 | 23.8713 | 45.9534 | 0.7296 | 1.5142 |
| rolling_168 | 2,351.0000 | 432.0000 | 14,134.0000 | 15,003.7143 | 24.0178 | 45.8436 | 0.7341 | 2.0132 |
| rolling_24 | 2,351.0000 | 432.0000 | 14,134.0000 | 15,177.0000 | 25.0123 | 46.2093 | 0.7645 | 2.4144 |
| seasonal_weekly | 2,351.0000 | 432.0000 | 14,134.0000 | 16,697.0000 | 29.6366 | 60.5823 | 0.9058 | 5.9329 |
| seasonal_daily | 2,351.0000 | 432.0000 | 14,134.0000 | 15,177.0000 | 35.8495 | 69.9249 | 1.0957 | 2.4144 |

## 训练期滚动 24 小时回测

| Model | Origins | MAE_mean | RMSE_mean | WAPE_mean | WAPE_std | Bias_mean |
| --- | --- | --- | --- | --- | --- | --- |
| direct_mean | 48.0000 | 26.2054 | 50.5688 | 0.7634 | 0.0435 | -0.4839 |
| rolling_336 | 48.0000 | 26.3195 | 50.6503 | 0.7668 | 0.0447 | -0.1708 |
| rolling_168 | 48.0000 | 26.3769 | 50.7758 | 0.7685 | 0.0451 | -0.0965 |
| rolling_24 | 48.0000 | 26.7292 | 51.5514 | 0.7785 | 0.0484 | 0.0105 |
| seasonal_weekly | 48.0000 | 33.8056 | 70.0692 | 0.9855 | 0.0708 | -0.2474 |
| seasonal_daily | 48.0000 | 34.5505 | 71.4979 | 1.0071 | 0.0703 | 0.0105 |

## 选定模型的误差

| Stage | Scope | SourceRegion | TaskType | n | ActualSum | ForecastSum | MAE | RMSE | WAPE | Bias | AggregateName |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | RegionTaskType | RegionA | AITraining | 24.0000 | 461.0000 | 632.9592 | 33.2594 | 39.2853 | 1.7315 | 7.1650 |  |
| validation | RegionTaskType | RegionA | BatchInference | 24.0000 | 381.0000 | 450.7347 | 11.8852 | 15.4586 | 0.7487 | 2.9056 |  |
| validation | RegionTaskType | RegionA | RealTimeInference | 24.0000 | 180.0000 | 236.7041 | 5.1209 | 5.9930 | 0.6828 | 2.3627 |  |
| validation | RegionTaskType | RegionB | AITraining | 24.0000 | 864.0000 | 622.0918 | 42.6402 | 60.5077 | 1.1844 | -10.0795 |  |
| validation | RegionTaskType | RegionB | BatchInference | 24.0000 | 380.0000 | 442.5204 | 16.3961 | 20.0543 | 1.0355 | 2.6050 |  |
| validation | RegionTaskType | RegionB | RealTimeInference | 24.0000 | 234.0000 | 199.6429 | 5.6401 | 7.4769 | 0.5785 | -1.4315 |  |
| validation | RegionTaskType | RegionC | AITraining | 24.0000 | 458.0000 | 592.0102 | 31.4169 | 43.5058 | 1.6463 | 5.5838 |  |
| validation | RegionTaskType | RegionC | BatchInference | 24.0000 | 275.0000 | 348.2245 | 9.8341 | 11.5531 | 0.8582 | 3.0510 |  |
| validation | RegionTaskType | RegionC | RealTimeInference | 24.0000 | 196.0000 | 170.0306 | 5.2571 | 6.6691 | 0.6437 | -1.0821 |  |
| validation | RegionTaskType | RegionD | AITraining | 24.0000 | 3,417.0000 | 4,088.1633 | 64.1817 | 81.1153 | 0.4508 | 27.9651 |  |
| validation | RegionTaskType | RegionD | BatchInference | 24.0000 | 506.0000 | 451.5102 | 12.7656 | 17.8063 | 0.6055 | -2.2704 |  |
| validation | RegionTaskType | RegionD | RealTimeInference | 24.0000 | 14.0000 | 19.8878 | 1.2739 | 2.4262 | 2.1838 | 0.2453 |  |
| validation | RegionTaskType | RegionE | AITraining | 24.0000 | 2,714.0000 | 2,909.1531 | 92.0716 | 106.0779 | 0.8142 | 8.1314 |  |
| validation | RegionTaskType | RegionE | BatchInference | 24.0000 | 275.0000 | 293.9898 | 10.5833 | 13.0655 | 0.9236 | 0.7912 |  |
| validation | RegionTaskType | RegionE | RealTimeInference | 24.0000 | 21.0000 | 26.2857 | 1.4306 | 1.8906 | 1.6349 | 0.2202 |  |
| validation | RegionTaskType | RegionF | AITraining | 24.0000 | 3,493.0000 | 2,938.3061 | 74.4225 | 106.6197 | 0.5113 | -23.1122 |  |
| validation | RegionTaskType | RegionF | BatchInference | 24.0000 | 242.0000 | 272.4082 | 8.9751 | 11.5079 | 0.8901 | 1.2670 |  |
| validation | RegionTaskType | RegionF | RealTimeInference | 24.0000 | 23.0000 | 19.9796 | 1.4439 | 2.3214 | 1.5067 | -0.1259 |  |
| validation | Region |  |  | 24.0000 | 1,022.0000 | 1,320.3980 | 40.0916 | 44.7604 | 0.9415 | 12.4332 | RegionA |
| validation | Region |  |  | 24.0000 | 1,478.0000 | 1,264.2551 | 40.0295 | 57.0546 | 0.6500 | -8.9060 | RegionB |
| validation | Region |  |  | 24.0000 | 929.0000 | 1,110.2653 | 33.5889 | 44.4569 | 0.8677 | 7.5527 | RegionC |
| validation | Region |  |  | 24.0000 | 3,937.0000 | 4,559.5612 | 66.5401 | 83.8889 | 0.4056 | 25.9401 | RegionD |
| validation | Region |  |  | 24.0000 | 3,010.0000 | 3,229.4286 | 96.6032 | 111.1965 | 0.7703 | 9.1429 | RegionE |
| validation | Region |  |  | 24.0000 | 3,758.0000 | 3,230.6939 | 73.4167 | 104.7925 | 0.4689 | -21.9711 | RegionF |
| validation | System |  |  | 24.0000 | 14,134.0000 | 14,714.6020 | 162.0181 | 181.2371 | 0.2751 | 24.1918 |  |
| test | RegionTaskType | RegionA | AITraining | 24.0000 | 579.0000 | 631.2222 | 39.4672 | 48.2305 | 1.6359 | 2.1759 |  |
| test | RegionTaskType | RegionA | BatchInference | 24.0000 | 513.0000 | 450.0303 | 15.0209 | 20.6022 | 0.7027 | -2.6237 |  |
| test | RegionTaskType | RegionA | RealTimeInference | 24.0000 | 208.0000 | 236.1313 | 5.5565 | 6.2327 | 0.6411 | 1.1721 |  |
| test | RegionTaskType | RegionB | AITraining | 24.0000 | 722.0000 | 624.5354 | 36.5889 | 43.6690 | 1.2163 | -4.0610 |  |
| test | RegionTaskType | RegionB | BatchInference | 24.0000 | 527.0000 | 441.8889 | 13.8750 | 19.1385 | 0.6319 | -3.5463 |  |
| test | RegionTaskType | RegionB | RealTimeInference | 24.0000 | 190.0000 | 199.9899 | 3.9999 | 4.9665 | 0.5052 | 0.4162 |  |
| test | RegionTaskType | RegionC | AITraining | 24.0000 | 424.0000 | 590.6566 | 27.7729 | 32.5311 | 1.5721 | 6.9440 |  |
| test | RegionTaskType | RegionC | BatchInference | 24.0000 | 309.0000 | 347.4848 | 9.6649 | 11.5079 | 0.7507 | 1.6035 |  |
| test | RegionTaskType | RegionC | RealTimeInference | 24.0000 | 156.0000 | 170.2929 | 4.3731 | 5.7753 | 0.6728 | 0.5955 |  |
| test | RegionTaskType | RegionD | AITraining | 24.0000 | 3,626.0000 | 4,081.3838 | 85.4311 | 102.4659 | 0.5655 | 18.9743 |  |
| test | RegionTaskType | RegionD | BatchInference | 24.0000 | 537.0000 | 452.0606 | 14.7220 | 18.6237 | 0.6580 | -3.5391 |  |
| test | RegionTaskType | RegionD | RealTimeInference | 24.0000 | 30.0000 | 19.8283 | 1.6631 | 2.4564 | 1.3305 | -0.4238 |  |
| test | RegionTaskType | RegionE | AITraining | 24.0000 | 4,189.0000 | 2,907.1818 | 98.7806 | 138.1184 | 0.5659 | -53.4091 |  |
| test | RegionTaskType | RegionE | BatchInference | 24.0000 | 462.0000 | 293.7980 | 13.3361 | 18.8252 | 0.6928 | -7.0084 |  |
| test | RegionTaskType | RegionE | RealTimeInference | 24.0000 | 31.0000 | 26.2323 | 1.7471 | 2.1204 | 1.3526 | -0.1987 |  |
| test | RegionTaskType | RegionF | AITraining | 24.0000 | 4,696.0000 | 2,943.9091 | 94.6957 | 123.8602 | 0.4840 | -73.0038 |  |
| test | RegionTaskType | RegionF | BatchInference | 24.0000 | 235.0000 | 272.1010 | 9.5979 | 11.4625 | 0.9802 | 1.5459 |  |
| test | RegionTaskType | RegionF | RealTimeInference | 24.0000 | 15.0000 | 20.0101 | 1.0419 | 1.5366 | 1.6670 | 0.2088 |  |
| test | Region |  |  | 24.0000 | 1,300.0000 | 1,317.3838 | 43.9546 | 57.8432 | 0.8115 | 0.7243 | RegionA |
| test | Region |  |  | 24.0000 | 1,439.0000 | 1,266.4141 | 36.4389 | 46.6664 | 0.6077 | -7.1911 | RegionB |
| test | Region |  |  | 24.0000 | 889.0000 | 1,108.4343 | 32.1494 | 37.1625 | 0.8679 | 9.1431 | RegionC |
| test | Region |  |  | 24.0000 | 4,193.0000 | 4,553.2727 | 86.1982 | 104.2919 | 0.4934 | 15.0114 | RegionD |
| test | Region |  |  | 24.0000 | 4,682.0000 | 3,227.2121 | 104.7165 | 148.4691 | 0.5368 | -60.6162 | RegionE |
| test | Region |  |  | 24.0000 | 4,946.0000 | 3,236.0202 | 90.2496 | 120.1434 | 0.4379 | -71.2492 | RegionF |
| test | System |  |  | 24.0000 | 17,449.0000 | 14,708.7374 | 176.6816 | 221.0539 | 0.2430 | -114.1776 |  |

## 区间覆盖与宽度

| n | Coverage80 | Coverage95 | Width80 | Width95 | Stage | Scope |
| --- | --- | --- | --- | --- | --- | --- |
| 432.0000 | 0.8356 | 0.9468 | 61.8663 | 98.2289 | validation | RegionTaskType |
| 432.0000 | 0.7894 | 0.9398 | 61.7805 | 98.2522 | test | RegionTaskType |
| 168.0000 | 0.8512 | 0.9702 | 254.9121 | 355.1055 | validation | Aggregate |
| 168.0000 | 0.8036 | 0.9345 | 254.4793 | 354.2985 | test | Aggregate |

## 图表

- `figures/validation_model_comparison.png`
- `figures/test_system_forecast.png`

## 解释边界

- 点预测只针对聚合 GPU 需求；其用途是预测评价与容量风险描述，不替代问题一最终基础调度的真实任务输入。
- Bootstrap 任务包可用于不确定性情景；该脚本默认仅输出区间汇总，不持久化大量重复的任务级情景文件。
- 不使用 MAPE，因为多个区域×类型×小时单元的实际 GPU 需求为 0。