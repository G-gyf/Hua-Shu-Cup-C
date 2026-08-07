# 本程序及代码是在 AI 工具辅助下完成的。
# AI 工具名称：Codex，开发机构/公司：OpenAI。
"""问题一：短期 GPU 需求预测。

点预测目标为来源区域×任务类型×小时的总 GPU 需求。
预测窗口为 24 小时；每个窗口均在预测原点一次性生成，绝不使用窗口内真实值。
不确定性由完整历史小时任务包的非参数 Bootstrap 给出。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
ROOT = STEP_DIR.parent
OUT_DIR = STEP_DIR / "预测结果"
CONFIG = json.loads((STEP_DIR / "forecast_config.json").read_text(encoding="utf-8"))

KEYS = ["SourceRegion", "TaskType"]
TARGET = "gpu_demand_sum"


def find_task_data() -> Path:
    candidates = [
        path / "processed" / "fact_tasks.csv"
        for path in ROOT.iterdir()
        if path.is_dir() and (path / "processed" / "fact_tasks.csv").is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError("Expected one Step 0 processed/fact_tasks.csv input.")
    return candidates[0]


def reset_output() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def md_table(frame: pd.DataFrame, decimals: int = 4, limit: int | None = None) -> str:
    view = frame.copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.select_dtypes(include=["number"]).columns:
        view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:,.{decimals}f}")
    header = [str(item).replace("|", "\\|") for item in view.columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in view.astype(object).where(pd.notna(view), "").itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |")
    return "\n".join(lines)


def make_panel(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = sorted(tasks["SourceRegion"].unique())
    types = sorted(tasks["TaskType"].unique())
    hours = pd.DataFrame({"Hour": np.arange(2400, dtype="int64")})
    groups = pd.MultiIndex.from_product([regions, types], names=KEYS).to_frame(index=False)
    grid = hours.assign(_k=1).merge(groups.assign(_k=1), on="_k").drop(columns="_k")
    aggregated = tasks.groupby(["ArrivalHour", *KEYS], as_index=False).agg(
        task_count=("TaskID", "size"),
        gpu_demand_sum=("GPU_Demand", "sum"),
    ).rename(columns={"ArrivalHour": "Hour"})
    panel = grid.merge(aggregated, on=["Hour", *KEYS], how="left")
    panel[["task_count", TARGET]] = panel[["task_count", TARGET]].fillna(0.0)
    panel["task_count"] = panel["task_count"].astype(int)
    return panel.sort_values([*KEYS, "Hour"]).reset_index(drop=True), groups


def target_window(panel: pd.DataFrame, origin: int, horizon: int) -> pd.DataFrame:
    return panel.loc[panel["Hour"].between(origin + 1, origin + horizon), ["Hour", *KEYS, TARGET]].copy()


def prediction_grid(origin: int, horizon: int, groups: pd.DataFrame) -> pd.DataFrame:
    future = pd.DataFrame({"Hour": np.arange(origin + 1, origin + horizon + 1, dtype=int)})
    return future.assign(_k=1).merge(groups.assign(_k=1), on="_k").drop(columns="_k")


def candidate_prediction(panel: pd.DataFrame, groups: pd.DataFrame, origin: int, horizon: int,
                         model_name: str) -> pd.DataFrame:
    """Generate an entire horizon using only Hour <= origin."""
    history = panel.loc[panel["Hour"] <= origin].copy()
    result = prediction_grid(origin, horizon, groups)
    if model_name == "direct_mean":
        values = history.groupby(KEYS)[TARGET].mean().rename("PointForecast")
        return result.merge(values, left_on=KEYS, right_index=True, how="left")

    if model_name.startswith("rolling_"):
        window = int(model_name.split("_")[1])
        values = history.loc[history["Hour"] > origin - window].groupby(KEYS)[TARGET].mean().rename("PointForecast")
        return result.merge(values, left_on=KEYS, right_index=True, how="left")

    if model_name == "seasonal_daily":
        source = panel.loc[panel["Hour"].between(origin - 23, origin), ["Hour", *KEYS, TARGET]].copy()
        source["Hour"] += 24
        return result.merge(source.rename(columns={TARGET: "PointForecast"}), on=["Hour", *KEYS], how="left")

    if model_name == "seasonal_weekly":
        source = panel.loc[panel["Hour"].between(origin - 167, origin - 144), ["Hour", *KEYS, TARGET]].copy()
        source["Hour"] += 168
        return result.merge(source.rename(columns={TARGET: "PointForecast"}), on=["Hour", *KEYS], how="left")
    raise ValueError(f"Unknown model: {model_name}")


def score_predictions(predictions: pd.DataFrame, actual: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    merged = predictions.merge(actual, on=["Hour", *KEYS], how="left", validate="one_to_one")
    merged["AbsoluteError"] = (merged["PointForecast"] - merged[TARGET]).abs()
    merged["SquaredError"] = (merged["PointForecast"] - merged[TARGET]) ** 2
    merged["SignedError"] = merged["PointForecast"] - merged[TARGET]

    def one_score(frame: pd.DataFrame) -> dict[str, float]:
        total = frame[TARGET].sum()
        return {
            "n": len(frame),
            "ActualSum": total,
            "ForecastSum": frame["PointForecast"].sum(),
            "MAE": frame["AbsoluteError"].mean(),
            "RMSE": np.sqrt(frame["SquaredError"].mean()),
            "WAPE": frame["AbsoluteError"].sum() / total if total > 0 else np.nan,
            "Bias": frame["SignedError"].mean(),
        }

    if group_cols:
        rows = [{**dict(zip(group_cols, values)), **one_score(group)}
                for values, group in merged.groupby(group_cols, dropna=False)]
        return pd.DataFrame(rows)
    return pd.DataFrame([one_score(merged)])


def official_model_selection(panel: pd.DataFrame, groups: pd.DataFrame, config: dict) -> tuple[str, pd.DataFrame]:
    origin = int(config["validation_train_end_hour"])
    horizon = int(config["horizon_hours"])
    candidates = ["direct_mean", *[f"rolling_{n}" for n in config["candidate_rolling_windows"]],
                  "seasonal_daily", "seasonal_weekly"]
    actual = target_window(panel, origin, horizon)
    rows = []
    for name in candidates:
        metrics = score_predictions(candidate_prediction(panel, groups, origin, horizon, name), actual, [])
        rows.append({"Model": name, "OriginHour": origin, **metrics.iloc[0].to_dict()})
    comparison = pd.DataFrame(rows).sort_values(["WAPE", "MAE"]).reset_index(drop=True)
    return str(comparison.iloc[0]["Model"]), comparison


def rolling_backtest(panel: pd.DataFrame, groups: pd.DataFrame, config: dict) -> pd.DataFrame:
    horizon = int(config["horizon_hours"])
    first = int(config["rolling_backtest_first_origin"])
    last = int(config["validation_train_end_hour"]) - horizon
    candidates = ["direct_mean", *[f"rolling_{n}" for n in config["candidate_rolling_windows"]],
                  "seasonal_daily", "seasonal_weekly"]
    rows = []
    for origin in range(first, last + 1, int(config["rolling_backtest_step_hours"])):
        actual = target_window(panel, origin, horizon)
        for name in candidates:
            metric = score_predictions(candidate_prediction(panel, groups, origin, horizon, name), actual, []).iloc[0]
            rows.append({"Model": name, "OriginHour": origin, **metric.to_dict()})
    detail = pd.DataFrame(rows)
    summary = detail.groupby("Model", as_index=False).agg(
        Origins=("OriginHour", "nunique"), MAE_mean=("MAE", "mean"), RMSE_mean=("RMSE", "mean"),
        WAPE_mean=("WAPE", "mean"), WAPE_std=("WAPE", "std"), Bias_mean=("Bias", "mean"),
    ).sort_values("WAPE_mean")
    write_csv(detail, "rolling_backtest_detail.csv")
    return summary


def bootstrap_simulation(panel: pd.DataFrame, groups: pd.DataFrame, origin: int,
                         horizon: int, simulations: int, seed: int) -> tuple[np.ndarray, list[tuple[str, str]]]:
    ordered_groups = list(groups.itertuples(index=False, name=None))
    historical = panel.loc[panel["Hour"] <= origin].pivot(index="Hour", columns=KEYS, values=TARGET)
    historical = historical.reindex(columns=pd.MultiIndex.from_tuples(ordered_groups, names=KEYS)).fillna(0.0)
    rng = np.random.default_rng(seed)
    sampled_hours = rng.integers(0, len(historical), size=(simulations, horizon))
    simulation = historical.to_numpy()[sampled_hours]
    return simulation, ordered_groups


def interval_levels(config: dict, scope: str) -> tuple[float, float, float, float]:
    settings = config["calibrated_interval_quantiles"][scope]
    lower80, upper80 = settings["interval80"]
    lower95, upper95 = settings["interval95"]
    return float(lower80), float(upper80), float(lower95), float(upper95)


def group_intervals(simulation: np.ndarray, ordered_groups: list[tuple[str, str]], origin: int,
                    config: dict) -> pd.DataFrame:
    lower80, upper80, lower95, upper95 = interval_levels(config, "region_task_type")
    quantiles = np.quantile(simulation, [lower95, lower80, 0.50, upper80, upper95], axis=0)
    rows = []
    for h in range(simulation.shape[1]):
        for j, (region, task_type) in enumerate(ordered_groups):
            rows.append({
                "Hour": origin + h + 1, "SourceRegion": region, "TaskType": task_type,
                "Lower95": quantiles[0, h, j], "Lower80": quantiles[1, h, j],
                "BootstrapMedian": quantiles[2, h, j], "Upper80": quantiles[3, h, j], "Upper95": quantiles[4, h, j],
            })
    return pd.DataFrame(rows)


def aggregate_intervals(simulation: np.ndarray, ordered_groups: list[tuple[str, str]], origin: int,
                        config: dict) -> pd.DataFrame:
    regions = sorted({region for region, _ in ordered_groups})
    lower80, upper80, lower95, upper95 = interval_levels(config, "aggregate")
    rows = []
    for label in regions + ["System"]:
        indices = [i for i, (region, _) in enumerate(ordered_groups) if label == "System" or region == label]
        values = simulation[:, :, indices].sum(axis=2)
        quantiles = np.quantile(values, [lower95, lower80, 0.50, upper80, upper95], axis=0)
        for h in range(values.shape[1]):
            rows.append({
                "Hour": origin + h + 1, "AggregateLevel": "System" if label == "System" else "Region",
                "AggregateName": label, "Lower95": quantiles[0, h], "Lower80": quantiles[1, h],
                "BootstrapMedian": quantiles[2, h], "Upper80": quantiles[3, h], "Upper95": quantiles[4, h],
            })
    return pd.DataFrame(rows)


def point_and_interval_results(panel: pd.DataFrame, groups: pd.DataFrame, model: str, origin: int,
                               stage: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[tuple[str, str]]]:
    horizon = int(config["horizon_hours"])
    point = candidate_prediction(panel, groups, origin, horizon, model)
    actual = target_window(panel, origin, horizon).rename(columns={TARGET: "ActualGPU"})
    simulation, ordered_groups = bootstrap_simulation(panel, groups, origin, horizon,
                                                       int(config["bootstrap_simulations"]),
                                                       int(config["random_seed"]) + origin)
    intervals = group_intervals(simulation, ordered_groups, origin, config)
    result = point.merge(actual, on=["Hour", *KEYS], how="left").merge(intervals, on=["Hour", *KEYS], how="left")
    result.insert(0, "Stage", stage)
    result.insert(1, "ForecastOriginHour", origin)
    result.insert(2, "Model", model)
    aggregates = aggregate_intervals(simulation, ordered_groups, origin, config)
    point_regions = point.groupby(["Hour", "SourceRegion"], as_index=False)["PointForecast"].sum().rename(columns={"SourceRegion": "AggregateName"})
    actual_regions = actual.groupby(["Hour", "SourceRegion"], as_index=False)["ActualGPU"].sum().rename(columns={"SourceRegion": "AggregateName"})
    point_regions["AggregateLevel"] = "Region"; actual_regions["AggregateLevel"] = "Region"
    point_system = point.groupby("Hour", as_index=False)["PointForecast"].sum(); point_system["AggregateLevel"] = "System"; point_system["AggregateName"] = "System"
    actual_system = actual.groupby("Hour", as_index=False)["ActualGPU"].sum(); actual_system["AggregateLevel"] = "System"; actual_system["AggregateName"] = "System"
    aggregate_point = pd.concat([point_regions, point_system], ignore_index=True)
    aggregate_actual = pd.concat([actual_regions, actual_system], ignore_index=True)
    aggregates = aggregates.merge(aggregate_point, on=["Hour", "AggregateLevel", "AggregateName"], how="left").merge(
        aggregate_actual, on=["Hour", "AggregateLevel", "AggregateName"], how="left"
    )
    aggregates.insert(0, "Stage", stage); aggregates.insert(1, "ForecastOriginHour", origin); aggregates.insert(2, "Model", model)
    return result, aggregates, simulation, ordered_groups


def interval_metrics(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    def calculate(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "n": len(frame),
            "Coverage80": ((frame["ActualGPU"] >= frame["Lower80"]) & (frame["ActualGPU"] <= frame["Upper80"])).mean(),
            "Coverage95": ((frame["ActualGPU"] >= frame["Lower95"]) & (frame["ActualGPU"] <= frame["Upper95"])).mean(),
            "Width80": (frame["Upper80"] - frame["Lower80"]).mean(),
            "Width95": (frame["Upper95"] - frame["Lower95"]).mean(),
        }
    if not group_cols:
        return pd.DataFrame([calculate(results)])
    return pd.DataFrame([{**dict(zip(group_cols, vals)), **calculate(group)}
                         for vals, group in results.groupby(group_cols)])


def value_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Score any frame that has ActualGPU and PointForecast without assuming task-type keys."""
    scored = frame.copy()
    scored["AbsoluteError"] = (scored["PointForecast"] - scored["ActualGPU"]).abs()
    scored["SquaredError"] = (scored["PointForecast"] - scored["ActualGPU"]) ** 2
    scored["SignedError"] = scored["PointForecast"] - scored["ActualGPU"]

    def one_score(part: pd.DataFrame) -> dict[str, float]:
        actual_sum = part["ActualGPU"].sum()
        return {
            "n": len(part), "ActualSum": actual_sum, "ForecastSum": part["PointForecast"].sum(),
            "MAE": part["AbsoluteError"].mean(), "RMSE": np.sqrt(part["SquaredError"].mean()),
            "WAPE": part["AbsoluteError"].sum() / actual_sum if actual_sum > 0 else np.nan,
            "Bias": part["SignedError"].mean(),
        }
    if not group_cols:
        return pd.DataFrame([one_score(scored)])
    return pd.DataFrame([{**dict(zip(group_cols, values)), **one_score(group)}
                         for values, group in scored.groupby(group_cols)])


def selected_metrics(results: pd.DataFrame, aggregates: pd.DataFrame, stage: str) -> pd.DataFrame:
    by_group = value_metrics(results, KEYS)
    by_group.insert(0, "Scope", "RegionTaskType")
    regional = value_metrics(aggregates[aggregates["AggregateLevel"] == "Region"], ["AggregateName"])
    regional.insert(0, "Scope", "Region")
    system = value_metrics(aggregates[aggregates["AggregateLevel"] == "System"], [])
    system.insert(0, "Scope", "System")
    combined = pd.concat([by_group, regional, system], ignore_index=True, sort=False)
    combined.insert(0, "Stage", stage)
    return combined


def create_figures(comparison: pd.DataFrame, aggregate_results: pd.DataFrame) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; figures skipped"]

    files = []
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = comparison.sort_values("WAPE")
    ax.bar(ordered["Model"], ordered["WAPE"], color="#4472C4")
    ax.set(title="Validation WAPE by candidate model", xlabel="Model", ylabel="WAPE")
    ax.tick_params(axis="x", rotation=25); fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "validation_model_comparison.png", dpi=180); plt.close(fig)
    files.append("validation_model_comparison.png")

    system = aggregate_results[(aggregate_results["Stage"] == "test") & (aggregate_results["AggregateLevel"] == "System")]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(system["Hour"], system["ActualGPU"], marker="o", label="Actual", color="#C00000")
    ax.plot(system["Hour"], system["PointForecast"], marker="o", label="Point forecast", color="#4472C4")
    ax.fill_between(system["Hour"], system["Lower80"], system["Upper80"], alpha=0.22, color="#4472C4", label="Calibrated 80% interval")
    ax.set(title="System GPU demand: 2376-2399", xlabel="Hour", ylabel="GPU demand")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "test_system_forecast.png", dpi=180); plt.close(fig)
    files.append("test_system_forecast.png")
    return files


def write_diagnostics(selected_model: str, comparison: pd.DataFrame, rolling: pd.DataFrame,
                      metrics: pd.DataFrame, interval_summary: pd.DataFrame, figures: Iterable[str], config: dict) -> None:
    group_levels = interval_levels(config, "region_task_type")
    aggregate_levels = interval_levels(config, "aggregate")
    lines = [
        "# 问题一短期 GPU 需求预测诊断", "",
        "## 预测口径", "",
        "- 目标：来源区域 × 任务类型 × 小时的总 GPU 需求。",
        "- 预测协议：每个验证/测试窗口均在预测原点一次性预测未来 24 小时；窗口内真实任务不参与任何特征或模型更新。",
        f"- 官方验证集选择的点预测模型：`{selected_model}`。",
        f"- 区间：从预测原点之前的完整历史小时任务包进行 {config['bootstrap_simulations']:,} 次 Bootstrap；每个任务包保留任务数、GPU 需求、时长和 SLA 属性的原始联合关系。",
        f"- 区域×类型经验校准区间：80% 为 P{group_levels[0] * 100:g}–P{group_levels[1] * 100:g}；95% 为 P{group_levels[2] * 100:g}–P{group_levels[3] * 100:g}。",
        f"- 区域/系统汇总经验校准区间：80% 为 P{aggregate_levels[0] * 100:g}–P{aggregate_levels[1] * 100:g}；95% 为 P{aggregate_levels[2] * 100:g}–P{aggregate_levels[3] * 100:g}。校准依据：{config['interval_calibration_basis']}",
        "",
        "## 官方验证集候选模型比较", "", md_table(comparison), "",
        "## 训练期滚动 24 小时回测", "", md_table(rolling), "",
        "## 选定模型的误差", "", md_table(metrics), "",
        "## 区间覆盖与宽度", "", md_table(interval_summary), "",
        "## 图表", "",
    ]
    lines.extend([f"- `figures/{item}`" for item in figures])
    lines.extend([
        "", "## 解释边界", "",
        "- 点预测只针对聚合 GPU 需求；其用途是预测评价与容量风险描述，不替代问题一最终基础调度的真实任务输入。",
        "- Bootstrap 任务包可用于不确定性情景；该脚本默认仅输出区间汇总，不持久化大量重复的任务级情景文件。",
        "- 不使用 MAPE，因为多个区域×类型×小时单元的实际 GPU 需求为 0。",
    ])
    (OUT_DIR / "forecast_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reset_output()
    tasks = pd.read_csv(find_task_data())
    panel, groups = make_panel(tasks)
    selected_model, comparison = official_model_selection(panel, groups, CONFIG)
    rolling = rolling_backtest(panel, groups, CONFIG)

    validation, validation_agg, _, _ = point_and_interval_results(
        panel, groups, selected_model, int(CONFIG["validation_train_end_hour"]), "validation", CONFIG
    )
    test, test_agg, _, _ = point_and_interval_results(
        panel, groups, selected_model, int(CONFIG["final_train_end_hour"]), "test", CONFIG
    )
    point_results = pd.concat([validation, test], ignore_index=True)
    aggregate_results = pd.concat([validation_agg, test_agg], ignore_index=True)
    metric_results = pd.concat([
        selected_metrics(validation, validation_agg, "validation"),
        selected_metrics(test, test_agg, "test"),
    ], ignore_index=True)
    interval_summary = pd.concat([
        interval_metrics(validation, []).assign(Stage="validation", Scope="RegionTaskType"),
        interval_metrics(test, []).assign(Stage="test", Scope="RegionTaskType"),
        interval_metrics(validation_agg, []).assign(Stage="validation", Scope="Aggregate"),
        interval_metrics(test_agg, []).assign(Stage="test", Scope="Aggregate"),
    ], ignore_index=True)

    write_csv(comparison, "forecast_model_comparison.csv")
    write_csv(rolling, "forecast_rolling_backtest.csv")
    write_csv(point_results, "forecast_point_results.csv")
    write_csv(aggregate_results, "forecast_aggregate_results.csv")
    write_csv(metric_results, "forecast_metrics.csv")
    write_csv(interval_summary, "forecast_interval_summary.csv")
    interval_rows = []
    for scope in ["region_task_type", "aggregate"]:
        lower80, upper80, lower95, upper95 = interval_levels(CONFIG, scope)
        interval_rows.append({
            "Scope": scope, "BootstrapSimulations": CONFIG["bootstrap_simulations"],
            "Interval80LowerQuantile": lower80, "Interval80UpperQuantile": upper80,
            "Interval95LowerQuantile": lower95, "Interval95UpperQuantile": upper95,
            "CalibrationBasis": CONFIG["interval_calibration_basis"],
        })
    write_csv(pd.DataFrame(interval_rows), "forecast_interval_config.csv")
    figures = create_figures(comparison, aggregate_results)
    write_diagnostics(selected_model, comparison, rolling, metric_results, interval_summary, figures, CONFIG)

    print("Question 1 forecast completed.")
    print(f"Selected model: {selected_model}")
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
