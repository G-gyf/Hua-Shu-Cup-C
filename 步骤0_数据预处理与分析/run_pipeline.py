# 本程序及代码是在 AI 工具辅助下完成的。
# AI 工具名称：Codex，开发机构/公司：OpenAI。
"""C 题数据预处理、质量控制与探索分析的可复现管道。

原始附件只读；所有产物写入 processed/ 和 reports/。
任务按整数小时开工、分钟级时长保留，逐小时占用由 task_hour_profile.csv 表示。
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
RAW_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and path.name == "附件数据")
PROCESSED_DIR = PIPELINE_DIR / "processed"
REPORT_DIR = PIPELINE_DIR / "reports"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
CONFIG_PATH = PIPELINE_DIR / "scenario_config.json"

TASK_COLUMNS = [
    "TaskID", "TaskType", "ArrivalHour", "GPU_Demand", "EstimatedDuration_min",
    "DelaySensitivity", "SourceRegion", "MaxLatency_ms", "LatestFinishHour",
    "EarliestStartHour", "ExecutionMode",
]
REGION_INPUT_COLUMNS = [
    "Hour", "Region", "PricePeriod", "ElectricityPrice_CNY_per_MWh",
    "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh",
    "AvailableRenewable_MW", "NonAI_IT_Load_MW", "DemandResponseLevel", "DataPeriod",
]
REGION_BASELINE_COLUMNS = [
    "Hour", "Region", "AITrainingPower_MW", "GPU_Utilization_Percent",
    "UsedRenewable_MW", "RenewableCharge_MW", "Curtailment_MW", "IT_Load_MW",
    "Total_Load_MW", "GridPurchase_MW", "GridCharge_MW", "GridSell_MW",
    "NetGridImport_MW", "CarbonEmission_tCO2", "SOC_MWh", "ChargePower_MW",
    "DischargePower_MW", "Baseline_AI_IT_Load_MW",
]


def setup_output_dirs() -> None:
    """Rebuild generated directories without touching the raw attachment directory."""
    for folder in (PROCESSED_DIR, REPORT_DIR):
        if folder.exists():
            shutil.rmtree(folder)
    for folder in (PROCESSED_DIR, REPORT_DIR, TABLE_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_first_sheet(stem: str) -> pd.DataFrame:
    file_path = RAW_DIR / f"{stem}.xlsx"
    return pd.read_excel(file_path, sheet_name=0)


def write_csv(frame: pd.DataFrame, name: str, directory: Path = PROCESSED_DIR) -> Path:
    path = directory / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None, decimals: int = 3) -> str:
    view = frame.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for column in view.select_dtypes(include=["number"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:,.{decimals}f}"
        )
    # Avoid pandas.DataFrame.to_markdown(), which requires the optional tabulate package.
    headers = [str(column).replace("|", "\\|") for column in view.columns]
    rows = []
    for values in view.astype(object).where(pd.notna(view), "").itertuples(index=False, name=None):
        rows.append([str(value).replace("|", "\\|").replace("\n", "<br>") for value in values])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def collect_manifest() -> dict[str, Any]:
    source_files = []
    for file_path in sorted(RAW_DIR.glob("*.xlsx")):
        excel = pd.ExcelFile(file_path)
        sheets = []
        for sheet in excel.sheet_names:
            raw = pd.read_excel(file_path, sheet_name=sheet, header=None)
            sheets.append({"sheet": sheet, "rows": int(raw.shape[0]), "columns": int(raw.shape[1])})
        source_files.append({
            "file": file_path.name,
            "sha256": sha256(file_path),
            "sheets": sheets,
        })
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "raw_directory": str(RAW_DIR),
        "source_files": source_files,
    }


def load_and_standardize() -> dict[str, pd.DataFrame]:
    gpu = read_first_sheet("GPU_information")
    workload = read_first_sheet("workload_trace")
    latency = read_first_sheet("network_latency")
    power = read_first_sheet("power_mapping")
    region_time = read_first_sheet("region_time_data")
    storage = read_first_sheet("storage_information")

    workload = workload[TASK_COLUMNS].copy()
    workload["TaskID"] = workload["TaskID"].astype("int64")
    for column in ("ArrivalHour", "GPU_Demand", "EstimatedDuration_min", "MaxLatency_ms",
                   "LatestFinishHour", "EarliestStartHour"):
        workload[column] = pd.to_numeric(workload[column], errors="raise")
    workload["duration_h"] = workload["EstimatedDuration_min"] / 60.0
    workload["duration_ceil_h"] = np.ceil(workload["duration_h"]).astype("int64")
    workload["immediate_finish_hour"] = workload["ArrivalHour"] + workload["duration_h"]
    workload["immediate_finish_boundary"] = workload["ArrivalHour"] + workload["duration_ceil_h"]
    workload["gpu_hours"] = workload["GPU_Demand"] * workload["duration_h"]

    region_time["Hour"] = pd.to_numeric(region_time["Hour"], errors="raise").astype("int64")
    region_time = region_time.sort_values(["Region", "Hour"]).reset_index(drop=True)

    gpu = gpu.sort_values("Region").reset_index(drop=True)
    latency = latency.sort_values(["FromRegion", "ToRegion"]).reset_index(drop=True)
    power = power.sort_values("TaskType").reset_index(drop=True)
    storage = storage.sort_values("Region").reset_index(drop=True)

    return {
        "dim_region": gpu,
        "fact_tasks": workload,
        "dim_network_latency": latency,
        "dim_task_power": power,
        "region_time": region_time,
        "dim_storage": storage,
    }


def build_task_hour_profile(tasks: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    power_lookup = power.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"]
    profile_rows: list[pd.DataFrame] = []
    for relative_hour in range(int(tasks["duration_ceil_h"].max())):
        chunk = tasks.loc[tasks["duration_h"] > relative_hour,
                          ["TaskID", "TaskType", "GPU_Demand", "duration_h"]].copy()
        chunk["relative_hour"] = relative_hour
        chunk["overlap_h"] = np.minimum(1.0, chunk["duration_h"] - relative_hour)
        chunk["gpu_hour"] = chunk["GPU_Demand"] * chunk["overlap_h"]
        chunk["gpu_power_mw"] = chunk["TaskType"].map(power_lookup)
        chunk["ai_it_energy_mwh"] = chunk["gpu_hour"] * chunk["gpu_power_mw"]
        profile_rows.append(chunk.drop(columns=["duration_h"]))
    profile = pd.concat(profile_rows, ignore_index=True)
    return profile.sort_values(["TaskID", "relative_hour"]).reset_index(drop=True)


def build_task_region_eligibility(tasks: pd.DataFrame, latency: pd.DataFrame) -> pd.DataFrame:
    task_context = tasks[["TaskID", "TaskType", "SourceRegion", "MaxLatency_ms"]]
    eligibility = task_context.merge(
        latency,
        left_on="SourceRegion",
        right_on="FromRegion",
        how="left",
        validate="many_to_many",
    )
    eligibility["is_latency_feasible"] = (
        eligibility["NetworkLatency_ms"] <= eligibility["MaxLatency_ms"]
    )
    return eligibility.sort_values(["TaskID", "ToRegion"]).reset_index(drop=True)


def build_forecast_panel(tasks: pd.DataFrame, regions: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    hours = pd.DataFrame({"Hour": np.arange(0, 2400, dtype="int64")})
    grid = (
        hours.assign(_key=1)
        .merge(regions[["Region"]].rename(columns={"Region": "SourceRegion"}).assign(_key=1), on="_key")
        .merge(power[["TaskType"]].assign(_key=1), on="_key")
        .drop(columns="_key")
    )
    aggregate = tasks.groupby(["ArrivalHour", "SourceRegion", "TaskType"], as_index=False).agg(
        task_count=("TaskID", "size"),
        gpu_demand_sum=("GPU_Demand", "sum"),
        gpu_hours_sum=("gpu_hours", "sum"),
    ).rename(columns={"ArrivalHour": "Hour"})
    panel = grid.merge(aggregate, on=["Hour", "SourceRegion", "TaskType"], how="left")
    panel[["task_count", "gpu_demand_sum", "gpu_hours_sum"]] = panel[
        ["task_count", "gpu_demand_sum", "gpu_hours_sum"]
    ].fillna(0)
    panel["task_count"] = panel["task_count"].astype("int64")
    panel["hour_of_day"] = panel["Hour"] % 24
    panel["day_index"] = panel["Hour"] // 24
    panel["day_of_week"] = panel["day_index"] % 7
    panel["is_weekend"] = panel["day_of_week"].isin([5, 6]).astype("int64")

    group_cols = ["SourceRegion", "TaskType"]
    panel = panel.sort_values(group_cols + ["Hour"]).reset_index(drop=True)
    for target in ("task_count", "gpu_demand_sum", "gpu_hours_sum"):
        grouped = panel.groupby(group_cols, sort=False)[target]
        panel[f"{target}_lag_1"] = grouped.shift(1)
        panel[f"{target}_lag_24"] = grouped.shift(24)
        panel[f"{target}_lag_168"] = grouped.shift(168)
        panel[f"{target}_roll_mean_24"] = grouped.transform(lambda series: series.shift(1).rolling(24).mean())
        panel[f"{target}_roll_mean_168"] = grouped.transform(lambda series: series.shift(1).rolling(168).mean())
    panel["split"] = np.select(
        [panel["Hour"] <= 2351, panel["Hour"] <= 2375],
        ["train", "validation"],
        default="test",
    )
    return panel


def check_quality(data: dict[str, pd.DataFrame], profile: pd.DataFrame,
                  eligibility: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = data["fact_tasks"]
    region_time = data["region_time"]
    regions = data["dim_region"]
    storage = data["dim_storage"]
    latency = data["dim_network_latency"]
    power = data["dim_task_power"]
    checks: list[dict[str, Any]] = []

    def record(name: str, failed_count: int, detail: str, maximum_error: float | None = None) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if failed_count == 0 else "FLAG",
            "failed_count": int(failed_count),
            "maximum_error": maximum_error,
            "detail": detail,
        })

    record("TaskID uniqueness", int(tasks["TaskID"].duplicated().sum()), "TaskID must be unique.")
    record("Task missing values", int(tasks.isna().sum().sum()), "Task table has no missing cells.")
    record("Task arrival range", int((~tasks["ArrivalHour"].between(0, 2399)).sum()), "ArrivalHour must be 0-2399.")
    record("Earliest start equals arrival", int((tasks["EarliestStartHour"] != tasks["ArrivalHour"]).sum()),
           "Attachment task records use arrival as earliest start.")
    record("Task immediate completion boundary", int((tasks["immediate_finish_boundary"] > 2406).sum()),
           "Immediate execution must finish no later than the 2406 boundary.")
    record("Region-hour key uniqueness", int(region_time.duplicated(["Region", "Hour"]).sum()),
           "(Region, Hour) must be unique.")
    expected_key_count = len(regions) * 2407
    record("Region-hour coverage", abs(len(region_time) - expected_key_count),
           f"Expected {expected_key_count} records for 6 regions and hours 0-2406.")
    record("Region-time missing values", int(region_time.isna().sum().sum()), "Region-time table has no missing cells.")
    record("Latency matrix completeness", abs(len(latency) - len(regions) ** 2), "Expected a 6x6 directed matrix.")
    record("Latency key uniqueness", int(latency.duplicated(["FromRegion", "ToRegion"]).sum()),
           "Each directed pair must appear once.")

    profile_sum = profile.groupby("TaskID", as_index=False)["overlap_h"].sum().merge(
        tasks[["TaskID", "duration_h"]], on="TaskID", how="outer", validate="one_to_one"
    )
    profile_error = (profile_sum["overlap_h"] - profile_sum["duration_h"]).abs()
    record("Task overlap-duration conservation", int((profile_error > 1e-10).sum()),
           "Sum of relative-hour overlaps must equal task duration in hours.", float(profile_error.max()))
    feasible_counts = eligibility.groupby("TaskID")["is_latency_feasible"].sum()
    record("Task latency feasibility", int((feasible_counts == 0).sum()),
           "Each task must have at least one destination satisfying its MaxLatency_ms.")

    pue = regions.set_index("Region")["PUE"]
    rt = region_time.copy()
    rt["PUE"] = rt["Region"].map(pue)
    identities = {
        "Baseline IT split": rt["IT_Load_MW"] - rt["NonAI_IT_Load_MW"] - rt["Baseline_AI_IT_Load_MW"],
        "Baseline PUE conversion": rt["Total_Load_MW"] - rt["IT_Load_MW"] * rt["PUE"],
        "Baseline charge split": rt["ChargePower_MW"] - rt["RenewableCharge_MW"] - rt["GridCharge_MW"],
        "Baseline net import": rt["NetGridImport_MW"] - rt["GridPurchase_MW"] + rt["GridSell_MW"],
        "Baseline carbon": rt["CarbonEmission_tCO2"] - rt["GridPurchase_MW"] * rt["CarbonIntensity_tCO2_per_MWh"],
        "Baseline renewable allocation": rt["AvailableRenewable_MW"] - rt["UsedRenewable_MW"]
            - rt["RenewableCharge_MW"] - rt["Curtailment_MW"] - rt["GridSell_MW"],
        "Baseline power balance": rt["GridPurchase_MW"] + rt["AvailableRenewable_MW"] + rt["DischargePower_MW"]
            - rt["Total_Load_MW"] - rt["ChargePower_MW"] - rt["GridSell_MW"] - rt["Curtailment_MW"],
    }
    for name, residual in identities.items():
        record(name, int((residual.abs() > 1e-3).sum()), "Verified against the formula in Attachment 1.",
               float(residual.abs().max()))

    constraints = rt.merge(storage[["Region", "StorageCapacity_MWh", "MinSOC_MWh", "InitialSOC_MWh",
                                     "MaxChargePower_MW", "MaxDischargePower_MW", "MaxGridImport_MW",
                                     "MaxGridExport_MW", "ChargeEfficiency", "DischargeEfficiency"]], on="Region")
    limit_checks = {
        "Baseline SOC minimum": constraints["SOC_MWh"] < constraints["MinSOC_MWh"] - 1e-6,
        "Baseline SOC capacity": constraints["SOC_MWh"] > constraints["StorageCapacity_MWh"] + 1e-6,
        "Baseline charge power limit": constraints["ChargePower_MW"] > constraints["MaxChargePower_MW"] + 1e-6,
        "Baseline discharge power limit": constraints["DischargePower_MW"] > constraints["MaxDischargePower_MW"] + 1e-6,
        "Baseline grid import limit": constraints["GridPurchase_MW"] > constraints["MaxGridImport_MW"] + 1e-6,
        "Baseline grid export limit": constraints["GridSell_MW"] > constraints["MaxGridExport_MW"] + 1e-6,
    }
    for name, failure in limit_checks.items():
        record(name, int(failure.sum()), "Checked against storage_information.xlsx.")

    soc_max_error = 0.0
    soc_failures = 0
    for region, group in constraints.sort_values(["Region", "Hour"]).groupby("Region"):
        initial = group["InitialSOC_MWh"].iloc[0]
        expected_previous = group["SOC_MWh"].shift(1).fillna(initial)
        expected_soc = expected_previous + group["ChargeEfficiency"] * group["ChargePower_MW"] - (
            group["DischargePower_MW"] / group["DischargeEfficiency"]
        )
        error = (group["SOC_MWh"] - expected_soc).abs()
        soc_max_error = max(soc_max_error, float(error.max()))
        soc_failures += int((error > 1e-3).sum())
    record("Baseline SOC recurrence", soc_failures, "Verified against Attachment 1 SOC formula.", soc_max_error)

    quality = pd.DataFrame(checks)
    metadata = {
        "task_rows": int(len(tasks)),
        "region_hour_rows": int(len(region_time)),
        "task_hour_profile_rows": int(len(profile)),
        "task_region_eligibility_rows": int(len(eligibility)),
        "quality_passes": int((quality["status"] == "PASS").sum()),
        "quality_flags": int((quality["status"] == "FLAG").sum()),
    }
    return quality, metadata


def build_summary_tables(data: dict[str, pd.DataFrame], eligibility: pd.DataFrame,
                         forecast: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tasks = data["fact_tasks"]
    region_time = data["region_time"]

    by_type = tasks.groupby("TaskType", as_index=False).agg(
        tasks=("TaskID", "size"),
        gpu_demand_min=("GPU_Demand", "min"),
        gpu_demand_mean=("GPU_Demand", "mean"),
        gpu_demand_max=("GPU_Demand", "max"),
        duration_min_mean=("EstimatedDuration_min", "mean"),
        duration_min_max=("EstimatedDuration_min", "max"),
        total_gpu_hours=("gpu_hours", "sum"),
    )
    by_type["task_share"] = by_type["tasks"] / len(tasks)

    by_source = tasks.groupby("SourceRegion", as_index=False).agg(
        tasks=("TaskID", "size"), total_gpu_hours=("gpu_hours", "sum"),
        gpu_demand_mean=("GPU_Demand", "mean"), duration_min_mean=("EstimatedDuration_min", "mean"),
    )
    feasible = eligibility.groupby("TaskID", as_index=False).agg(
        feasible_region_count=("is_latency_feasible", "sum")
    ).merge(tasks[["TaskID", "TaskType"]], on="TaskID", validate="one_to_one")
    feasibility_by_type = feasible.groupby("TaskType", as_index=False).agg(
        feasible_region_min=("feasible_region_count", "min"),
        feasible_region_mean=("feasible_region_count", "mean"),
        feasible_region_max=("feasible_region_count", "max"),
    )

    energy_cols = [
        "ElectricityPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh", "AvailableRenewable_MW",
        "Baseline_AI_IT_Load_MW", "NonAI_IT_Load_MW", "NetGridImport_MW",
    ]
    regional_energy = region_time.groupby("Region")[energy_cols].agg(["min", "mean", "max"])
    regional_energy.columns = [f"{field}_{stat}" for field, stat in regional_energy.columns]
    regional_energy = regional_energy.reset_index()

    forecast_windows = forecast.groupby("split", as_index=False).agg(
        records=("Hour", "size"), task_count=("task_count", "sum"),
        gpu_demand_sum=("gpu_demand_sum", "sum"), gpu_hours_sum=("gpu_hours_sum", "sum"),
    )
    return {
        "task_type_summary.csv": by_type,
        "task_source_region_summary.csv": by_source,
        "latency_feasibility_by_type.csv": feasibility_by_type,
        "region_energy_summary.csv": regional_energy,
        "forecast_window_summary.csv": forecast_windows,
    }


def build_scenarios(region_inputs: pd.DataFrame) -> pd.DataFrame:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_rows = []
    frames = []
    means = region_inputs.groupby("Region", as_index=False).agg(
        electricity_price_mean=("ElectricityPrice_CNY_per_MWh", "mean"),
        sell_price_mean=("SellPrice_CNY_per_MWh", "mean"),
    )
    base = region_inputs.merge(means, on="Region", validate="many_to_one")
    scenario_id = 0
    for carbon_ratio in config["carbon_budget_ratios"]:
        for price_factor in config["price_spread_factors"]:
            for renewable_factor in config["renewable_scale_factors"]:
                scenario_id += 1
                scenario_name = f"C{carbon_ratio:.2f}_P{price_factor:.2f}_R{renewable_factor:.2f}"
                scenario = base[["Hour", "Region"]].copy()
                scenario["scenario_id"] = scenario_name
                scenario["carbon_budget_ratio"] = carbon_ratio
                scenario["price_spread_factor"] = price_factor
                scenario["renewable_scale_factor"] = renewable_factor
                scenario["ElectricityPrice_CNY_per_MWh_scenario"] = np.maximum(
                    0.0,
                    base["electricity_price_mean"] + price_factor * (
                        base["ElectricityPrice_CNY_per_MWh"] - base["electricity_price_mean"]
                    ),
                )
                scenario["SellPrice_CNY_per_MWh_scenario"] = np.maximum(
                    0.0,
                    base["sell_price_mean"] + price_factor * (
                        base["SellPrice_CNY_per_MWh"] - base["sell_price_mean"]
                    ),
                )
                scenario["AvailableRenewable_MW_scenario"] = (
                    base["AvailableRenewable_MW"] * renewable_factor
                )
                frames.append(scenario)
                config_rows.append({
                    "scenario_id": scenario_name,
                    "carbon_budget_ratio": carbon_ratio,
                    "price_spread_factor": price_factor,
                    "renewable_scale_factor": renewable_factor,
                    "description": "Carbon budget ratio; price deviation from regional mean; renewable multiplier.",
                })
    write_csv(pd.DataFrame(config_rows), "scenario_config_effective.csv")
    return pd.concat(frames, ignore_index=True)


def write_data_dictionary(data: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame]) -> None:
    descriptions = {
        "dim_region.csv": "区域 GPU、IT/设施功率、PUE 和区域角色。",
        "dim_storage.csv": "区域储能参数、SOC 初值、充放电与购售电上限。",
        "dim_task_power.csv": "三类任务的等效 GPU IT 功率映射。",
        "dim_network_latency.csv": "源区域到目标区域的单向网络时延。",
        "fact_tasks.csv": "原始任务字段及分钟时长换算的派生字段。",
        "fact_region_hour_input.csv": "问题 2–4 的逐时输入参数及不可迁移 NonAI IT 负荷。",
        "fact_region_hour_baseline.csv": "附件提供的逐时基准状态与基准结果。",
        "task_hour_profile.csv": "任务在相对执行小时内的精确重叠、GPUh 与 AI IT 能量。",
        "task_region_eligibility.csv": "每项任务到各区域的单向时延及阈值可行标记。",
        "forecast_panel.csv": "问题 1 的 18 条小时序列及无未来信息的时序特征。",
        "scenario_region_hour.csv": "问题 4 的参数化价格/可再生能源情景数据。",
    }
    lines = ["# 处理后数据字典", "", "所有 CSV 使用 UTF-8 with BOM；原始附件不作修改。", ""]
    for file_name, description in descriptions.items():
        frame = derived.get(file_name, data.get(file_name.replace(".csv", "")))
        if frame is None:
            continue
        lines.extend([f"## `{file_name}`", "", description, "", "字段：", ""])
        fields = pd.DataFrame({"field": frame.columns, "dtype": frame.dtypes.astype(str).values})
        lines.append(markdown_table(fields, decimals=0))
        lines.append("")
    (PROCESSED_DIR / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")


def create_figures(tasks: pd.DataFrame, forecast: pd.DataFrame, region_time: pd.DataFrame) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; figures skipped"]

    plt.style.use("seaborn-v0_8-whitegrid")
    created: list[str] = []
    type_counts = tasks.groupby("TaskType")["TaskID"].size().sort_values(ascending=False)
    ax = type_counts.plot(kind="bar", figsize=(8, 4), color="#4472C4")
    ax.set(title="Task counts by task type", xlabel="Task type", ylabel="Task count")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "task_counts_by_type.png", dpi=180); plt.close()
    created.append("task_counts_by_type.png")

    arrivals = forecast.groupby(["Hour", "TaskType"], as_index=False)["task_count"].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    for task_type, group in arrivals.groupby("TaskType"):
        ax.plot(group["Hour"], group["task_count"], linewidth=0.6, label=task_type)
    ax.set(title="Hourly task arrivals by type", xlabel="Hour", ylabel="Task count")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(FIGURE_DIR / "hourly_task_arrivals.png", dpi=180); plt.close(fig)
    created.append("hourly_task_arrivals.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    tasks.groupby("SourceRegion")["gpu_hours"].sum().sort_values(ascending=False).plot(kind="bar", ax=ax, color="#70AD47")
    ax.set(title="Total task GPU-hours by source region", xlabel="Source region", ylabel="GPU-hours")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "gpu_hours_by_source_region.png", dpi=180); plt.close(fig)
    created.append("gpu_hours_by_source_region.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for region, group in region_time.groupby("Region"):
        ax.scatter(group["ElectricityPrice_CNY_per_MWh"], group["CarbonIntensity_tCO2_per_MWh"], s=3, alpha=0.18, label=region)
    ax.set(title="Electricity price and carbon intensity", xlabel="CNY/MWh", ylabel="tCO2/MWh")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(FIGURE_DIR / "price_carbon_scatter.png", dpi=180); plt.close(fig)
    created.append("price_carbon_scatter.png")
    return created


def write_reports(data: dict[str, pd.DataFrame], quality: pd.DataFrame, metadata: dict[str, Any],
                  summaries: dict[str, pd.DataFrame], figures: list[str]) -> None:
    for name, table in summaries.items():
        write_csv(table, name, TABLE_DIR)
    write_csv(quality, "data_quality_checks.csv", TABLE_DIR)

    lines = [
        "# C 题数据预处理与探索分析报告", "",
        "## 数据范围", "",
        f"- 任务记录：{metadata['task_rows']:,} 条；区域—小时记录：{metadata['region_hour_rows']:,} 条。",
        f"- 任务逐小时重叠剖面：{metadata['task_hour_profile_rows']:,} 条；任务—目标区域时延记录：{metadata['task_region_eligibility_rows']:,} 条。",
        "- 原始 Excel 仅作读取；基准字段与输入字段已分表保存。",
        "",
        "## 数据质量", "",
        f"- PASS：{metadata['quality_passes']} 项；FLAG：{metadata['quality_flags']} 项。",
        "- 质量校验仅标记附件数据与附件公式间的差异，不会修改源数据。",
        "",
        markdown_table(quality, decimals=6), "",
        "## 任务类型汇总", "", markdown_table(summaries["task_type_summary.csv"]), "",
        "## 任务来源区域汇总", "", markdown_table(summaries["task_source_region_summary.csv"]), "",
        "## 时延阈值可行区域数", "", markdown_table(summaries["latency_feasibility_by_type.csv"]), "",
        "## 区域逐时能源数据汇总", "", markdown_table(summaries["region_energy_summary.csv"]), "",
        "## 预测划分汇总", "", markdown_table(summaries["forecast_window_summary.csv"]), "",
        "## 图表", "",
    ]
    lines.extend([f"- `figures/{figure}`" for figure in figures])
    lines.extend([
        "", "## 口径说明", "",
        "- 问题 1 的预测面板按照题目给定的训练、验证和测试窗口标记。",
        "- `task_hour_profile.csv` 采用整数小时开工、分钟级时长与实际重叠量；执行实际时点为开工时段加相对小时。",
        "- 问题 2/4 应以任务调度重新形成 AI IT 负荷，并叠加 `NonAI_IT_Load_MW`；问题 3 使用附件给定的 `Baseline_AI_IT_Load_MW` 与 `NonAI_IT_Load_MW`。",
        "- `scenario_region_hour.csv` 为问题 4 参数化情景，不属于附件原始数据。",
    ])
    (REPORT_DIR / "data_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_output_dirs()
    manifest = collect_manifest()
    data = load_and_standardize()

    profile = build_task_hour_profile(data["fact_tasks"], data["dim_task_power"])
    eligibility = build_task_region_eligibility(data["fact_tasks"], data["dim_network_latency"])
    forecast = build_forecast_panel(data["fact_tasks"], data["dim_region"], data["dim_task_power"])
    region_inputs = data["region_time"][REGION_INPUT_COLUMNS].copy()
    region_baseline = data["region_time"][REGION_BASELINE_COLUMNS].copy()

    write_csv(data["dim_region"], "dim_region.csv")
    write_csv(data["dim_storage"], "dim_storage.csv")
    write_csv(data["dim_task_power"], "dim_task_power.csv")
    write_csv(data["dim_network_latency"], "dim_network_latency.csv")
    write_csv(data["fact_tasks"], "fact_tasks.csv")
    write_csv(region_inputs, "fact_region_hour_input.csv")
    write_csv(region_baseline, "fact_region_hour_baseline.csv")
    write_csv(profile, "task_hour_profile.csv")
    write_csv(eligibility, "task_region_eligibility.csv")
    write_csv(forecast, "forecast_panel.csv")

    scenarios = build_scenarios(region_inputs)
    write_csv(scenarios, "scenario_region_hour.csv")
    quality, metadata = check_quality(data, profile, eligibility)
    write_csv(quality, "data_quality_checks.csv")
    (PROCESSED_DIR / "manifest.json").write_text(
        json.dumps({**manifest, **metadata}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    derived = {
        "task_hour_profile.csv": profile,
        "task_region_eligibility.csv": eligibility,
        "forecast_panel.csv": forecast,
        "scenario_region_hour.csv": scenarios,
    }
    write_data_dictionary(data, derived)
    summaries = build_summary_tables(data, eligibility, forecast)
    figures = create_figures(data["fact_tasks"], forecast, data["region_time"])
    write_reports(data, quality, metadata, summaries, figures)

    print("Pipeline completed.")
    print(f"Processed data: {PROCESSED_DIR}")
    print(f"Reports: {REPORT_DIR}")
    print(f"Quality checks: {metadata['quality_passes']} pass, {metadata['quality_flags']} flag")


if __name__ == "__main__":
    main()
