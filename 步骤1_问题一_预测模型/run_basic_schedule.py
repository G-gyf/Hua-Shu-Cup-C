# 本程序及代码是在 AI 工具辅助下完成的。
# AI 工具名称：Codex，开发机构/公司：OpenAI。
"""问题一：末 24 小时实际任务的基础 GPU 调度。

模型只处理问题一的算力调度，不引入电价、碳排、新能源或储能目标。
所有任务固定在到达时刻启动，仅优化可行执行区域；网络时延作为 SLA 硬约束。
GPU、IT 功率和设施功率均按分钟级跨小时重叠精确计算。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack


STEP_DIR = Path(__file__).resolve().parent
ROOT = STEP_DIR.parent
OUT_DIR = STEP_DIR / "基础调度结果"
CONFIG = json.loads((STEP_DIR / "basic_schedule_config.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Inputs:
    tasks: pd.DataFrame
    profile: pd.DataFrame
    eligibility: pd.DataFrame
    regions: pd.DataFrame
    region_hour: pd.DataFrame


@dataclass(frozen=True)
class ModelData:
    candidates: pd.DataFrame
    matrix: csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    peak_objective: np.ndarray
    peak_index: int
    task_count: int
    resource_rows: list[tuple[str, int]]


def find_processed_dir() -> Path:
    candidates = [
        path / "processed"
        for path in ROOT.iterdir()
        if path.is_dir() and (path / "processed" / "fact_tasks.csv").is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError("Expected one Step 0 processed data directory.")
    return candidates[0]


def reset_output() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def md_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    view = frame.copy()
    for col in view.select_dtypes(include=["number"]).columns:
        view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:,.{decimals}f}")
    lines = [
        "| " + " | ".join(str(item).replace("|", "\\|") for item in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for row in view.astype(object).where(pd.notna(view), "").itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |")
    return "\n".join(lines)


def load_inputs() -> Inputs:
    processed = find_processed_dir()
    return Inputs(
        tasks=pd.read_csv(processed / "fact_tasks.csv", encoding="utf-8-sig"),
        profile=pd.read_csv(processed / "task_hour_profile.csv", encoding="utf-8-sig"),
        eligibility=pd.read_csv(processed / "task_region_eligibility.csv", encoding="utf-8-sig"),
        regions=pd.read_csv(processed / "dim_region.csv", encoding="utf-8-sig"),
        region_hour=pd.read_csv(processed / "fact_region_hour_input.csv", encoding="utf-8-sig"),
    )


def select_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    start = int(CONFIG["main_start_hour"])
    end = int(CONFIG["main_end_hour"])
    main = tasks.loc[tasks["ArrivalHour"].between(start, end)].copy()
    context = tasks.loc[(tasks["ArrivalHour"] < start) & (tasks["immediate_finish_hour"] > start)].copy()
    main["is_context_task"] = False
    context["is_context_task"] = True
    selected = pd.concat([context, main], ignore_index=True).sort_values(["is_context_task", "TaskID"]).reset_index(drop=True)
    if len(main) != 538 or len(context) != 58:
        raise ValueError(f"Unexpected Q1 window selection: main={len(main)}, context={len(context)}.")
    return selected


def candidate_starts(task: pd.Series, immediate_only: bool = False) -> range:
    settlement_end = int(CONFIG["settlement_end_hour"])
    if immediate_only or bool(task["is_context_task"]) or task["TaskType"] == "RealTimeInference":
        return range(int(task["ArrivalHour"]), int(task["ArrivalHour"]) + 1)
    latest = min(int(task["LatestFinishHour"]), settlement_end) - int(task["duration_ceil_h"])
    return range(int(task["EarliestStartHour"]), latest + 1)


def build_candidates(selected: pd.DataFrame, eligibility: pd.DataFrame, immediate_only: bool = False) -> pd.DataFrame:
    feasible = eligibility.loc[eligibility["is_latency_feasible"], ["TaskID", "ToRegion", "NetworkLatency_ms"]].copy()
    task_lookup = selected.set_index("TaskID")
    rows: list[dict[str, object]] = []
    for task_id, task in task_lookup.iterrows():
        starts = candidate_starts(task, immediate_only=immediate_only)
        for eligible in feasible.loc[feasible["TaskID"] == task_id].itertuples(index=False):
            for start_hour in starts:
                rows.append({
                    "TaskID": int(task_id),
                    "TaskType": task.TaskType,
                    "SourceRegion": task.SourceRegion,
                    "ExecuteRegion": eligible.ToRegion,
                    "StartHour": int(start_hour),
                    "EndHour": float(start_hour + task.duration_h),
                    "WaitHour": float(start_hour - task.ArrivalHour),
                    "NetworkLatency_ms": float(eligible.NetworkLatency_ms),
                    "GPU_Demand": int(task.GPU_Demand),
                    "is_context_task": bool(task.is_context_task),
                })
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No feasible scheduling candidates were generated.")
    candidate_counts = candidates.groupby("TaskID").size()
    missing = sorted(set(selected["TaskID"]) - set(candidate_counts.index))
    if missing:
        raise ValueError(f"Tasks without feasible candidates: {missing[:10]}")
    return candidates.reset_index(drop=True)


def build_profile_lookup(profile: pd.DataFrame, selected_task_ids: set[int]) -> dict[int, list[tuple[int, float, float]]]:
    subset = profile.loc[profile["TaskID"].isin(selected_task_ids), ["TaskID", "relative_hour", "gpu_hour", "ai_it_energy_mwh"]]
    return {
        int(task_id): [(int(row.relative_hour), float(row.gpu_hour), float(row.ai_it_energy_mwh)) for row in group.itertuples(index=False)]
        for task_id, group in subset.groupby("TaskID")
    }


def build_model(selected: pd.DataFrame, candidates: pd.DataFrame, inputs: Inputs) -> ModelData:
    start_hour = int(CONFIG["main_start_hour"])
    settlement_end = int(CONFIG["settlement_end_hour"])
    hours = list(range(start_hour, settlement_end))
    regions = inputs.regions.sort_values("Region").reset_index(drop=True)
    region_names = regions["Region"].tolist()
    task_ids = selected["TaskID"].astype(int).tolist()
    task_row = {task_id: index for index, task_id in enumerate(task_ids)}
    resource_rows = [(region, hour) for region in region_names for hour in hours]
    resource_row = {key: index for index, key in enumerate(resource_rows)}
    n_tasks, n_resources, n_candidates = len(task_ids), len(resource_rows), len(candidates)
    peak_index = n_candidates
    total_rows = n_tasks + 4 * n_resources

    region_lookup = regions.set_index("Region")
    hourly = inputs.region_hour.loc[
        inputs.region_hour["Hour"].between(start_hour, settlement_end - 1),
        ["Region", "Hour", "NonAI_IT_Load_MW"],
    ].copy()
    hourly_lookup = hourly.set_index(["Region", "Hour"])["NonAI_IT_Load_MW"].to_dict()
    if len(hourly_lookup) != n_resources:
        raise ValueError("Missing non-AI IT load for a region-hour in the Q1 horizon.")

    profile_lookup = build_profile_lookup(inputs.profile, set(task_ids))
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for column, candidate in candidates.iterrows():
        task_id = int(candidate.TaskID)
        rows.append(task_row[task_id]); cols.append(column); values.append(1.0)
        for relative_hour, gpu_hour, ai_energy in profile_lookup[task_id]:
            hour = int(candidate.StartHour) + relative_hour
            key = (candidate.ExecuteRegion, hour)
            if key not in resource_row:
                continue
            offset = resource_row[key]
            rows.extend([n_tasks + offset, n_tasks + n_resources + offset, n_tasks + 2 * n_resources + offset,
                         n_tasks + 3 * n_resources + offset])
            cols.extend([column, column, column, column])
            values.extend([
                gpu_hour,
                ai_energy,
                float(region_lookup.loc[candidate.ExecuteRegion, "PUE"]) * ai_energy,
                gpu_hour / float(region_lookup.loc[candidate.ExecuteRegion, "Available_GPU"]),
            ])
    # Umax column is -1 in the utilization linking rows.
    for offset in range(n_resources):
        rows.append(n_tasks + 3 * n_resources + offset)
        cols.append(peak_index)
        values.append(-1.0)

    matrix = coo_matrix((values, (rows, cols)), shape=(total_rows, n_candidates + 1)).tocsr()
    lower = np.full(total_rows, -np.inf, dtype=float)
    upper = np.full(total_rows, np.inf, dtype=float)
    lower[:n_tasks] = 1.0
    upper[:n_tasks] = 1.0
    for offset, (region, hour) in enumerate(resource_rows):
        non_ai = float(hourly_lookup[(region, hour)])
        params = region_lookup.loc[region]
        upper[n_tasks + offset] = float(params.Available_GPU)
        upper[n_tasks + n_resources + offset] = float(params.Max_IT_Power_MW) - non_ai
        upper[n_tasks + 2 * n_resources + offset] = float(params.Max_Facility_Power_MW) - float(params.PUE) * non_ai
        upper[n_tasks + 3 * n_resources + offset] = 0.0

    integrality = np.zeros(n_candidates + 1, dtype=int)
    integrality[:n_candidates] = 1
    variable_lower = np.zeros(n_candidates + 1, dtype=float)
    variable_upper = np.full(n_candidates + 1, np.inf, dtype=float)
    variable_upper[:n_candidates] = 1.0
    peak_objective = np.zeros(n_candidates + 1, dtype=float)
    peak_objective[peak_index] = 1.0
    return ModelData(
        candidates=candidates,
        matrix=matrix,
        lower=lower,
        upper=upper,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        peak_objective=peak_objective,
        peak_index=peak_index,
        task_count=n_tasks,
        resource_rows=resource_rows,
    )


def solve_stage(model: ModelData, objective: np.ndarray, extra: list[LinearConstraint], name: str):
    constraints = [LinearConstraint(model.matrix, model.lower, model.upper), *extra]
    result = milp(
        c=objective,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=constraints,
        options={
            "time_limit": float(CONFIG["mip_time_limit_seconds"]),
            "mip_rel_gap": float(CONFIG["mip_relative_gap"]),
            "disp": False,
        },
    )
    if result.x is None:
        raise RuntimeError(f"{name} did not produce a feasible solution: status={result.status}; {result.message}")
    return result


def solve_peak_utilization(model: ModelData):
    """Solve the sole Q1 optimization objective after fixed immediate starts."""
    return solve_stage(model, model.peak_objective, [], "Peak-utilization minimization")


def selected_schedule(model: ModelData, solution: np.ndarray, selected: pd.DataFrame) -> pd.DataFrame:
    selected_rows = model.candidates.loc[solution[:len(model.candidates)] > 0.5].copy()
    if selected_rows.groupby("TaskID").size().ne(1).any() or len(selected_rows) != len(selected):
        raise RuntimeError("The MIP solution does not select exactly one candidate per task.")
    metadata = selected[["TaskID", "ArrivalHour", "LatestFinishHour", "duration_h", "duration_ceil_h", "MaxLatency_ms"]]
    schedule = selected_rows.merge(metadata, on="TaskID", how="left", validate="one_to_one")
    schedule["FinishBoundaryHour"] = schedule["StartHour"] + schedule["duration_ceil_h"]
    schedule["gpu_hours"] = schedule["GPU_Demand"] * schedule["duration_h"]
    schedule["Migrated"] = schedule["SourceRegion"] != schedule["ExecuteRegion"]
    return schedule.sort_values(["ExecuteRegion", "StartHour", "TaskID"]).reset_index(drop=True)


def region_hour_schedule(schedule: pd.DataFrame, inputs: Inputs) -> pd.DataFrame:
    start_hour, settlement_end = int(CONFIG["main_start_hour"]), int(CONFIG["settlement_end_hour"])
    regions = inputs.regions.sort_values("Region").reset_index(drop=True)
    hours = pd.DataFrame({"Hour": range(start_hour, settlement_end)})
    panel = regions.assign(_key=1).merge(hours.assign(_key=1), on="_key").drop(columns="_key")
    panel = panel.merge(
        inputs.region_hour.loc[inputs.region_hour["Hour"].between(start_hour, settlement_end - 1),
                               ["Region", "Hour", "NonAI_IT_Load_MW"]],
        on=["Region", "Hour"], how="left", validate="one_to_one",
    )
    profile_lookup = build_profile_lookup(inputs.profile, set(schedule["TaskID"].astype(int)))
    load_rows: list[dict[str, object]] = []
    for task in schedule.itertuples(index=False):
        for relative_hour, gpu_hour, ai_energy in profile_lookup[int(task.TaskID)]:
            hour = int(task.StartHour) + relative_hour
            if start_hour <= hour < settlement_end:
                load_rows.append({"Region": task.ExecuteRegion, "Hour": hour, "GPU_Used": gpu_hour, "AI_IT_Load_MW": ai_energy})
    loads = pd.DataFrame(load_rows)
    if loads.empty:
        loads = pd.DataFrame(columns=["Region", "Hour", "GPU_Used", "AI_IT_Load_MW"])
    else:
        loads = loads.groupby(["Region", "Hour"], as_index=False)[["GPU_Used", "AI_IT_Load_MW"]].sum()
    panel = panel.merge(loads, on=["Region", "Hour"], how="left")
    panel[["GPU_Used", "AI_IT_Load_MW"]] = panel[["GPU_Used", "AI_IT_Load_MW"]].fillna(0.0)
    panel["GPU_Utilization"] = panel["GPU_Used"] / panel["Available_GPU"]
    panel["Total_IT_Load_MW"] = panel["NonAI_IT_Load_MW"] + panel["AI_IT_Load_MW"]
    panel["Facility_Load_MW"] = panel["Total_IT_Load_MW"] * panel["PUE"]
    panel["GPU_Headroom"] = panel["Available_GPU"] - panel["GPU_Used"]
    panel["IT_Headroom_MW"] = panel["Max_IT_Power_MW"] - panel["Total_IT_Load_MW"]
    panel["Facility_Headroom_MW"] = panel["Max_Facility_Power_MW"] - panel["Facility_Load_MW"]
    return panel.sort_values(["Region", "Hour"]).reset_index(drop=True)


def make_source_baseline(selected: pd.DataFrame) -> pd.DataFrame:
    """Reference only: no migration and immediate start, assessed but not asserted feasible."""
    baseline = selected.copy()
    baseline = baseline.rename(columns={"SourceRegion": "ExecuteRegion"})
    baseline["StartHour"] = baseline["ArrivalHour"]
    baseline["EndHour"] = baseline["StartHour"] + baseline["duration_h"]
    baseline["WaitHour"] = 0.0
    baseline["NetworkLatency_ms"] = 0.0
    baseline["GPU_Demand"] = baseline["GPU_Demand"].astype(int)
    baseline["Migrated"] = False
    return baseline[["TaskID", "TaskType", "ExecuteRegion", "StartHour", "EndHour", "WaitHour", "NetworkLatency_ms", "GPU_Demand", "is_context_task", "Migrated"]]


def constraint_checks(schedule: pd.DataFrame, panel: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    tolerance = 1e-7
    realtime = schedule.loc[schedule["TaskType"] == "RealTimeInference"]
    selected_profile = profile.loc[profile["TaskID"].isin(schedule["TaskID"]), ["TaskID", "overlap_h"]]
    profile_hours = selected_profile.groupby("TaskID")["overlap_h"].sum()
    declared_hours = schedule.set_index("TaskID")["duration_h"]
    duration_error = float((profile_hours - declared_hours).abs().max())
    checks = [
        ("Task one-candidate selection", 0.0, len(schedule), "Each selected/context task has one scheduling record."),
        ("Real-time immediate start", float((realtime["StartHour"] - realtime["ArrivalHour"]).abs().max()) if len(realtime) else 0.0, len(realtime), "Real-time start must equal arrival."),
        ("Network latency", float((schedule["NetworkLatency_ms"] - schedule["MaxLatency_ms"]).max()), len(schedule), "Maximum value is latency minus task limit."),
        ("Latest finish hour", float((schedule["FinishBoundaryHour"] - schedule[["LatestFinishHour"]].iloc[:, 0]).max()), len(schedule), "Maximum value is completion boundary minus task deadline."),
        ("Settlement boundary", float((schedule["FinishBoundaryHour"] - int(CONFIG["settlement_end_hour"])).max()), len(schedule), "All tasks must finish by t=2406."),
        ("Minute-level duration conservation", duration_error, len(schedule), "Profile overlap hours must equal every task duration."),
        ("GPU capacity", float((-panel["GPU_Headroom"]).max()), len(panel), "Maximum value is GPU usage minus available GPU."),
        ("IT power capacity", float((-panel["IT_Headroom_MW"]).max()), len(panel), "Maximum value is IT load minus IT limit."),
        ("Facility power capacity", float((-panel["Facility_Headroom_MW"]).max()), len(panel), "Maximum value is facility load minus facility limit."),
    ]
    result = pd.DataFrame(checks, columns=["check", "maximum_violation", "records_checked", "detail"])
    result["status"] = np.where(result["maximum_violation"] <= tolerance, "PASS", "FAIL")
    return result[["check", "status", "maximum_violation", "records_checked", "detail"]]


def summary_table(schedule: pd.DataFrame, panel: pd.DataFrame, baseline_panel: pd.DataFrame, result) -> pd.DataFrame:
    main = schedule.loc[~schedule["is_context_task"]].copy()
    baseline_peak = float(baseline_panel["GPU_Utilization"].max())
    values = [
        ("main_task_count", float(len(main))),
        ("context_task_count", float(schedule["is_context_task"].sum())),
        ("migrated_main_task_count", float(main["Migrated"].sum())),
        ("main_migration_rate", float(main["Migrated"].mean())),
        ("main_average_wait_hour", float(main["WaitHour"].mean())),
        ("main_average_latency_ms", float(main["NetworkLatency_ms"].mean())),
        ("maximum_gpu_utilization", float(panel["GPU_Utilization"].max())),
        ("source_immediate_baseline_maximum_gpu_utilization", baseline_peak),
        ("source_immediate_baseline_feasible", float(
            (baseline_panel["GPU_Headroom"] >= -1e-7).all()
            and (baseline_panel["IT_Headroom_MW"] >= -1e-7).all()
            and (baseline_panel["Facility_Headroom_MW"] >= -1e-7).all()
        )),
        ("peak_solver_status", float(result.status)),
        ("peak_mip_gap", float(getattr(result, "mip_gap", np.nan))),
    ]
    return pd.DataFrame(values, columns=["metric", "value"])


def dispatch_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    main = schedule.loc[~schedule["is_context_task"]].copy()
    return main.groupby(["TaskType", "ExecuteRegion"], as_index=False).agg(
        task_count=("TaskID", "size"),
        gpu_demand_sum=("GPU_Demand", "sum"),
        gpu_hours=("gpu_hours", "sum"),
        average_wait_hour=("WaitHour", "mean"),
        average_latency_ms=("NetworkLatency_ms", "mean"),
        migrated_task_count=("Migrated", "sum"),
    ).sort_values(["TaskType", "ExecuteRegion"])


def plot_results(schedule: pd.DataFrame, panel: pd.DataFrame) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return []
    figure_names: list[str] = []
    colors = {"AITraining": "#4472C4", "BatchInference": "#70AD47", "RealTimeInference": "#ED7D31"}
    start, end = int(CONFIG["main_start_hour"]), int(CONFIG["settlement_end_hour"])
    active = schedule.loc[(schedule["EndHour"] > start) & (schedule["StartHour"] < end)].copy()
    regions = sorted(panel["Region"].unique())
    fig, axes = plt.subplots(len(regions), 1, figsize=(14, 3.5 * len(regions)), sharex=True)
    for ax, region in zip(axes, regions):
        jobs = active.loc[active["ExecuteRegion"] == region].sort_values(["StartHour", "EndHour", "TaskID"]).reset_index(drop=True)
        for y, job in jobs.iterrows():
            left = max(float(job.StartHour), start)
            width = min(float(job.EndHour), end) - left
            ax.broken_barh([(left, width)], (y - 0.4, 0.8), facecolors=colors[job.TaskType], alpha=0.88)
        ax.axvspan(2400, end, color="#D9E2F3", alpha=0.35)
        ax.set_title(region)
        ax.set_ylabel("Tasks")
        ax.set_ylim(-1, max(1, len(jobs)))
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Hour")
    axes[0].legend(handles=[Patch(color=color, label=kind) for kind, color in colors.items()], loc="upper right")
    fig.suptitle("Question 1 basic schedule: active tasks in hours 2376-2405", y=0.995)
    fig.tight_layout()
    gantt = OUT_DIR / "figures" / "gantt_last24h.png"
    fig.savefig(gantt, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figure_names.append(gantt.name)

    fig, ax = plt.subplots(figsize=(14, 6))
    for region, group in panel.groupby("Region"):
        ax.plot(group["Hour"], 100 * group["GPU_Utilization"], marker="o", linewidth=1.6, markersize=3, label=region)
    ax.axvspan(2400, end, color="#D9E2F3", alpha=0.35, label="Settlement period")
    ax.set(title="GPU utilization by region", xlabel="Hour", ylabel="GPU utilization (%)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    utilization = OUT_DIR / "figures" / "gpu_utilization_by_region.png"
    fig.savefig(utilization, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figure_names.append(utilization.name)
    return figure_names


def write_report(schedule: pd.DataFrame, panel: pd.DataFrame, checks: pd.DataFrame, summary: pd.DataFrame,
                 dispatch: pd.DataFrame, model: ModelData, result, figures: list[str], candidate_mode: str) -> None:
    status = "optimal" if result.status == 0 else "feasible_near_optimal"
    peak_value = float(result.x[model.peak_index])
    report = [
        "# 问题一末 24 小时基础调度报告", "",
        f"- 求解状态：**{status}**。",
        f"- 主任务：538 条（到达小时 2376–2399）；边界携入任务：58 条。",
        f"- 候选二元变量：{len(model.candidates):,}；资源约束区域—小时数：{len(model.resource_rows):,}。",
        f"- 候选列策略：`{candidate_mode}`。",
        f"- 固定即时开工；最大 GPU 利用率：{peak_value:.6%}。",
        f"- 峰值 MIP gap：{getattr(result, 'mip_gap', np.nan):.6g}；状态码：{result.status}。",
        "",
        "## 调度目标", "",
        "所有任务在到达时刻启动；在满足单向网络时延 SLA 及资源约束的可行执行区域中，最小化最大 GPU 利用率。网络时延作为硬约束和结果评价指标报告。",
        "",
        "## 汇总指标", "", md_table(summary), "",
        "## 按任务类型与执行区域汇总", "", md_table(dispatch), "",
        "## 约束检查", "", md_table(checks), "",
        "## 图表", "",
    ]
    report.extend([f"- `figures/{name}`" for name in figures] or ["- 未生成：缺少 matplotlib。"])
    report.extend([
        "", "## 口径", "",
        "- GPU 和 AI IT 负荷按 `task_hour_profile.csv` 的分钟级跨小时重叠计算。",
        "- IT/设施功率约束均叠加实际 `NonAI_IT_Load_MW`。",
        "- 当前数据中所有任务到达即启动可行，因此固定开工时刻；若迁移到其他数据后即时开工不可行，应恢复 Batch/Training 的延后开工建模。",
        "- 来源区立即执行基线仅作负载比较；若其 `source_immediate_baseline_feasible=0`，则不应作为可行调度方案。",
        "- 2400–2405 为结算期；没有任务允许占用第 2406 小时。",
    ])
    (OUT_DIR / "q1_schedule_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    reset_output()
    inputs = load_inputs()
    selected = select_tasks(inputs.tasks)
    candidate_mode = "immediate_start_fixed"
    candidates = build_candidates(selected, inputs.eligibility, immediate_only=True)
    model = build_model(selected, candidates, inputs)
    result = solve_peak_utilization(model)
    schedule = selected_schedule(model, result.x, selected)
    panel = region_hour_schedule(schedule, inputs)
    baseline = make_source_baseline(selected)
    baseline_panel = region_hour_schedule(baseline, inputs)
    checks = constraint_checks(schedule, panel, inputs.profile)
    if not (checks["status"] == "PASS").all():
        raise RuntimeError("Constraint validation failed after optimization. See generated checks for details.")
    summary = summary_table(schedule, panel, baseline_panel, result)
    dispatch = dispatch_summary(schedule)
    figures = plot_results(schedule, panel)
    write_csv(schedule, "task_schedule.csv")
    write_csv(panel, "region_hour_schedule.csv")
    write_csv(checks, "constraint_checks.csv")
    write_csv(summary, "schedule_summary.csv")
    write_csv(dispatch, "task_type_destination_summary.csv")
    write_report(schedule, panel, checks, summary, dispatch, model, result, figures, candidate_mode)
    print(f"Q1 basic schedule completed: {OUT_DIR}")
    print(f"Tasks: main={int((~schedule['is_context_task']).sum())}, context={int(schedule['is_context_task'].sum())}")
    print(f"Candidates: {len(candidates):,}; mode={candidate_mode}; max GPU utilization: {panel['GPU_Utilization'].max():.4%}")


if __name__ == "__main__":
    main()
