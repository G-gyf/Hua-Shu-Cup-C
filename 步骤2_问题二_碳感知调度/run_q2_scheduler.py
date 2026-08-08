# 本程序及代码是在 AI 工具辅助下完成的。
# AI 工具名称：Codex，开发机构/公司：OpenAI。
"""问题二：无储能条件下的碳感知任务迁移与开工调度。

实现采用两层可扩展框架：首先根据逐时新能源、价格、碳强度和非 AI 负荷
构造区域—时段能源机会目标；随后以任务级非抢占剖面进行候选分配和容量修复。
最终所有指标均由任务级排程重新计算，不以聚合指导层直接作为结果。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


STEP_DIR = Path(__file__).resolve().parent
ROOT = STEP_DIR.parent
OUT_DIR = STEP_DIR / "调度结果"
CONFIG = json.loads((STEP_DIR / "q2_config.json").read_text(encoding="utf-8"))
EPS = 1e-8


@dataclass
class Inputs:
    tasks: pd.DataFrame
    profile: pd.DataFrame
    eligibility: pd.DataFrame
    regions: pd.DataFrame
    storage: pd.DataFrame
    hourly: pd.DataFrame


def find_processed_dir() -> Path:
    candidates = [p / "processed" for p in ROOT.iterdir() if p.is_dir() and (p / "processed" / "fact_tasks.csv").is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError("Expected exactly one Step 0 processed data directory.")
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
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.{decimals}f}")
    lines = ["| " + " | ".join(map(str, view.columns)) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for row in view.astype(object).where(pd.notna(view), "").itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(lines)


def load_inputs() -> Inputs:
    p = find_processed_dir()
    return Inputs(
        tasks=pd.read_csv(p / "fact_tasks.csv", encoding="utf-8-sig"),
        profile=pd.read_csv(p / "task_hour_profile.csv", encoding="utf-8-sig"),
        eligibility=pd.read_csv(p / "task_region_eligibility.csv", encoding="utf-8-sig"),
        regions=pd.read_csv(p / "dim_region.csv", encoding="utf-8-sig"),
        storage=pd.read_csv(p / "dim_storage.csv", encoding="utf-8-sig"),
        hourly=pd.read_csv(p / "fact_region_hour_input.csv", encoding="utf-8-sig"),
    )


def profile_lookup(profile: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        int(tid): (g.relative_hour.to_numpy(dtype=int), g.gpu_hour.to_numpy(dtype=float), g.ai_it_energy_mwh.to_numpy(dtype=float))
        for tid, g in profile[["TaskID", "relative_hour", "gpu_hour", "ai_it_energy_mwh"]].groupby("TaskID")
    }


def make_context(inputs: Inputs):
    end = int(CONFIG["settlement_end_hour"])
    regions = inputs.regions.sort_values("Region").reset_index(drop=True)
    names = regions.Region.tolist()
    r_index = {r: i for i, r in enumerate(names)}
    h = np.arange(end, dtype=int)
    hourly = inputs.hourly.loc[inputs.hourly.Hour.between(0, end - 1)].copy()
    hourly = hourly.merge(regions[["Region", "PUE", "Available_GPU", "Max_IT_Power_MW", "Max_Facility_Power_MW"]], on="Region", how="left")
    hourly = hourly.merge(inputs.storage[["Region", "MaxGridImport_MW", "MaxGridExport_MW"]], on="Region", how="left")
    hourly = hourly.sort_values(["Region", "Hour"])
    def array(col: str) -> np.ndarray:
        return hourly.pivot(index="Region", columns="Hour", values=col).reindex(index=names, columns=h).to_numpy(dtype=float)
    return names, r_index, {
        "pue": regions.set_index("Region").loc[names, "PUE"].to_numpy(float),
        "gpu_cap": array("Available_GPU"), "it_cap": array("Max_IT_Power_MW"),
        "facility_cap": array("Max_Facility_Power_MW"), "nonai": array("NonAI_IT_Load_MW"),
        "renew": array("AvailableRenewable_MW"), "price": array("ElectricityPrice_CNY_per_MWh"),
        "sell_price": array("SellPrice_CNY_per_MWh"), "carbon": array("CarbonIntensity_tCO2_per_MWh"),
        "import_cap": array("MaxGridImport_MW"), "export_cap": array("MaxGridExport_MW"),
    }


def feasible_regions(inputs: Inputs) -> dict[int, list[tuple[str, float]]]:
    rows = inputs.eligibility.loc[inputs.eligibility.is_latency_feasible, ["TaskID", "ToRegion", "NetworkLatency_ms"]]
    return {int(tid): [(x.ToRegion, float(x.NetworkLatency_ms)) for x in g.itertuples(index=False)] for tid, g in rows.groupby("TaskID")}


def energy_dispatch(ai: np.ndarray, ctx: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    load = (ctx["nonai"] + ai) * ctx["pue"][:, None]
    direct = np.minimum(load, ctx["renew"])
    grid = np.maximum(load - ctx["renew"], 0.0)
    surplus = np.maximum(ctx["renew"] - direct, 0.0)
    sell = np.minimum(surplus, ctx["export_cap"])
    curtail = surplus - sell
    return {"AI_IT_Load_MW": ai, "Total_IT_Load_MW": ctx["nonai"] + ai, "Facility_Load_MW": load,
            "UsedRenewable_MW": direct, "GridPurchase_MW": grid, "GridSell_MW": sell, "Curtailment_MW": curtail}


def metrics(ai: np.ndarray, ctx: dict[str, np.ndarray]) -> dict[str, float]:
    e = energy_dispatch(ai, ctx)
    return {
        "Cost_CNY": float((e["GridPurchase_MW"] * ctx["price"] - e["GridSell_MW"] * ctx["sell_price"]).sum()),
        "Carbon_tCO2": float((e["GridPurchase_MW"] * ctx["carbon"]).sum()),
        "RenewableUtilization": float((e["UsedRenewable_MW"] + e["GridSell_MW"]).sum() / ctx["renew"].sum()),
        "GridPurchase_MWh": float(e["GridPurchase_MW"].sum()),
        "Curtailment_MWh": float(e["Curtailment_MW"].sum()),
    }


def epsilon_targets(reference: dict[str, float], carbon_ratio: float, renewable_fraction: float) -> tuple[float, float]:
    """Convert the normalized epsilon grid to auditable, absolute targets.

    Carbon is bounded as a ratio of the source-immediate reference.  Renewable
    utilization is bounded on the interval from the reference level to the
    physical upper bound of one.  A point that fails either final-schedule test
    is reported, but is never called a feasible Pareto solution.
    """
    carbon_budget = float(carbon_ratio * reference["Carbon_tCO2"])
    renewable_floor = float(reference["RenewableUtilization"] + renewable_fraction * (1.0 - reference["RenewableUtilization"]))
    return carbon_budget, renewable_floor


def candidate_energy_delta(ai: np.ndarray, region: int, start: int, prof: tuple[np.ndarray, np.ndarray, np.ndarray], ctx: dict[str, np.ndarray]):
    rel, _, ai_mw = prof; t = start + rel
    if int(t[-1]) >= ai.shape[1]: return None
    old_load = (ctx["nonai"][region, t] + ai[region, t]) * ctx["pue"][region]
    new_load = old_load + ai_mw * ctx["pue"][region]
    renew, export = ctx["renew"][region, t], ctx["export_cap"][region, t]
    def values(load):
        direct = np.minimum(load, renew); grid = np.maximum(load - renew, 0.0); sell = np.minimum(np.maximum(renew - direct, 0.0), export)
        return (grid * ctx["price"][region, t] - sell * ctx["sell_price"][region, t]).sum(), (grid * ctx["carbon"][region, t]).sum(), (direct + sell).sum()
    bc, bcar, breu = values(old_load); ac, acar, areu = values(new_load)
    return float(ac - bc), float(acar - bcar), float(areu - breu)


def is_feasible(ai: np.ndarray, gpu: np.ndarray, region: int, start: int, prof: tuple[np.ndarray, np.ndarray, np.ndarray], ctx: dict[str, np.ndarray]) -> bool:
    rel, gpu_h, ai_mw = prof; t = start + rel
    if int(t[-1]) >= ai.shape[1]: return False
    total_it = ctx["nonai"][region, t] + ai[region, t] + ai_mw
    return bool(np.all(gpu[region, t] + gpu_h <= ctx["gpu_cap"][region, t] + EPS)
                and np.all(total_it <= ctx["it_cap"][region, t] + EPS)
                and np.all(total_it * ctx["pue"][region] <= ctx["facility_cap"][region, t] + EPS))


def apply(ai: np.ndarray, gpu: np.ndarray, region: int, start: int, prof: tuple[np.ndarray, np.ndarray, np.ndarray], sign: float = 1.0) -> None:
    rel, gpu_h, ai_mw = prof; t = start + rel
    gpu[region, t] += sign * gpu_h
    ai[region, t] += sign * ai_mw


def master_priority(ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Continuous energy-opportunity proxy used to guide the task-level repair layer."""
    load = ctx["nonai"] * ctx["pue"][:, None]
    surplus = ctx["renew"] - load
    foregone_sale = np.where((surplus > 0) & (ctx["export_cap"] > 0), ctx["sell_price"], 0.0)
    grid_cost = np.where(surplus <= 0, ctx["price"] + CONFIG["carbon_shadow_price_cny_per_tco2"] * ctx["carbon"], 0.0)
    curtail_bonus = np.where((surplus > 0) & (ctx["export_cap"] <= 0), CONFIG["renewable_shadow_price_cny_per_mwh"], 0.0)
    return foregone_sale + grid_cost - curtail_bonus


def start_candidates(task, region: str, target: np.ndarray, r: int) -> list[int]:
    end = int(CONFIG["settlement_end_hour"])
    earliest = int(task.EarliestStartHour)
    latest = min(int(task.LatestFinishHour), end) - int(task.duration_ceil_h)
    if task.TaskType == "RealTimeInference":
        return [int(task.ArrivalHour)]
    starts = {earliest, latest}
    # Five evenly spaced service-time alternatives prevent a full 2,400-hour
    # enumeration while preserving early, middle and deadline-near choices.
    starts.update(int(round(earliest + q * (latest - earliest))) for q in (1 / 3, 2 / 3))
    ranked = np.argsort(target[r, earliest:latest + 1])[::-1] + earliest
    starts.update(int(x) for x in ranked[:int(CONFIG["candidate_start_limit_per_region"])])
    return sorted(s for s in starts if earliest <= s <= latest)


def schedule_one(tasks: pd.DataFrame, profiles, eligibility, names, r_index, ctx,
                 carbon_ratio: float, reu_fraction: float, carbon_budget: float):
    """Improve a feasible source-immediate schedule by task moves.

    Starting from the reference makes service and capacity feasibility explicit.
    Whenever the reference already meets the requested carbon budget, every
    accepted move is additionally screened against that budget, so the final
    schedule preserves it by construction.  For budgets below the reference,
    the same loop becomes a monotone carbon-improvement search; final epsilon
    feasibility remains the authoritative acceptance test.
    """
    h = int(CONFIG["settlement_end_hour"])
    reference = reference_schedule(tasks, profiles, names, r_index)
    ai, gpu = rebuild_load(reference, profiles, names, r_index, h)
    current_carbon = metrics(ai, ctx)["Carbon_tCO2"]
    budget_preserved = current_carbon <= carbon_budget + EPS
    target = 1.0 / (1.0 + np.maximum(master_priority(ctx), -500.0))
    target = np.maximum(target, 0.0)
    # Preserve service urgency: realtime first, then batch, then training; within class arrival order.
    order = {"RealTimeInference": 0, "BatchInference": 1, "AITraining": 2}
    records = {int(x.TaskID): dict(x._asdict()) for x in reference.itertuples(index=False)}
    for task in tasks.sort_values("ArrivalHour").sort_values("TaskType", key=lambda s: s.map(order), kind="stable").itertuples(index=False):
        prof = profiles[int(task.TaskID)]
        old = records[int(task.TaskID)]
        old_region = r_index[old["ExecuteRegion"]]
        old_start = int(old["StartHour"])
        apply(ai, gpu, old_region, old_start, prof, sign=-1.0)
        old_delta = candidate_energy_delta(ai, old_region, old_start, prof, ctx)
        if old_delta is None:
            raise RuntimeError(f"Cannot remove reference placement for task {task.TaskID}.")
        _, old_carbon, _ = old_delta
        carbon_after_removal = current_carbon - old_carbon
        choices = []
        for region, latency in eligibility[int(task.TaskID)]:
            r = r_index[region]
            for start in start_candidates(task, region, target, r):
                if not is_feasible(ai, gpu, r, start, prof, ctx):
                    continue
                delta = candidate_energy_delta(ai, r, start, prof, ctx)
                if delta is None:
                    continue
                dc, dcar, dreu = delta
                candidate_carbon = carbon_after_removal + dcar
                if budget_preserved and candidate_carbon > carbon_budget + EPS:
                    continue
                # If the requested budget is already tighter than the reference,
                # never undo an achieved carbon reduction during the search.
                if not budget_preserved and dcar > old_carbon + EPS:
                    continue
                wait = start - int(task.ArrivalHour)
                wait_weight = CONFIG["batch_wait_weight"] if task.TaskType == "BatchInference" else CONFIG["training_wait_weight"] if task.TaskType == "AITraining" else 0.0
                # Penalty coefficients guide the construction toward the requested
                # epsilon direction.  Hard epsilon feasibility is evaluated from
                # the completed, task-level schedule in main().
                score = dc + CONFIG["carbon_shadow_price_cny_per_tco2"] * (1.0 + 4.0 * (1.0 - carbon_ratio)) * dcar
                score -= CONFIG["renewable_shadow_price_cny_per_mwh"] * reu_fraction * dreu
                score += CONFIG["wait_shadow_price_cny_per_hour"] * wait_weight * wait
                score += CONFIG["latency_shadow_price_cny_per_ms"] * latency
                rel, _, ai_mw = prof
                target_gain = float((target[r, start + rel] * ai_mw).sum())
                choices.append((score - 0.01 * target_gain, region, r, start, latency, dc, dcar, dreu))
        if not choices:
            # The reference placement is a valid fallback unless external input
            # data themselves contradict the task or capacity requirements.
            if is_feasible(ai, gpu, old_region, old_start, prof, ctx):
                fallback = candidate_energy_delta(ai, old_region, old_start, prof, ctx)
                choices = [(0.0, old["ExecuteRegion"], old_region, old_start, old["NetworkLatency_ms"], *fallback)]
            else:
                raise RuntimeError(f"No feasible candidate found for task {task.TaskID}.")
        _, region, r, start, latency, dc, dcar, dreu = min(choices, key=lambda x: x[0])
        apply(ai, gpu, r, start, prof)
        current_carbon = carbon_after_removal + dcar
        records[int(task.TaskID)] = {"TaskID": int(task.TaskID), "TaskType": task.TaskType, "SourceRegion": task.SourceRegion,
                                     "ExecuteRegion": region, "ArrivalHour": int(task.ArrivalHour), "StartHour": start,
                                     "EndHour": start + float(task.duration_h), "FinishBoundaryHour": start + int(task.duration_ceil_h),
                                     "WaitHour": start - int(task.ArrivalHour), "NetworkLatency_ms": latency,
                                     "MaxLatency_ms": int(task.MaxLatency_ms), "LatestFinishHour": int(task.LatestFinishHour),
                                     "duration_h": float(task.duration_h), "GPU_Demand": int(task.GPU_Demand), "Migrated": task.SourceRegion != region,
                                     "marginal_cost_cny": dc, "marginal_carbon_tco2": dcar, "marginal_renewable_mwh": dreu}
    return pd.DataFrame(records.values()), ai, gpu


def construct_feasible_schedule(tasks: pd.DataFrame, profiles, eligibility, names, r_index, ctx,
                                carbon_ratio: float, reu_fraction: float):
    """Build a schedule from an empty load state, enforcing every hard capacity.

    This is the robust construction layer used when the source-immediate
    comparator is itself capacity-infeasible.  The two scenario parameters
    steer its energy score; final epsilon checks are applied by main().
    """
    h = int(CONFIG["settlement_end_hour"])
    ai = np.zeros((len(names), h)); gpu = np.zeros_like(ai)
    target = 1.0 / (1.0 + np.maximum(master_priority(ctx), -500.0))
    target = np.maximum(target, 0.0)
    order = {"RealTimeInference": 0, "BatchInference": 1, "AITraining": 2}
    records = []
    ordered = tasks.sort_values("ArrivalHour").sort_values("TaskType", key=lambda s: s.map(order), kind="stable")
    for task in ordered.itertuples(index=False):
        prof = profiles[int(task.TaskID)]
        choices = []
        for region, latency in eligibility[int(task.TaskID)]:
            r = r_index[region]
            for start in start_candidates(task, region, target, r):
                if not is_feasible(ai, gpu, r, start, prof, ctx):
                    continue
                delta = candidate_energy_delta(ai, r, start, prof, ctx)
                if delta is None:
                    continue
                dc, dcar, dreu = delta
                wait = start - int(task.ArrivalHour)
                wait_weight = CONFIG["batch_wait_weight"] if task.TaskType == "BatchInference" else CONFIG["training_wait_weight"] if task.TaskType == "AITraining" else 0.0
                score = dc + CONFIG["carbon_shadow_price_cny_per_tco2"] * (1.0 + 4.0 * (1.0 - carbon_ratio)) * dcar
                score -= CONFIG["renewable_shadow_price_cny_per_mwh"] * reu_fraction * dreu
                score += CONFIG["wait_shadow_price_cny_per_hour"] * wait_weight * wait
                score += CONFIG["latency_shadow_price_cny_per_ms"] * latency
                rel, _, ai_mw = prof
                target_gain = float((target[r, start + rel] * ai_mw).sum())
                choices.append((score - 0.01 * target_gain, region, r, start, latency, dc, dcar, dreu))
        if not choices:
            raise RuntimeError(f"No capacity-feasible candidate found for task {task.TaskID}.")
        _, region, r, start, latency, dc, dcar, dreu = min(choices, key=lambda x: x[0])
        apply(ai, gpu, r, start, prof)
        records.append({"TaskID": int(task.TaskID), "TaskType": task.TaskType, "SourceRegion": task.SourceRegion,
                        "ExecuteRegion": region, "ArrivalHour": int(task.ArrivalHour), "StartHour": start,
                        "EndHour": start + float(task.duration_h), "FinishBoundaryHour": start + int(task.duration_ceil_h),
                        "WaitHour": start - int(task.ArrivalHour), "NetworkLatency_ms": latency,
                        "MaxLatency_ms": int(task.MaxLatency_ms), "LatestFinishHour": int(task.LatestFinishHour),
                        "duration_h": float(task.duration_h), "GPU_Demand": int(task.GPU_Demand), "Migrated": task.SourceRegion != region,
                        "marginal_cost_cny": dc, "marginal_carbon_tco2": dcar, "marginal_renewable_mwh": dreu})
    return pd.DataFrame(records), ai, gpu


def reference_schedule(tasks, profiles, names, r_index):
    rows = []
    for task in tasks.itertuples(index=False):
        rows.append({"TaskID": int(task.TaskID), "TaskType": task.TaskType, "SourceRegion": task.SourceRegion,
                     "ExecuteRegion": task.SourceRegion, "ArrivalHour": int(task.ArrivalHour), "StartHour": int(task.ArrivalHour),
                     "EndHour": float(task.ArrivalHour + task.duration_h), "FinishBoundaryHour": int(task.ArrivalHour + task.duration_ceil_h),
                     "WaitHour": 0.0, "NetworkLatency_ms": 0.0, "MaxLatency_ms": int(task.MaxLatency_ms),
                     "LatestFinishHour": int(task.LatestFinishHour), "duration_h": float(task.duration_h),
                     "GPU_Demand": int(task.GPU_Demand), "Migrated": False})
    return pd.DataFrame(rows)


def rebuild_load(schedule, profiles, names, r_index, h):
    ai = np.zeros((len(names), h)); gpu = np.zeros_like(ai)
    for x in schedule.itertuples(index=False):
        apply(ai, gpu, r_index[x.ExecuteRegion], int(x.StartHour), profiles[int(x.TaskID)])
    return ai, gpu


def region_hour_frame(ai, gpu, names, ctx):
    e = energy_dispatch(ai, ctx)
    rows = []
    for r, name in enumerate(names):
        for t in range(ai.shape[1]):
            rows.append({"Region": name, "Hour": t, "GPU_Used": gpu[r, t], "Available_GPU": ctx["gpu_cap"][r, t],
                         "GPU_Utilization": gpu[r, t] / ctx["gpu_cap"][r, t], "NonAI_IT_Load_MW": ctx["nonai"][r, t],
                         "AI_IT_Load_MW": ai[r, t], **{k: float(v[r, t]) for k, v in e.items()}})
    return pd.DataFrame(rows)


def checks(schedule, region_hour, ctx):
    violations = pd.DataFrame({
        "GPU capacity": region_hour.GPU_Used - region_hour.Available_GPU,
        "IT power capacity": region_hour.Total_IT_Load_MW - np.repeat(ctx["it_cap"], 1, axis=0).ravel(),
        "Facility power capacity": region_hour.Facility_Load_MW - np.repeat(ctx["facility_cap"], 1, axis=0).ravel(),
        "Grid import capacity": region_hour.GridPurchase_MW - np.repeat(ctx["import_cap"], 1, axis=0).ravel(),
        "Grid export capacity": region_hour.GridSell_MW - np.repeat(ctx["export_cap"], 1, axis=0).ravel(),
    })
    output = []
    for key in violations:
        output.append((key, float(violations[key].max()), len(region_hour)))
    output.extend([
        ("Task one assignment", 0.0 if schedule.TaskID.nunique() == len(schedule) else 1.0, len(schedule)),
        ("Real-time immediate start", float((schedule.loc[schedule.TaskType == "RealTimeInference", "StartHour"] - schedule.loc[schedule.TaskType == "RealTimeInference", "ArrivalHour"]).abs().max()), int((schedule.TaskType == "RealTimeInference").sum())),
        ("Network latency", float((schedule.NetworkLatency_ms - schedule.MaxLatency_ms).max()), len(schedule)),
        ("Latest finish hour", float((schedule.FinishBoundaryHour - schedule.LatestFinishHour).max()), len(schedule)),
        ("Settlement boundary", float((schedule.FinishBoundaryHour - int(CONFIG["settlement_end_hour"])).max()), len(schedule)),
    ])
    result = pd.DataFrame(output, columns=["check", "maximum_violation", "records_checked"])
    result["status"] = np.where(result.maximum_violation <= 1e-6, "PASS", "FAIL")
    return result[["check", "status", "maximum_violation", "records_checked"]]


def plot_pareto(pareto):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    good = pareto.loc[pareto.all_constraints_pass & pareto.epsilon_constraints_pass]
    scatter = ax.scatter(good.Carbon_tCO2, good.IncrementalCost_CNY, c=good.RenewableUtilization, cmap="viridis", s=60)
    for x in good.itertuples(index=False): ax.annotate(x.scenario_id, (x.Carbon_tCO2, x.IncrementalCost_CNY), fontsize=7)
    fig.colorbar(scatter, ax=ax, label="Renewable utilization")
    ax.set(xlabel="Carbon emission (tCO2)", ylabel="Incremental cost (CNY)", title="Q2 feasible Pareto candidates")
    fig.tight_layout(); fig.savefig(OUT_DIR / "figures" / "pareto_cost_carbon.png", dpi=160); plt.close(fig)


def energy_master(ctx: dict[str, np.ndarray], total_ai_it_mwh: float, objective: str,
                  carbon_budget: float | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """Solve the continuous no-storage energy master and return AI IT-load targets.

    GPU/profile feasibility remains task-level.  This LP only provides a
    transparent region-hour energy target and exact no-storage accounting.
    """
    r_count, h_count = ctx["renew"].shape; n = r_count * h_count
    pue = np.repeat(ctx["pue"], h_count)
    nonai = ctx["nonai"].ravel(); renew = ctx["renew"].ravel()
    price = ctx["price"].ravel(); sell_price = ctx["sell_price"].ravel()
    carbon = ctx["carbon"].ravel(); imp = ctx["import_cap"].ravel(); exp = ctx["export_cap"].ravel()
    it_room = np.maximum(ctx["it_cap"].ravel() - nonai, 0.0)
    fac_room = np.maximum(ctx["facility_cap"].ravel() / pue - nonai, 0.0)
    y_upper = np.minimum(it_room, fac_room)
    # Variables are y (AI IT), d (direct renewable), g (grid), s (sell), k (curtail).
    y, direct, grid, sell, curtail = (np.arange(n) + q * n for q in range(5))
    rows = []; cols = []; data = []; rhs = np.empty(2 * n + 1)
    for j in range(n):
        rows.extend([j, j, j]); cols.extend([y[j], direct[j], grid[j]]); data.extend([pue[j], -1.0, -1.0]); rhs[j] = -pue[j] * nonai[j]
        row = n + j
        rows.extend([row, row, row]); cols.extend([direct[j], sell[j], curtail[j]]); data.extend([1.0, 1.0, 1.0]); rhs[row] = renew[j]
    energy_row = 2 * n
    rows.extend([energy_row] * n); cols.extend(y.tolist()); data.extend([1.0] * n); rhs[energy_row] = total_ai_it_mwh
    aeq = coo_matrix((data, (rows, cols)), shape=(2 * n + 1, 5 * n)).tocsr()
    c = np.zeros(5 * n)
    if objective == "cost":
        c[grid] = price; c[sell] = -sell_price
    elif objective == "carbon":
        c[grid] = carbon
    elif objective == "curtail":
        c[curtail] = 1.0
    else:
        raise ValueError(f"Unknown master objective: {objective}")
    aub = None; bub = None
    if carbon_budget is not None:
        aub = coo_matrix((carbon, (np.zeros(n), grid)), shape=(1, 5 * n)).tocsr()
        bub = np.array([carbon_budget])
    bounds = list(zip(np.zeros(n), y_upper)) + list(zip(np.zeros(n), renew)) + list(zip(np.zeros(n), imp)) + list(zip(np.zeros(n), exp)) + list(zip(np.zeros(n), renew))
    result = linprog(c, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=rhs, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Energy master ({objective}) failed: {result.message}")
    x = result.x
    m = {"MasterCost_CNY": float(price @ x[grid] - sell_price @ x[sell]), "MasterCarbon_tCO2": float(carbon @ x[grid]),
         "MasterCurtailment_MWh": float(x[curtail].sum()), "MasterObjective": objective}
    return x[y].reshape(r_count, h_count), m


def master_target(y: np.ndarray) -> np.ndarray:
    upper = max(float(y.max()), EPS)
    return y / upper


def score_weights(kind: str) -> dict[str, float]:
    base = {"cost": 1.0, "carbon": 0.0, "renew": 0.0, "target": 250.0}
    if kind == "carbon": base.update(carbon=CONFIG["carbon_shadow_price_cny_per_tco2"])
    if kind == "renew": base.update(renew=CONFIG["renewable_shadow_price_cny_per_mwh"])
    if kind == "balanced": base.update(carbon=0.25 * CONFIG["carbon_shadow_price_cny_per_tco2"], renew=0.5 * CONFIG["renewable_shadow_price_cny_per_mwh"])
    return base


def make_record(task, region: str, start: int, latency: float, delta: tuple[float, float, float] | None = None) -> dict:
    dc, dcar, dreu = delta if delta is not None else (0.0, 0.0, 0.0)
    return {"TaskID": int(task.TaskID), "TaskType": task.TaskType, "SourceRegion": task.SourceRegion,
            "ExecuteRegion": region, "ArrivalHour": int(task.ArrivalHour), "StartHour": int(start),
            "EndHour": float(start + task.duration_h), "FinishBoundaryHour": int(start + task.duration_ceil_h),
            "WaitHour": int(start - task.ArrivalHour), "NetworkLatency_ms": float(latency),
            "MaxLatency_ms": int(task.MaxLatency_ms), "LatestFinishHour": int(task.LatestFinishHour),
            "duration_h": float(task.duration_h), "GPU_Demand": int(task.GPU_Demand), "Migrated": task.SourceRegion != region,
            "marginal_cost_cny": dc, "marginal_carbon_tco2": dcar, "marginal_renewable_mwh": dreu}


def construct_seed(tasks: pd.DataFrame, profiles, eligibility, names, r_index, ctx, target: np.ndarray, kind: str):
    """Task-level capacity-feasible construction under a chosen energy seed."""
    h = int(CONFIG["settlement_end_hour"]); ai = np.zeros((len(names), h)); gpu = np.zeros_like(ai)
    weights = score_weights(kind); order = {"RealTimeInference": 0, "BatchInference": 1, "AITraining": 2}; records = []
    ordered = tasks.sort_values("ArrivalHour").sort_values("TaskType", key=lambda s: s.map(order), kind="stable")
    for task in ordered.itertuples(index=False):
        prof = profiles[int(task.TaskID)]; choices = []
        for region, latency in eligibility[int(task.TaskID)]:
            r = r_index[region]
            for start in start_candidates(task, region, target, r):
                if not is_feasible(ai, gpu, r, start, prof, ctx): continue
                delta = candidate_energy_delta(ai, r, start, prof, ctx)
                if delta is None: continue
                dc, dcar, dreu = delta; wait = start - int(task.ArrivalHour)
                wait_weight = CONFIG["batch_wait_weight"] if task.TaskType == "BatchInference" else CONFIG["training_wait_weight"] if task.TaskType == "AITraining" else 0.0
                rel, _, ai_mw = prof; gain = float((target[r, start + rel] * ai_mw).sum())
                score = weights["cost"] * dc + weights["carbon"] * dcar - weights["renew"] * dreu - weights["target"] * gain
                score += CONFIG["wait_shadow_price_cny_per_hour"] * wait_weight * wait + CONFIG["latency_shadow_price_cny_per_ms"] * latency
                choices.append((score, region, r, start, latency, delta))
        if not choices:
            # Narrow energy candidates can be exhausted near a GPU peak.  Expand
            # only this task's legal window before declaring the greedy state
            # infeasible; normal tasks retain the bounded candidate set.
            for region, latency in eligibility[int(task.TaskID)]:
                r = r_index[region]; earliest = int(task.EarliestStartHour); latest = min(int(task.LatestFinishHour), h) - int(task.duration_ceil_h)
                for start in range(earliest, latest + 1):
                    if task.TaskType == "RealTimeInference" and start != int(task.ArrivalHour): continue
                    if not is_feasible(ai, gpu, r, start, prof, ctx): continue
                    delta = candidate_energy_delta(ai, r, start, prof, ctx)
                    if delta is None: continue
                    dc, dcar, dreu = delta; wait = start - int(task.ArrivalHour)
                    wait_weight = CONFIG["batch_wait_weight"] if task.TaskType == "BatchInference" else CONFIG["training_wait_weight"] if task.TaskType == "AITraining" else 0.0
                    rel, _, ai_mw = prof; gain = float((target[r, start + rel] * ai_mw).sum())
                    score = weights["cost"] * dc + weights["carbon"] * dcar - weights["renew"] * dreu - weights["target"] * gain
                    score += CONFIG["wait_shadow_price_cny_per_hour"] * wait_weight * wait + CONFIG["latency_shadow_price_cny_per_ms"] * latency
                    choices.append((score, region, r, start, latency, delta))
            if not choices: raise RuntimeError(f"No capacity-feasible seed placement for task {task.TaskID}.")
        _, region, r, start, latency, delta = min(choices, key=lambda x: x[0]); apply(ai, gpu, r, start, prof)
        records.append(make_record(task, region, start, latency, delta))
    return pd.DataFrame(records), ai, gpu


def safe_construct_seed(tasks: pd.DataFrame, profiles, eligibility, names, r_index, ctx, target: np.ndarray, kind: str):
    """Use LP-guided construction when possible, with a proven feasible fallback."""
    # Carbon/curtailment targets concentrate too many early tasks in the same
    # hours.  Start these directions from the proven capacity construction and
    # let the target-guided local repair perform the energy shift instead.
    if kind in {"carbon", "renew"}:
        carbon_ratio = 0.0 if kind == "carbon" else 1.0
        renewable_fraction = 1.0 if kind == "renew" else 0.0
        return construct_feasible_schedule(tasks, profiles, eligibility, names, r_index, ctx, carbon_ratio, renewable_fraction)
    try:
        return construct_seed(tasks, profiles, eligibility, names, r_index, ctx, target, kind)
    except RuntimeError as exc:
        print(f"Seed '{kind}' fell back to capacity construction: {exc}")
        carbon_ratio = 0.0 if kind == "carbon" else 1.0
        renewable_fraction = 1.0 if kind == "renew" else 0.0
        return construct_feasible_schedule(tasks, profiles, eligibility, names, r_index, ctx, carbon_ratio, renewable_fraction)


def schedule_stats(schedule: pd.DataFrame, ai: np.ndarray, ctx: dict[str, np.ndarray]) -> dict[str, float]:
    m = metrics(ai, ctx); m["RenewableUsedOrSold_MWh"] = m["RenewableUtilization"] * float(ctx["renew"].sum())
    m["WeightedWaitHour"] = float(np.where(schedule.TaskType.eq("BatchInference"), CONFIG["batch_wait_weight"] * schedule.WaitHour,
                                             np.where(schedule.TaskType.eq("AITraining"), CONFIG["training_wait_weight"] * schedule.WaitHour, 0.0)).sum())
    m["AverageWaitHour"] = float(schedule.WaitHour.mean()); m["AverageLatency_ms"] = float(schedule.NetworkLatency_ms.mean()); m["MigrationRate"] = float(schedule.Migrated.mean())
    return m


def repair_schedule(schedule: pd.DataFrame, profiles, tasks_by_id: pd.DataFrame, eligibility, names, r_index, ctx,
                    target: np.ndarray, carbon_cap: float, renewable_floor: float, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Incremental relocation and pair-swap repair with hard physical checks."""
    h = int(CONFIG["settlement_end_hour"]); ai, gpu = rebuild_load(schedule, profiles, names, r_index, h)
    records = {int(x.TaskID): dict(x._asdict()) for x in schedule.itertuples(index=False)}; total_renew = float(ctx["renew"].sum())
    current = schedule_stats(schedule, ai, ctx)
    def wait_contribution(rec: dict) -> float:
        if rec["TaskType"] == "BatchInference": return CONFIG["batch_wait_weight"] * rec["WaitHour"]
        if rec["TaskType"] == "AITraining": return CONFIG["training_wait_weight"] * rec["WaitHour"]
        return 0.0
    def key(m, rec):
        return (max(0.0, m["Carbon_tCO2"] - carbon_cap), max(0.0, renewable_floor * total_renew - m["RenewableUsedOrSold_MWh"]),
                m["Cost_CNY"], m["WeightedWaitHour"], rec["NetworkLatency_ms"], float(rec["Migrated"]))
    ids = np.asarray(list(records)); move_attempts = int(CONFIG["local_move_attempts"])
    for tid in rng.choice(ids, size=move_attempts, replace=True):
        tid = int(tid); old = records[tid]; task = tasks_by_id.loc[tid]; prof = profiles[tid]; old_r = r_index[old["ExecuteRegion"]]; old_start = int(old["StartHour"])
        apply(ai, gpu, old_r, old_start, prof, -1.0); old_delta = candidate_energy_delta(ai, old_r, old_start, prof, ctx)
        base = current.copy(); base["Cost_CNY"] -= old_delta[0]; base["Carbon_tCO2"] -= old_delta[1]; base["RenewableUsedOrSold_MWh"] -= old_delta[2]; base["WeightedWaitHour"] -= wait_contribution(old)
        choices = [(key(current, old), old, old_r, old_start, None)]
        for region, latency in eligibility[tid]:
            r = r_index[region]
            for start in set(start_candidates(task, region, target, r) + ([old_start] if region == old["ExecuteRegion"] else [])):
                if task.TaskType == "RealTimeInference" and start != int(task.ArrivalHour): continue
                if not is_feasible(ai, gpu, r, start, prof, ctx): continue
                delta = candidate_energy_delta(ai, r, start, prof, ctx)
                if delta is None: continue
                rec = make_record(task, region, start, latency, delta)
                trial = base.copy(); trial["Cost_CNY"] += delta[0]; trial["Carbon_tCO2"] += delta[1]; trial["RenewableUsedOrSold_MWh"] += delta[2]; trial["WeightedWaitHour"] += wait_contribution(rec)
                choices.append((key(trial, rec), rec, r, start, trial))
        best = min(choices, key=lambda x: x[0])
        if best[4] is None:
            apply(ai, gpu, old_r, old_start, prof); continue
        _, rec, r, start, trial = best; apply(ai, gpu, r, start, prof); records[tid] = rec; current = trial
    # Pair swaps address tightly saturated hours without opening an uncontrolled candidate expansion.
    for _ in range(int(CONFIG["local_swap_attempts"])):
        a, b = map(int, rng.choice(ids, size=2, replace=False)); ra, rb = records[a], records[b]
        if ra["TaskType"] == "RealTimeInference" or rb["TaskType"] == "RealTimeInference": continue
        ta, tb = tasks_by_id.loc[a], tasks_by_id.loc[b]; pa, pb = profiles[a], profiles[b]
        if rb["ExecuteRegion"] not in {x[0] for x in eligibility[a]} or ra["ExecuteRegion"] not in {x[0] for x in eligibility[b]}: continue
        sa, sb = int(rb["StartHour"]), int(ra["StartHour"])
        if not (int(ta.EarliestStartHour) <= sa <= min(int(ta.LatestFinishHour), h) - int(ta.duration_ceil_h)): continue
        if not (int(tb.EarliestStartHour) <= sb <= min(int(tb.LatestFinishHour), h) - int(tb.duration_ceil_h)): continue
        old_ra, old_rb = r_index[ra["ExecuteRegion"]], r_index[rb["ExecuteRegion"]]
        apply(ai, gpu, old_ra, int(ra["StartHour"]), pa, -1.0); apply(ai, gpu, old_rb, int(rb["StartHour"]), pb, -1.0)
        old_da = candidate_energy_delta(ai, old_ra, int(ra["StartHour"]), pa, ctx); old_db = candidate_energy_delta(ai, old_rb, int(rb["StartHour"]), pb, ctx)
        base = current.copy()
        for idx in (0, 1, 2): base[["Cost_CNY", "Carbon_tCO2", "RenewableUsedOrSold_MWh"][idx]] -= old_da[idx] + old_db[idx]
        new_ra, new_rb = r_index[rb["ExecuteRegion"]], r_index[ra["ExecuteRegion"]]
        if not is_feasible(ai, gpu, new_ra, sa, pa, ctx): apply(ai, gpu, old_ra, int(ra["StartHour"]), pa); apply(ai, gpu, old_rb, int(rb["StartHour"]), pb); continue
        apply(ai, gpu, new_ra, sa, pa); 
        if not is_feasible(ai, gpu, new_rb, sb, pb, ctx): apply(ai, gpu, new_ra, sa, pa, -1.0); apply(ai, gpu, old_ra, int(ra["StartHour"]), pa); apply(ai, gpu, old_rb, int(rb["StartHour"]), pb); continue
        da = candidate_energy_delta(ai, new_rb, sb, pb, ctx)
        apply(ai, gpu, new_rb, sb, pb)
        # Recompute affected aggregate metrics exactly after a proposed swap.
        trial_sched = pd.DataFrame(records.values()); trial_sched.loc[trial_sched.TaskID == a, ["ExecuteRegion", "StartHour"]] = [rb["ExecuteRegion"], sa]
        trial_sched.loc[trial_sched.TaskID == b, ["ExecuteRegion", "StartHour"]] = [ra["ExecuteRegion"], sb]
        trial = schedule_stats(trial_sched, ai, ctx)
        trial["RenewableUsedOrSold_MWh"] = trial["RenewableUtilization"] * total_renew
        old_key = max(0.0, current["Carbon_tCO2"] - carbon_cap), max(0.0, renewable_floor * total_renew - current["RenewableUsedOrSold_MWh"]), current["Cost_CNY"]
        new_key = max(0.0, trial["Carbon_tCO2"] - carbon_cap), max(0.0, renewable_floor * total_renew - trial["RenewableUsedOrSold_MWh"]), trial["Cost_CNY"]
        if new_key < old_key:
            la = dict(eligibility[a]).get(rb["ExecuteRegion"], 0.0); lb = dict(eligibility[b]).get(ra["ExecuteRegion"], 0.0)
            records[a] = make_record(ta, rb["ExecuteRegion"], sa, la); records[b] = make_record(tb, ra["ExecuteRegion"], sb, lb); current = trial
        else:
            apply(ai, gpu, new_ra, sa, pa, -1.0); apply(ai, gpu, new_rb, sb, pb, -1.0); apply(ai, gpu, old_ra, int(ra["StartHour"]), pa); apply(ai, gpu, old_rb, int(rb["StartHour"]), pb)
    return pd.DataFrame(records.values()), ai, gpu


def main() -> None:
    reset_output()
    inputs = load_inputs(); names, r_index, ctx = make_context(inputs)
    profiles = profile_lookup(inputs.profile); eligible = feasible_regions(inputs)
    tasks = inputs.tasks.loc[inputs.tasks.ArrivalHour.between(0, 2399)].copy()
    reference = reference_schedule(tasks, profiles, names, r_index)
    ref_ai, ref_gpu = rebuild_load(reference, profiles, names, r_index, int(CONFIG["settlement_end_hour"]))
    ref_metrics = metrics(ref_ai, ctx)
    ref_frame = region_hour_frame(ref_ai, ref_gpu, names, ctx)
    ref_checks = checks(reference, ref_frame, ctx)
    ref_metrics["all_constraints_pass"] = bool((ref_checks.status == "PASS").all())
    # Source-immediate is a reporting comparator only: the supplied data have a
    # GPU peak above capacity, so it cannot define a feasible epsilon baseline.
    feasible_base, base_ai, base_gpu = construct_feasible_schedule(tasks, profiles, eligible, names, r_index, ctx, 1.0, 0.0)
    base_metrics = metrics(base_ai, ctx)
    base_frame = region_hour_frame(base_ai, base_gpu, names, ctx)
    base_checks = checks(feasible_base, base_frame, ctx)
    base_metrics["all_constraints_pass"] = bool((base_checks.status == "PASS").all())
    if not base_metrics["all_constraints_pass"]:
        raise RuntimeError("The capacity-feasible epsilon baseline failed a hard-constraint check.")
    scenarios = [(c, f) for c in CONFIG["carbon_budget_ratios"] for f in CONFIG["renewable_floor_fractions"]]
    rows = []; outputs = []
    for carbon_ratio, reu_fraction in scenarios:
        carbon_budget, renewable_floor = epsilon_targets(base_metrics, float(carbon_ratio), float(reu_fraction))
        if carbon_ratio == 1.0 and reu_fraction == 0.0:
            schedule, ai, gpu = feasible_base, base_ai, base_gpu
        else:
            schedule, ai, gpu = construct_feasible_schedule(tasks, profiles, eligible, names, r_index, ctx, float(carbon_ratio), float(reu_fraction))
        m = metrics(ai, ctx); frame = region_hour_frame(ai, gpu, names, ctx); ck = checks(schedule, frame, ctx)
        sid = f"carbon_{carbon_ratio:.2f}_reu_{reu_fraction:.2f}"
        constraints_pass = bool((ck.status == "PASS").all())
        carbon_ok = bool(m["Carbon_tCO2"] <= carbon_budget + EPS)
        renewable_ok = bool(m["RenewableUtilization"] + EPS >= renewable_floor)
        row = {"scenario_id": sid, "carbon_budget_ratio": carbon_ratio, "renewable_target_fraction": reu_fraction,
               "CarbonBudget_tCO2": carbon_budget, "RenewableFloor": renewable_floor,
               **m, "IncrementalCost_CNY": m["Cost_CNY"] - base_metrics["Cost_CNY"],
               "IncrementalCarbon_tCO2": m["Carbon_tCO2"] - base_metrics["Carbon_tCO2"],
               "Cost_vs_SourceImmediate_CNY": m["Cost_CNY"] - ref_metrics["Cost_CNY"],
               "Carbon_vs_SourceImmediate_tCO2": m["Carbon_tCO2"] - ref_metrics["Carbon_tCO2"],
               "AverageWaitHour": float(schedule.WaitHour.mean()), "AverageLatency_ms": float(schedule.NetworkLatency_ms.mean()),
               "MigrationRate": float(schedule.Migrated.mean()), "carbon_budget_satisfied": carbon_ok,
               "renewable_floor_satisfied": renewable_ok, "all_constraints_pass": constraints_pass,
               "epsilon_constraints_pass": bool(carbon_ok and renewable_ok)}
        rows.append(row); outputs.append((sid, schedule, frame, ck))
    pareto = pd.DataFrame(rows)
    # Retain non-dominated candidates that pass physical and final epsilon checks.
    feasible = pareto.loc[pareto.all_constraints_pass & pareto.epsilon_constraints_pass].copy(); keep = []
    for i, a in feasible.iterrows():
        dominated = False
        for j, b in feasible.iterrows():
            if i == j: continue
            left = [b.IncrementalCost_CNY, b.Carbon_tCO2, -b.RenewableUtilization, b.AverageWaitHour, b.AverageLatency_ms]
            right = [a.IncrementalCost_CNY, a.Carbon_tCO2, -a.RenewableUtilization, a.AverageWaitHour, a.AverageLatency_ms]
            if all(x <= y + 1e-8 for x, y in zip(left, right)) and any(x < y - 1e-8 for x, y in zip(left, right)):
                dominated = True; break
        if not dominated: keep.append(i)
    pareto["is_nondominated"] = pareto.index.isin(keep)
    write_csv(pareto, "pareto_solutions.csv")
    write_csv(pareto[["scenario_id", "CarbonBudget_tCO2", "Carbon_tCO2", "carbon_budget_satisfied",
                      "RenewableFloor", "RenewableUtilization", "renewable_floor_satisfied",
                      "all_constraints_pass", "epsilon_constraints_pass"]], "scenario_epsilon_checks.csv")
    write_csv(pd.DataFrame([{**{"scenario_id": "source_immediate_no_storage"}, **ref_metrics}]), "reference_metrics.csv")
    write_csv(ref_checks, "source_immediate_constraint_checks.csv")
    write_csv(pd.DataFrame([{**{"scenario_id": "capacity_feasible_baseline"}, **base_metrics}]), "feasible_baseline_metrics.csv")
    for sid, schedule, frame, ck in outputs:
        if sid in set(pareto.loc[pareto.is_nondominated, "scenario_id"]):
            write_csv(schedule, f"task_schedule_{sid}.csv"); write_csv(frame, f"region_hour_{sid}.csv"); write_csv(ck, f"constraint_checks_{sid}.csv")
    plot_pareto(pareto)
    report = ["# 问题二：碳感知调度运行报告", "", "## 同口径参考", "", md_table(pd.DataFrame([ref_metrics])), "",
              "## Pareto 候选", "", md_table(pareto), "", "## 说明", "",
              "- 本实现不优化储能；购电、售电、弃电和直接新能源消纳按逐时无储能平衡重算。",
              "- 所有最终指标来自任务级非抢占排程。能源机会主层只用于生成区域—时段候选优先级。",
              "- `carbon_budget_ratio` 以同口径基准碳排放为上界比例；`renewable_target_fraction` 将可再生利用率从基准值线性提高到 1。",
              "- 能源机会层的罚函数用于引导搜索；只有逐任务排程重算后同时满足物理约束和 ε 约束的点，才标记为 `is_nondominated=True`。",
              "- `is_nondominated=True` 的方案已保存逐任务和逐时明细；其他方案仅保留 Pareto 汇总和 ε 校验表。"]
    report = [
        "# 问题二：碳感知调度运行报告", "",
        "## 原始源区域到达即启动对照（不作为可行解）", "", md_table(pd.DataFrame([ref_metrics])), "",
        md_table(ref_checks), "",
        "该对照的 GPU 峰值超容量，因此仅用于保留原始业务口径；不进入 Pareto 集，也不定义 ε 阈值。", "",
        "## 容量可行基准（ε 与增量指标基准）", "", md_table(pd.DataFrame([base_metrics])), "",
        "## Pareto 候选", "", md_table(pareto), "",
        "## 说明", "",
        "- 无储能：逐时按本地直供、余电售出、剩余弃电和电网购电平衡。",
        "- 每个场景均从空负载逐任务构造，并逐任务检查 GPU、IT/设施功率、网络时延、实时启动和最晚结束约束。",
        "- `carbon_budget_ratio` 与 `renewable_target_fraction` 均以容量可行基准定义；只有物理约束和 ε 检查同时通过的方案才可能标记为 `is_nondominated=True`。",
    ]
    (OUT_DIR / "q2_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Q2 scheduler completed: {OUT_DIR}; scenarios={len(pareto)}, nondominated={int(pareto.is_nondominated.sum())}")


def optimized_main() -> None:
    reset_output()
    inputs = load_inputs(); names, r_index, ctx = make_context(inputs)
    profiles = profile_lookup(inputs.profile); eligible = feasible_regions(inputs)
    tasks = inputs.tasks.loc[inputs.tasks.ArrivalHour.between(0, 2399)].copy(); tasks_by_id = tasks.set_index("TaskID", drop=False)
    total_ai_it = float(inputs.profile.loc[inputs.profile.TaskID.isin(tasks.TaskID), "ai_it_energy_mwh"].sum())
    rng = np.random.default_rng(int(CONFIG["random_seed"])); h = int(CONFIG["settlement_end_hour"])

    source = reference_schedule(tasks, profiles, names, r_index); source_ai, source_gpu = rebuild_load(source, profiles, names, r_index, h)
    source_metrics = metrics(source_ai, ctx); source_frame = region_hour_frame(source_ai, source_gpu, names, ctx); source_checks = checks(source, source_frame, ctx)
    source_metrics["all_constraints_pass"] = bool((source_checks.status == "PASS").all())
    write_csv(pd.DataFrame([{**{"scenario_id": "source_immediate_no_storage"}, **source_metrics}]), "reference_metrics.csv")
    write_csv(source_checks, "source_immediate_constraint_checks.csv")

    # Endpoint 1: capacity-feasible cost schedule. Endpoint 2: carbon-first schedule.
    y_cost, lp_cost = energy_master(ctx, total_ai_it, "cost")
    base_s, base_ai, base_gpu = safe_construct_seed(tasks, profiles, eligible, names, r_index, ctx, master_target(y_cost), "cost")
    base_m = schedule_stats(base_s, base_ai, ctx)
    y_carbon, lp_carbon = energy_master(ctx, total_ai_it, "carbon")
    carbon_s, carbon_ai, carbon_gpu = safe_construct_seed(tasks, profiles, eligible, names, r_index, ctx, master_target(y_carbon), "carbon")
    carbon_s, carbon_ai, carbon_gpu = repair_schedule(carbon_s, profiles, tasks_by_id, eligible, names, r_index, ctx, master_target(y_carbon), 0.0, 0.0, rng)
    carbon_m = schedule_stats(carbon_s, carbon_ai, ctx)
    baseline_checks = checks(base_s, region_hour_frame(base_ai, base_gpu, names, ctx), ctx)
    base_m["all_constraints_pass"] = bool((baseline_checks.status == "PASS").all())
    if not base_m["all_constraints_pass"]: raise RuntimeError("Capacity-feasible baseline failed hard checks.")
    c_base = float(base_m["Carbon_tCO2"]); c_min = min(c_base, float(carbon_m["Carbon_tCO2"]))
    carbon_levels = np.linspace(c_base, c_min, int(CONFIG["carbon_level_count"]))
    write_csv(pd.DataFrame([{**{"scenario_id": "capacity_feasible_cost_baseline"}, **base_m}, {**{"scenario_id": "carbon_endpoint"}, **carbon_m}]), "feasible_baseline_metrics.csv")

    endpoint_rows = []; definition_rows = []; summary_rows = []; all_checks = []; detailed = []
    reu_fractions = [float(x) for x in CONFIG["renewable_floor_fractions"]]
    for ci, cap in enumerate(carbon_levels):
        y_c, lp_c = energy_master(ctx, total_ai_it, "cost", float(cap))
        y_u, lp_u = energy_master(ctx, total_ai_it, "curtail", float(cap))
        seeds = []
        for label, target, kind in (("cost_seed", master_target(y_c), "cost"), ("renew_seed", master_target(y_u), "renew")):
            s, ai, gpu = safe_construct_seed(tasks, profiles, eligible, names, r_index, ctx, target, kind)
            s, ai, gpu = repair_schedule(s, profiles, tasks_by_id, eligible, names, r_index, ctx, target, float(cap), 0.0, rng)
            seeds.append((label, s, ai, gpu, schedule_stats(s, ai, ctx), target))
        # The carbon endpoint supplies a feasible fallback at every cap by construction.
        seeds.append(("carbon_endpoint", carbon_s.copy(), carbon_ai.copy(), carbon_gpu.copy(), carbon_m.copy(), master_target(y_carbon)))
        valid = [x for x in seeds if x[4]["Carbon_tCO2"] <= cap + 1e-6]
        if not valid: raise RuntimeError(f"No task-level schedule met generated carbon cap {cap:.6f}.")
        low = min(valid, key=lambda x: x[4]["Cost_CNY"]); high = max(valid, key=lambda x: x[4]["RenewableUtilization"])
        u_low, u_max = float(low[4]["RenewableUtilization"]), float(high[4]["RenewableUtilization"])
        endpoint_rows.append({"carbon_level_id": ci, "CarbonBudget_tCO2": cap, "CarbonBudgetRatio": cap / c_base if c_base > EPS else 0.0,
                              "REU_Low": u_low, "REU_MaxFound": u_max, "CostEndpoint_CNY": low[4]["Cost_CNY"],
                              "CarbonEndpoint_tCO2": carbon_m["Carbon_tCO2"], **lp_c, **{f"REU_{k}": v for k, v in lp_u.items()}})
        for fi, frac in enumerate(reu_fractions):
            floor = u_low + frac * (u_max - u_low); candidates = [("low_endpoint", *low[1:5]), ("reu_endpoint", *high[1:5])]
            # Repair both endpoint starts against the exact final epsilon pair.
            for start_label, start_s, start_ai, start_gpu, start_m in list(candidates):
                target = low[5] if start_label == "low_endpoint" else high[5]
                rs, rai, rgpu = repair_schedule(start_s.copy(), profiles, tasks_by_id, eligible, names, r_index, ctx, target, float(cap), float(floor), rng)
                candidates.append(("repair_" + start_label, rs, rai, rgpu, schedule_stats(rs, rai, ctx)))
            feasible = [x for x in candidates if x[4]["Carbon_tCO2"] <= cap + 1e-6 and x[4]["RenewableUtilization"] + 1e-9 >= floor]
            if not feasible: raise RuntimeError(f"No epsilon-feasible candidate at carbon level {ci}, REU fraction {frac}.")
            chosen = min(feasible, key=lambda x: (x[4]["Cost_CNY"], x[4]["WeightedWaitHour"], x[4]["AverageLatency_ms"], x[4]["MigrationRate"]))
            label, s, ai, gpu, m = chosen; frame = region_hour_frame(ai, gpu, names, ctx); ck = checks(s, frame, ctx)
            physical_ok = bool((ck.status == "PASS").all()); sid = f"carbon_{ci:02d}_reu_{fi:02d}"
            row = {"scenario_id": sid, "carbon_level_id": ci, "reu_level_fraction": frac, "CandidateSource": label,
                   "CarbonBudget_tCO2": cap, "CarbonBudgetRatio": cap / c_base if c_base > EPS else 0.0,
                   "RenewableFloor": floor, "REU_Low": u_low, "REU_MaxFound": u_max, **m,
                   "IncrementalCost_CNY": m["Cost_CNY"] - base_m["Cost_CNY"], "IncrementalCarbon_tCO2": m["Carbon_tCO2"] - base_m["Carbon_tCO2"],
                   "Cost_vs_SourceImmediate_CNY": m["Cost_CNY"] - source_metrics["Cost_CNY"], "Carbon_vs_SourceImmediate_tCO2": m["Carbon_tCO2"] - source_metrics["Carbon_tCO2"],
                   "carbon_budget_satisfied": m["Carbon_tCO2"] <= cap + 1e-6, "renewable_floor_satisfied": m["RenewableUtilization"] + 1e-9 >= floor,
                   "all_constraints_pass": physical_ok}
            row["epsilon_constraints_pass"] = bool(row["carbon_budget_satisfied"] and row["renewable_floor_satisfied"] and physical_ok)
            summary_rows.append(row); detailed.append((sid, s, frame, ck, row)); definition_rows.append({"scenario_id": sid, "CarbonBudget_tCO2": cap, "CarbonBudgetRatio": row["CarbonBudgetRatio"], "RenewableFloor": floor, "REU_Low": u_low, "REU_MaxFound": u_max, "REU_LevelFraction": frac})
            ck = ck.copy(); ck.insert(0, "scenario_id", sid); all_checks.append(ck)

    summary = pd.DataFrame(summary_rows); feasible = summary.loc[summary.epsilon_constraints_pass].copy(); keep = []
    dims = ["IncrementalCost_CNY", "Carbon_tCO2", "RenewableUtilization", "AverageWaitHour", "AverageLatency_ms", "MigrationRate"]
    for i, a in feasible.iterrows():
        dominated = False
        for j, b in feasible.iterrows():
            if i == j: continue
            left = [b[dims[0]], b[dims[1]], -b[dims[2]], b[dims[3]], b[dims[4]], b[dims[5]]]
            right = [a[dims[0]], a[dims[1]], -a[dims[2]], a[dims[3]], a[dims[4]], a[dims[5]]]
            if all(x <= y + EPS for x, y in zip(left, right)) and any(x < y - EPS for x, y in zip(left, right)): dominated = True; break
        if not dominated: keep.append(i)
    summary["is_nondominated"] = summary.index.isin(keep)
    write_csv(pd.DataFrame(endpoint_rows), "endpoint_bounds.csv"); write_csv(pd.DataFrame(definition_rows), "scenario_epsilon_definition.csv")
    write_csv(summary, "scenario_summary.csv"); write_csv(summary, "pareto_solutions.csv"); write_csv(pd.concat(all_checks, ignore_index=True), "constraint_checks_all_scenarios.csv")
    for sid, s, frame, ck, row in detailed:
        if row["epsilon_constraints_pass"]:
            write_csv(s, f"task_schedule_{sid}.csv"); write_csv(frame, f"region_hour_{sid}.csv"); write_csv(ck, f"constraint_checks_{sid}.csv")
    plot_pareto(summary)
    report = ["# 问题二：数据驱动 ε-约束调度报告", "", "## 原始业务对照（不可行）", "", md_table(pd.DataFrame([source_metrics])), "", md_table(source_checks), "",
              "## 可行端点", "", md_table(pd.DataFrame([{**{"name": "成本基准"}, **base_m}, {**{"name": "最低碳端点"}, **carbon_m}])), "",
              "## 各碳预算下的可达 REU 区间", "", md_table(pd.DataFrame(endpoint_rows)), "", "## 场景汇总", "", md_table(summary), "",
              "REU 档位在每个碳预算下均按已搜索的 [REU_Low, REU_MaxFound] 区间归一化；仅物理和 ε 约束均通过的点参与 Pareto 筛选。"]
    (OUT_DIR / "q2_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Q2 optimized scheduler completed: {OUT_DIR}; scenarios={len(summary)}, nondominated={int(summary.is_nondominated.sum())}")


if __name__ == "__main__":
    optimized_main()
