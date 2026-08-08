"""Question 3: regional storage co-optimization under fixed loads.

The model intentionally has no inter-regional electricity-flow variables.  Every
region can exchange electricity only with the external grid, under the limits
given in the attachment.  The implementation is a sparse LP solved by HiGHS.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROCESSED = ROOT / "步骤0_数据预处理与分析" / "processed"
CONFIG_PATH = HERE / "q3_config.json"


@dataclass
class Inputs:
    regions: list[str]
    hours: np.ndarray
    load: np.ndarray
    renewable: np.ndarray
    price: np.ndarray
    sell_price: np.ndarray
    carbon: np.ndarray
    capacity: np.ndarray
    min_soc: np.ndarray
    initial_soc: np.ndarray
    max_charge: np.ndarray
    max_discharge: np.ndarray
    eta_charge: np.ndarray
    eta_discharge: np.ndarray
    max_import: np.ndarray
    max_export: np.ndarray
    pue: np.ndarray
    max_it: np.ndarray
    max_facility: np.ndarray

    @property
    def r_count(self) -> int:
        return len(self.regions)

    @property
    def t_count(self) -> int:
        return len(self.hours)


@dataclass
class Candidate:
    name: str
    result: OptimizeResult
    metrics: dict[str, float]
    source: str
    nondominated: bool = False
    labels: str = ""


def log(message: str) -> None:
    print(message, flush=True)


def read_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def read_inputs(smoke_hours: int | None) -> Inputs:
    region = pd.read_csv(PROCESSED / "dim_region.csv")
    storage = pd.read_csv(PROCESSED / "dim_storage.csv")
    hourly = pd.read_csv(PROCESSED / "fact_region_hour_input.csv")
    baseline = pd.read_csv(PROCESSED / "fact_region_hour_baseline.csv")

    merged = hourly.merge(
        baseline[["Hour", "Region", "Baseline_AI_IT_Load_MW"]],
        on=["Hour", "Region"], validate="one_to_one"
    ).merge(
        region[["Region", "PUE", "Max_IT_Power_MW", "Max_Facility_Power_MW"]],
        on="Region", validate="many_to_one"
    ).merge(storage, on="Region", validate="many_to_one")

    regions = region["Region"].tolist()
    full_hours = np.arange(0, 2407, dtype=int)
    if set(merged["Hour"]) != set(full_hours):
        raise ValueError("Processed regional-hour input must cover every hour 0..2406.")
    if set(merged["Region"]) != set(regions):
        raise ValueError("Region identifiers differ between processed tables.")

    if smoke_hours is not None:
        if smoke_hours < 3 or smoke_hours > len(full_hours):
            raise ValueError("--smoke-hours must be between 3 and 2407.")
        hours = full_hours[:smoke_hours]
    else:
        hours = full_hours
    terminal_hour = int(hours[-1])

    merged = merged[merged["Hour"].isin(hours)].copy()
    merged["FacilityLoad_MW"] = (
        merged["Baseline_AI_IT_Load_MW"] + merged["NonAI_IT_Load_MW"]
    ) * merged["PUE"]
    merged.sort_values(["Region", "Hour"], inplace=True)

    expected_rows = len(regions) * len(hours)
    if len(merged) != expected_rows:
        raise ValueError("Regional-hour data are incomplete after time filtering.")

    def grid_column(column: str) -> np.ndarray:
        return (
            merged.pivot(index="Region", columns="Hour", values=column)
            .reindex(index=regions, columns=hours)
            .to_numpy(dtype=float)
        )

    static = merged.groupby("Region", sort=False).first().reindex(regions)
    result = Inputs(
        regions=regions,
        hours=hours,
        load=grid_column("FacilityLoad_MW"),
        renewable=grid_column("AvailableRenewable_MW"),
        price=grid_column("ElectricityPrice_CNY_per_MWh"),
        sell_price=grid_column("SellPrice_CNY_per_MWh"),
        carbon=grid_column("CarbonIntensity_tCO2_per_MWh"),
        capacity=static["StorageCapacity_MWh"].to_numpy(float),
        min_soc=static["MinSOC_MWh"].to_numpy(float),
        initial_soc=static["InitialSOC_MWh"].to_numpy(float),
        max_charge=static["MaxChargePower_MW"].to_numpy(float),
        max_discharge=static["MaxDischargePower_MW"].to_numpy(float),
        eta_charge=static["ChargeEfficiency"].to_numpy(float),
        eta_discharge=static["DischargeEfficiency"].to_numpy(float),
        max_import=static["MaxGridImport_MW"].to_numpy(float),
        max_export=np.minimum(
            static["SellLimit_MW"].to_numpy(float),
            static["MaxGridExport_MW"].to_numpy(float),
        ),
        pue=static["PUE"].to_numpy(float),
        max_it=static["Max_IT_Power_MW"].to_numpy(float),
        max_facility=static["Max_Facility_Power_MW"].to_numpy(float),
    )
    if np.any(result.initial_soc < result.min_soc) or np.any(result.initial_soc > result.capacity):
        raise ValueError("Initial SOC is outside its configured bounds.")
    if np.any(result.load < -1e-9) or np.any(result.renewable < -1e-9):
        raise ValueError("Load and renewable input must be non-negative.")
    if np.any(result.price + 1e-9 < result.sell_price):
        raise ValueError("This implementation requires buy price >= sell price in every period.")
    log(f"Loaded {result.r_count} regions × {result.t_count} hours; terminal settlement hour={terminal_hour}.")
    return result


def static_checks(data: Inputs, tolerance: float) -> pd.DataFrame:
    rows: list[dict] = []
    for r, name in enumerate(data.regions):
        fixed_it = data.load[r] / data.pue[r]
        rows.append({
            "check": "fixed_it_load_limit",
            "region": name,
            "maximum_violation": float(max(fixed_it.max() - data.max_it[r], 0.0)),
            "passed": bool(fixed_it.max() <= data.max_it[r] + tolerance),
        })
        rows.append({
            "check": "fixed_facility_load_limit",
            "region": name,
            "maximum_violation": float(max(data.load[r].max() - data.max_facility[r], 0.0)),
            "passed": bool(data.load[r].max() <= data.max_facility[r] + tolerance),
        })
        rows.append({
            "check": "buy_price_not_below_sell_price",
            "region": name,
            "maximum_violation": float(max((data.sell_price[r] - data.price[r]).max(), 0.0)),
            "passed": bool(np.all(data.price[r] + tolerance >= data.sell_price[r])),
        })
    return pd.DataFrame(rows)


class StorageLP:
    """Sparse LP representation; all regions are solved together for global KPIs."""

    variable_names = ("direct_renew", "renew_charge", "grid_charge", "discharge",
                      "grid_load", "grid_sell", "curtailment", "soc")

    def __init__(self, data: Inputs, config: dict):
        self.data = data
        self.config = config
        self.r = data.r_count
        self.t = data.t_count
        self.rt = self.r * self.t
        self.p_start = len(self.variable_names) * self.rt
        self.z_start = self.p_start + self.r
        self.n_vars = self.z_start + self.r * (self.t - 1)
        self.bounds = self._build_bounds()
        self.A_eq, self.b_eq, self.A_ub, self.b_ub = self._build_constraints()
        self.objectives = self._build_objectives()

    def ix(self, variable: str, r: int, t: int) -> int:
        return self.variable_names.index(variable) * self.rt + r * self.t + t

    def p_ix(self, r: int) -> int:
        return self.p_start + r

    def z_ix(self, r: int, t: int) -> int:
        # t is in 1..T-1, representing |net_t-net_(t-1)|.
        return self.z_start + r * (self.t - 1) + (t - 1)

    def _build_bounds(self) -> list[tuple[float, float | None]]:
        d = self.data
        bounds: list[tuple[float, float | None]] = [(0.0, None)] * self.n_vars
        for r in range(self.r):
            for t in range(self.t):
                # Operational priority: contemporaneous renewable first serves local load.
                # This rules out a financial cycle that discharges storage while exporting
                # renewable electricity in the same hour.
                direct = float(min(d.load[r, t], d.renewable[r, t]))
                bounds[self.ix("direct_renew", r, t)] = (direct, direct)
                bounds[self.ix("renew_charge", r, t)] = (0.0, float(d.max_charge[r]))
                # Grid charging is meaningful only during a local renewable deficit.
                # When renewable already covers the local load, allowing grid charging
                # alongside renewable export creates an accounting-only arbitrage loop.
                grid_charge_limit = float(d.max_charge[r]) if d.renewable[r, t] < d.load[r, t] else 0.0
                bounds[self.ix("grid_charge", r, t)] = (0.0, grid_charge_limit)
                bounds[self.ix("discharge", r, t)] = (0.0, float(d.max_discharge[r]))
                bounds[self.ix("grid_load", r, t)] = (0.0, float(d.max_import[r]))
                bounds[self.ix("grid_sell", r, t)] = (0.0, float(d.max_export[r]))
                bounds[self.ix("curtailment", r, t)] = (0.0, float(d.renewable[r, t]))
                bounds[self.ix("soc", r, t)] = (float(d.min_soc[r]), float(d.capacity[r]))
            # The final supplied hour is terminal settlement: no storage action.
            end = self.t - 1
            for variable in ("renew_charge", "grid_charge", "discharge"):
                bounds[self.ix(variable, r, end)] = (0.0, 0.0)
            bounds[self.ix("soc", r, end)] = (float(max(d.min_soc[r], d.initial_soc[r])), float(d.capacity[r]))
            bounds[self.p_ix(r)] = (0.0, None)
            for t in range(1, self.t):
                bounds[self.z_ix(r, t)] = (0.0, None)
        return bounds

    @staticmethod
    def _matrix(rows: list[int], cols: list[int], vals: list[float], n_rows: int, n_cols: int) -> sparse.csr_matrix:
        return sparse.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

    def _build_constraints(self) -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray]:
        d = self.data
        eq_rows: list[int] = []
        eq_cols: list[int] = []
        eq_vals: list[float] = []
        b_eq: list[float] = []
        row = 0
        for r in range(self.r):
            for t in range(self.t):
                # Renewable allocation: direct + renewable charge + sell + curtail = available renewable.
                for v in ("direct_renew", "renew_charge", "grid_sell", "curtailment"):
                    eq_rows.append(row); eq_cols.append(self.ix(v, r, t)); eq_vals.append(1.0)
                b_eq.append(float(d.renewable[r, t])); row += 1
                # Local facility load: direct renewable + battery discharge + grid-to-load = fixed load.
                for v in ("direct_renew", "discharge", "grid_load"):
                    eq_rows.append(row); eq_cols.append(self.ix(v, r, t)); eq_vals.append(1.0)
                b_eq.append(float(d.load[r, t])); row += 1
                # End-of-hour SOC equation. At t=0, prior SOC is the supplied initial state.
                eq_rows.append(row); eq_cols.append(self.ix("soc", r, t)); eq_vals.append(1.0)
                if t > 0:
                    eq_rows.append(row); eq_cols.append(self.ix("soc", r, t - 1)); eq_vals.append(-1.0)
                    rhs = 0.0
                else:
                    rhs = float(d.initial_soc[r])
                eq_rows.append(row); eq_cols.append(self.ix("renew_charge", r, t)); eq_vals.append(-float(d.eta_charge[r]))
                eq_rows.append(row); eq_cols.append(self.ix("grid_charge", r, t)); eq_vals.append(-float(d.eta_charge[r]))
                eq_rows.append(row); eq_cols.append(self.ix("discharge", r, t)); eq_vals.append(1.0 / float(d.eta_discharge[r]))
                b_eq.append(rhs); row += 1
        A_eq = self._matrix(eq_rows, eq_cols, eq_vals, row, self.n_vars)

        ub_rows: list[int] = []
        ub_cols: list[int] = []
        ub_vals: list[float] = []
        b_ub: list[float] = []
        row = 0
        for r in range(self.r):
            for t in range(self.t):
                # Grid import includes local supply and grid charging.
                for v in ("grid_load", "grid_charge"):
                    ub_rows.append(row); ub_cols.append(self.ix(v, r, t)); ub_vals.append(1.0)
                b_ub.append(float(d.max_import[r])); row += 1
                # Total charging power.
                for v in ("renew_charge", "grid_charge"):
                    ub_rows.append(row); ub_cols.append(self.ix(v, r, t)); ub_vals.append(1.0)
                b_ub.append(float(d.max_charge[r])); row += 1
                # Battery can only support residual local load, never export.
                ub_rows.extend((row, row))
                ub_cols.extend((self.ix("discharge", r, t), self.ix("direct_renew", r, t)))
                ub_vals.extend((1.0, 1.0))
                b_ub.append(float(d.load[r, t])); row += 1
                # Peak positive net grid import: grid load + grid charge - sale <= peak.
                for v, coef in (("grid_load", 1.0), ("grid_charge", 1.0), ("grid_sell", -1.0)):
                    ub_rows.append(row); ub_cols.append(self.ix(v, r, t)); ub_vals.append(coef)
                ub_rows.append(row); ub_cols.append(self.p_ix(r)); ub_vals.append(-1.0)
                b_ub.append(0.0); row += 1
                if t > 0:
                    # net_t - net_(t-1) <= z_t and its reverse.
                    for sign in (1.0, -1.0):
                        for time, time_coef in ((t, sign), (t - 1, -sign)):
                            for v, coef in (("grid_load", 1.0), ("grid_charge", 1.0), ("grid_sell", -1.0)):
                                ub_rows.append(row); ub_cols.append(self.ix(v, r, time)); ub_vals.append(time_coef * coef)
                        ub_rows.append(row); ub_cols.append(self.z_ix(r, t)); ub_vals.append(-1.0)
                        b_ub.append(0.0); row += 1
        A_ub = self._matrix(ub_rows, ub_cols, ub_vals, row, self.n_vars)
        return A_eq, np.asarray(b_eq), A_ub, np.asarray(b_ub)

    def _build_objectives(self) -> dict[str, np.ndarray]:
        d = self.data
        objectives = {name: np.zeros(self.n_vars, dtype=float) for name in ("cost", "carbon", "curtail", "peak", "tv")}
        for r in range(self.r):
            for t in range(self.t):
                objectives["cost"][self.ix("grid_load", r, t)] = d.price[r, t]
                objectives["cost"][self.ix("grid_charge", r, t)] = d.price[r, t]
                objectives["cost"][self.ix("grid_sell", r, t)] = -d.sell_price[r, t]
                objectives["carbon"][self.ix("grid_load", r, t)] = d.carbon[r, t]
                objectives["carbon"][self.ix("grid_charge", r, t)] = d.carbon[r, t]
                objectives["curtail"][self.ix("curtailment", r, t)] = 1.0
                if t > 0:
                    objectives["tv"][self.z_ix(r, t)] = 1.0
            # A 1 MW floor avoids division by zero for regions with no no-storage import peak.
            objectives["peak"][self.p_ix(r)] = 1.0 / max(float(self.no_storage_peak[r]), 1.0)
        return objectives

    @property
    def no_storage_peak(self) -> np.ndarray:
        direct = np.minimum(self.data.load, self.data.renewable)
        surplus = np.maximum(self.data.renewable - direct, 0.0)
        sell = np.minimum(surplus, self.data.max_export[:, None])
        grid = np.maximum(self.data.load - direct, 0.0)
        return np.maximum((grid - sell).max(axis=1), 0.0)

    def solve(self, objective: np.ndarray, caps: Iterable[tuple[np.ndarray, float]] = ()) -> OptimizeResult:
        matrices = [self.A_ub]
        rhs = [self.b_ub]
        for vector, value in caps:
            matrices.append(sparse.csr_matrix(vector.reshape(1, -1)))
            rhs.append(np.asarray([value], dtype=float))
        result = linprog(
            c=objective,
            A_ub=sparse.vstack(matrices, format="csr"),
            b_ub=np.concatenate(rhs),
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method="highs",
            options={"primal_feasibility_tolerance": float(self.config["solver_feasibility_tolerance"]),
                     "dual_feasibility_tolerance": float(self.config["solver_feasibility_tolerance"])},
        )
        if not result.success:
            raise RuntimeError(f"HiGHS failed: {result.message}")
        return result

    def tolerance(self, value: float) -> float:
        return float(self.config["lexicographic_absolute_tolerance"]) + float(self.config["lexicographic_relative_tolerance"]) * max(abs(value), 1.0)

    def solve_lexicographic(self, sequence: list[str], initial_caps: Iterable[tuple[np.ndarray, float]] = ()) -> OptimizeResult:
        caps = list(initial_caps)
        result: OptimizeResult | None = None
        for name in sequence:
            result = self.solve(self.objectives[name], caps)
            value = float(np.dot(self.objectives[name], result.x))
            caps.append((self.objectives[name], value + self.tolerance(value)))
        assert result is not None
        return result

    def arrays(self, x: np.ndarray) -> dict[str, np.ndarray]:
        arrays = {}
        for i, name in enumerate(self.variable_names):
            arrays[name] = x[i * self.rt:(i + 1) * self.rt].reshape(self.r, self.t)
        arrays["peak_variable"] = x[self.p_start:self.p_start + self.r]
        arrays["net_grid"] = arrays["grid_load"] + arrays["grid_charge"] - arrays["grid_sell"]
        arrays["grid_purchase"] = arrays["grid_load"] + arrays["grid_charge"]
        return arrays

    def metrics(self, x: np.ndarray) -> dict[str, float]:
        a = self.arrays(x)
        purchase_cost = float(np.sum(a["grid_purchase"] * self.data.price))
        sale_revenue = float(np.sum(a["grid_sell"] * self.data.sell_price))
        net = a["net_grid"]
        total_renewable = float(self.data.renewable.sum())
        return {
            "NetCost_CNY": purchase_cost - sale_revenue,
            "PurchaseCost_CNY": purchase_cost,
            "SaleRevenue_CNY": sale_revenue,
            "Carbon_tCO2": float(np.sum(a["grid_purchase"] * self.data.carbon)),
            "GridPurchase_MWh": float(a["grid_purchase"].sum()),
            "GridSell_MWh": float(a["grid_sell"].sum()),
            "Curtailment_MWh": float(a["curtailment"].sum()),
            "RenewableUtilization": float(1.0 - a["curtailment"].sum() / total_renewable),
            "PeakScore": float(np.dot(self.objectives["peak"], x)),
            "TotalVariation_MW": float(np.abs(np.diff(net, axis=1)).sum()),
            "MeanRegionalNetGridStd_MW": float(np.std(net, axis=1).mean()),
            "Charge_MWh": float((a["renew_charge"] + a["grid_charge"]).sum()),
            "RenewableCharge_MWh": float(a["renew_charge"].sum()),
            "GridCharge_MWh": float(a["grid_charge"].sum()),
            "Discharge_MWh": float(a["discharge"].sum()),
            "MaxRegionalNetImport_MW": float(np.maximum(net, 0.0).max()),
        }


def no_storage_solution(model: StorageLP) -> np.ndarray:
    """Deterministic feasible reference with storage actions set to zero."""
    d = model.data
    x = np.zeros(model.n_vars, dtype=float)
    direct = np.minimum(d.load, d.renewable)
    surplus = np.maximum(d.renewable - direct, 0.0)
    sell = np.minimum(surplus, d.max_export[:, None])
    curtail = surplus - sell
    grid_load = d.load - direct
    for r in range(model.r):
        for t in range(model.t):
            x[model.ix("direct_renew", r, t)] = direct[r, t]
            x[model.ix("grid_load", r, t)] = grid_load[r, t]
            x[model.ix("grid_sell", r, t)] = sell[r, t]
            x[model.ix("curtailment", r, t)] = curtail[r, t]
            x[model.ix("soc", r, t)] = d.initial_soc[r]
        x[model.p_ix(r)] = max(float((grid_load[r] - sell[r]).max()), 0.0)
        for t in range(1, model.t):
            x[model.z_ix(r, t)] = abs((grid_load[r, t] - sell[r, t]) - (grid_load[r, t - 1] - sell[r, t - 1]))
    return x


def validate_solution(model: StorageLP, x: np.ndarray, scenario: str, source: str) -> list[dict]:
    a = model.arrays(x)
    d = model.data
    tol = float(model.config["solver_feasibility_tolerance"]) * 20
    rows: list[dict] = []
    def add(check: str, violation: float) -> None:
        rows.append({"scenario": scenario, "source": source, "check": check,
                     "maximum_violation": float(max(violation, 0.0)), "passed": bool(violation <= tol)})
    renewable_balance = np.abs(a["direct_renew"] + a["renew_charge"] + a["grid_sell"] + a["curtailment"] - d.renewable).max()
    local_balance = np.abs(a["direct_renew"] + a["discharge"] + a["grid_load"] - d.load).max()
    soc_prev = np.concatenate([d.initial_soc[:, None], a["soc"][:, :-1]], axis=1)
    soc_rhs = soc_prev + d.eta_charge[:, None] * (a["renew_charge"] + a["grid_charge"]) - a["discharge"] / d.eta_discharge[:, None]
    add("renewable_balance", float(renewable_balance))
    add("local_load_balance", float(local_balance))
    add("soc_recurrence", float(np.abs(a["soc"] - soc_rhs).max()))
    add("soc_lower_bound", float((d.min_soc[:, None] - a["soc"]).max()))
    add("soc_upper_bound", float((a["soc"] - d.capacity[:, None]).max()))
    add("terminal_soc", float((d.initial_soc - a["soc"][:, -1]).max()))
    add("charge_power", float((a["renew_charge"] + a["grid_charge"] - d.max_charge[:, None]).max()))
    add("discharge_power", float((a["discharge"] - d.max_discharge[:, None]).max()))
    add("grid_import", float((a["grid_purchase"] - d.max_import[:, None]).max()))
    add("grid_export", float((a["grid_sell"] - d.max_export[:, None]).max()))
    add("battery_not_exported", float((a["discharge"] + a["direct_renew"] - d.load).max()))
    add("terminal_storage_inactive", float(max(a["renew_charge"][:, -1].max(), a["grid_charge"][:, -1].max(), a["discharge"][:, -1].max())))
    add("simultaneous_grid_buy_sell", float(np.minimum(a["grid_purchase"], a["grid_sell"]).max()))
    add("simultaneous_charge_discharge", float(np.minimum(a["renew_charge"] + a["grid_charge"], a["discharge"]).max()))
    return rows


def frontier(model: StorageLP, config: dict, prefix: str = "base") -> list[Candidate]:
    log(f"[{prefix}] Solving lexicographic endpoint: economic.")
    economic = model.solve_lexicographic(["cost", "carbon", "curtail", "peak", "tv"])
    log(f"[{prefix}] Solving lexicographic endpoint: renewable utilization.")
    renewable = model.solve_lexicographic(["curtail", "carbon", "cost", "peak", "tv"])
    log(f"[{prefix}] Solving lexicographic endpoint: low carbon.")
    carbon = model.solve_lexicographic(["carbon", "cost", "curtail", "peak", "tv"])
    log(f"[{prefix}] Solving lexicographic endpoint: grid-friendly.")
    grid = model.solve_lexicographic(["peak", "tv", "curtail", "carbon", "cost"])
    candidates = [
        Candidate("economic_endpoint", economic, model.metrics(economic.x), "endpoint"),
        Candidate("renewable_endpoint", renewable, model.metrics(renewable.x), "endpoint"),
        Candidate("carbon_endpoint", carbon, model.metrics(carbon.x), "endpoint"),
        Candidate("grid_endpoint", grid, model.metrics(grid.x), "endpoint"),
    ]
    econ_cost = candidates[0].metrics["NetCost_CNY"]
    for anchor, objective_sequence in ((candidates[1], ["curtail", "carbon", "peak", "tv"]),
                                       (candidates[3], ["peak", "tv", "curtail", "carbon"])):
        delta = max(anchor.metrics["NetCost_CNY"] - econ_cost, 0.0)
        if delta <= model.tolerance(econ_cost):
            log(f"[{prefix}] {anchor.name}: no material cost span; epsilon candidates skipped.")
            continue
        for alpha in config["epsilon_alphas"]:
            cap = econ_cost + float(alpha) * delta
            log(f"[{prefix}] Solving {anchor.name} epsilon alpha={alpha:g}.")
            result = model.solve_lexicographic(objective_sequence, [(model.objectives["cost"], cap)])
            name = f"{anchor.name.replace('_endpoint', '')}_epsilon_{int(round(alpha * 100)):02d}"
            candidates.append(Candidate(name, result, model.metrics(result.x), "epsilon"))
    return candidates


OBJECTIVE_COLUMNS = ["NetCost_CNY", "Carbon_tCO2", "Curtailment_MWh", "PeakScore", "TotalVariation_MW"]


def is_same_metrics(a: Candidate, b: Candidate, config: dict) -> bool:
    rel = float(config["duplicate_relative_tolerance"])
    absolute = float(config["duplicate_absolute_tolerance"])
    return all(math.isclose(a.metrics[c], b.metrics[c], rel_tol=rel, abs_tol=absolute) for c in OBJECTIVE_COLUMNS)


def dominates(a: Candidate, b: Candidate, config: dict) -> bool:
    rel = float(config["duplicate_relative_tolerance"])
    absolute = float(config["duplicate_absolute_tolerance"])
    no_worse = all(a.metrics[c] <= b.metrics[c] + absolute + rel * max(abs(b.metrics[c]), 1.0) for c in OBJECTIVE_COLUMNS)
    strictly_better = any(a.metrics[c] < b.metrics[c] - absolute - rel * max(abs(b.metrics[c]), 1.0) for c in OBJECTIVE_COLUMNS)
    return no_worse and strictly_better


def select_candidates(candidates: list[Candidate], config: dict) -> list[Candidate]:
    unique: list[Candidate] = []
    for candidate in candidates:
        if not any(is_same_metrics(candidate, kept, config) for kept in unique):
            unique.append(candidate)
    for candidate in unique:
        candidate.nondominated = not any(other is not candidate and dominates(other, candidate, config) for other in unique)
    front = [c for c in unique if c.nondominated]
    if not front:
        raise RuntimeError("No non-dominated candidate remained after filtering.")
    labels: dict[str, Candidate] = {
        "economic": min(front, key=lambda c: (c.metrics["NetCost_CNY"], c.name)),
        "renewable": min(front, key=lambda c: (c.metrics["Curtailment_MWh"], c.metrics["NetCost_CNY"], c.name)),
        "grid_friendly": min(front, key=lambda c: (c.metrics["PeakScore"], c.metrics["TotalVariation_MW"], c.name)),
    }
    ranges = {col: (min(c.metrics[col] for c in front), max(c.metrics[col] for c in front)) for col in OBJECTIVE_COLUMNS}
    def ideal_distance(candidate: Candidate) -> tuple[float, float, str]:
        squared = 0.0
        used = 0
        for col, (low, high) in ranges.items():
            if high - low > 1e-10:
                squared += ((candidate.metrics[col] - low) / (high - low)) ** 2
                used += 1
        return (squared / max(used, 1), candidate.metrics["NetCost_CNY"], candidate.name)
    labels["balanced"] = min(front, key=ideal_distance)
    for label, candidate in labels.items():
        existing = candidate.labels.split(";") if candidate.labels else []
        candidate.labels = ";".join(existing + [label])
    return unique


def candidate_frame(candidates: list[Candidate], scenario_group: str) -> pd.DataFrame:
    rows = []
    for c in candidates:
        rows.append({"scenario_group": scenario_group, "scenario": c.name, "source": c.source,
                     "is_nondominated": c.nondominated, "recommended_labels": c.labels, **c.metrics,
                     "solver_status": int(c.result.status), "solver_message": c.result.message})
    return pd.DataFrame(rows).sort_values(["is_nondominated", "NetCost_CNY", "scenario"], ascending=[False, True, True])


def dispatch_frame(model: StorageLP, candidate: Candidate) -> pd.DataFrame:
    a = model.arrays(candidate.result.x)
    rows = []
    for r, region in enumerate(model.data.regions):
        for t, hour in enumerate(model.data.hours):
            rows.append({
                "Hour": int(hour), "Region": region, "FacilityLoad_MW": model.data.load[r, t],
                "AvailableRenewable_MW": model.data.renewable[r, t],
                "DirectRenewable_MW": a["direct_renew"][r, t], "RenewableCharge_MW": a["renew_charge"][r, t],
                "GridCharge_MW": a["grid_charge"][r, t], "DischargePower_MW": a["discharge"][r, t],
                "GridLoad_MW": a["grid_load"][r, t], "GridPurchase_MW": a["grid_purchase"][r, t],
                "GridSell_MW": a["grid_sell"][r, t], "Curtailment_MW": a["curtailment"][r, t],
                "NetGridImport_MW": a["net_grid"][r, t], "SOC_MWh": a["soc"][r, t],
                "ElectricityPrice_CNY_per_MWh": model.data.price[r, t],
                "SellPrice_CNY_per_MWh": model.data.sell_price[r, t],
                "CarbonIntensity_tCO2_per_MWh": model.data.carbon[r, t],
            })
    return pd.DataFrame(rows)


def region_summary(model: StorageLP, candidate: Candidate) -> pd.DataFrame:
    dispatch = dispatch_frame(model, candidate)
    dispatch["purchase_cost"] = dispatch["GridPurchase_MW"] * dispatch["ElectricityPrice_CNY_per_MWh"]
    dispatch["sale_revenue"] = dispatch["GridSell_MW"] * dispatch["SellPrice_CNY_per_MWh"]
    dispatch["carbon"] = dispatch["GridPurchase_MW"] * dispatch["CarbonIntensity_tCO2_per_MWh"]
    rows = []
    for region, item in dispatch.groupby("Region", sort=False):
        net = item["NetGridImport_MW"].to_numpy()
        rows.append({
            "Region": region, "NetCost_CNY": item["purchase_cost"].sum() - item["sale_revenue"].sum(),
            "PurchaseCost_CNY": item["purchase_cost"].sum(), "SaleRevenue_CNY": item["sale_revenue"].sum(),
            "Carbon_tCO2": item["carbon"].sum(), "GridPurchase_MWh": item["GridPurchase_MW"].sum(),
            "GridSell_MWh": item["GridSell_MW"].sum(), "Curtailment_MWh": item["Curtailment_MW"].sum(),
            "RenewableUtilization": 1.0 - item["Curtailment_MW"].sum() / item["AvailableRenewable_MW"].sum(),
            "PeakNetImport_MW": max(net.max(), 0.0), "NetGridStd_MW": net.std(),
            "TotalVariation_MW": np.abs(np.diff(net)).sum(), "Charge_MWh": item["RenewableCharge_MW"].sum() + item["GridCharge_MW"].sum(),
            "Discharge_MWh": item["DischargePower_MW"].sum(), "TerminalSOC_MWh": item["SOC_MWh"].iloc[-1],
        })
    return pd.DataFrame(rows)


def attachment_baseline(data: Inputs) -> pd.DataFrame:
    base = pd.read_csv(PROCESSED / "fact_region_hour_baseline.csv")
    hourly = pd.read_csv(PROCESSED / "fact_region_hour_input.csv")
    base = base.merge(hourly[["Hour", "Region", "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh"]], on=["Hour", "Region"])
    base = base[base["Hour"].isin(data.hours)]
    purchase_cost = (base["GridPurchase_MW"] * base["ElectricityPrice_CNY_per_MWh"]).sum()
    sale_revenue = (base["GridSell_MW"] * base["SellPrice_CNY_per_MWh"]).sum()
    renewable_available = pd.read_csv(PROCESSED / "fact_region_hour_input.csv")
    renewable_available = renewable_available[renewable_available["Hour"].isin(data.hours)]["AvailableRenewable_MW"].sum()
    return pd.DataFrame([{
        "scenario": "attachment_baseline", "source": "attachment", "NetCost_CNY": purchase_cost - sale_revenue,
        "PurchaseCost_CNY": purchase_cost, "SaleRevenue_CNY": sale_revenue,
        "Carbon_tCO2": base["CarbonEmission_tCO2"].sum(), "GridPurchase_MWh": base["GridPurchase_MW"].sum(),
        "GridSell_MWh": base["GridSell_MW"].sum(), "Curtailment_MWh": base["Curtailment_MW"].sum(),
        "RenewableUtilization": 1.0 - base["Curtailment_MW"].sum() / renewable_available,
        "MaxRegionalNetImport_MW": max((base["GridPurchase_MW"] - base["GridSell_MW"]).max(), 0.0),
        "TotalVariation_MW": np.nan, "MeanRegionalNetGridStd_MW": np.nan,
    }])


def write_figures(model: StorageLP, selected: list[Candidate], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib unavailable; figures skipped.")
        return
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    hours = model.data.hours
    fig, ax = plt.subplots(figsize=(12, 5))
    for candidate in selected:
        net = model.arrays(candidate.result.x)["net_grid"].sum(axis=0)
        ax.plot(hours, net, linewidth=1.0, label=candidate.labels or candidate.name)
    ax.set_title("System net grid exchange by recommended strategy")
    ax.set_xlabel("Hour"); ax.set_ylabel("MW; positive = import"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figure_dir / "system_net_grid.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(model.r, 1, figsize=(12, 1.8 * model.r), sharex=True)
    reference = selected[0]
    soc = model.arrays(reference.result.x)["soc"]
    for r, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(hours, soc[r], color="#1f77b4", linewidth=0.9)
        ax.axhline(model.data.initial_soc[r], color="#d62728", linestyle="--", linewidth=0.7)
        ax.set_ylabel(model.data.regions[r])
        ax.grid(alpha=0.2)
    axes[0].set_title(f"SOC: {reference.labels or reference.name}")
    axes[-1].set_xlabel("Hour")
    fig.tight_layout(); fig.savefig(figure_dir / "soc_by_region.png", dpi=150); plt.close(fig)


def run_group(data: Inputs, config: dict, output: Path, name: str, price_multiplier: float = 1.0,
              export_multiplier: float = 1.0, write_dispatch: bool = True,
              write_plots: bool = True) -> tuple[pd.DataFrame, list[Candidate], StorageLP, list[dict]]:
    changed = Inputs(**{**data.__dict__,
                        "sell_price": np.minimum(data.sell_price * price_multiplier, data.price),
                        "max_export": data.max_export * export_multiplier})
    model = StorageLP(changed, config)
    reference_x = no_storage_solution(model)
    reference_candidate = Candidate("no_storage_reference", OptimizeResult(x=reference_x, status=0, message="deterministic reference"), model.metrics(reference_x), "reference")
    candidates = frontier(model, config, name)
    candidates = select_candidates(candidates, config)
    all_candidates = [reference_candidate] + candidates
    checks: list[dict] = []
    for candidate in all_candidates:
        checks.extend(validate_solution(model, candidate.result.x, candidate.name, candidate.source))
    summary = candidate_frame(all_candidates, name)
    summary["sell_price_multiplier"] = price_multiplier
    summary["export_capacity_multiplier"] = export_multiplier
    if write_dispatch:
        selected = [c for c in candidates if c.nondominated]
        for candidate in selected:
            dispatch_frame(model, candidate).to_csv(output / f"region_hour_dispatch_{name}_{candidate.name}.csv", index=False, encoding="utf-8-sig")
            region_summary(model, candidate).to_csv(output / f"region_summary_{name}_{candidate.name}.csv", index=False, encoding="utf-8-sig")
        if write_plots:
            write_figures(model, selected, output)
    return summary, candidates, model, checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 3 fixed-load storage co-optimization")
    parser.add_argument("--smoke-hours", type=int, default=None, help="Use the first N hours only; validation use only.")
    parser.add_argument("--include-sensitivity", action="store_true", help="Run sell-price and export-capacity sensitivity grid.")
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate PNG figures.")
    args = parser.parse_args()

    config = read_config()
    data = read_inputs(args.smoke_hours)
    output = HERE / "储能结果" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    (output / "effective_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    static = static_checks(data, float(config["solver_feasibility_tolerance"]))
    static.to_csv(output / "static_input_checks.csv", index=False, encoding="utf-8-sig")
    if not static["passed"].all():
        raise RuntimeError("Fixed-load precheck failed; see static_input_checks.csv.")

    summary, candidates, model, checks = run_group(
        data, config, output, "base", write_dispatch=True, write_plots=not args.skip_plots
    )
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(output / "feasibility_checks.csv", index=False, encoding="utf-8-sig")

    base_reference = Candidate("no_storage_reference", OptimizeResult(x=no_storage_solution(model), status=0, message="deterministic reference"), model.metrics(no_storage_solution(model)), "reference")
    comparison_rows = [attachment_baseline(data), candidate_frame([base_reference] + candidates, "base")]
    comparison = pd.concat(comparison_rows, ignore_index=True, sort=False)
    comparison.to_csv(output / "comparison_to_baseline.csv", index=False, encoding="utf-8-sig")

    if not pd.DataFrame(checks)["passed"].all():
        raise RuntimeError("A feasibility check failed; results retained for diagnosis but are not valid.")

    if args.include_sensitivity:
        sensitivity_summaries = []
        sensitivity_checks = []
        for price_multiplier in config["sensitivity_price_multipliers"]:
            for export_multiplier in config["sensitivity_export_multipliers"]:
                if math.isclose(price_multiplier, 1.0) and math.isclose(export_multiplier, 1.0):
                    continue
                name = f"sell_{price_multiplier:g}_export_{export_multiplier:g}".replace(".", "p")
                log(f"Running sensitivity: {name}")
                group_summary, _, _, group_checks = run_group(
                    data, config, output, name, float(price_multiplier), float(export_multiplier), write_dispatch=False
                )
                sensitivity_summaries.append(group_summary)
                sensitivity_checks.extend(group_checks)
        if sensitivity_summaries:
            pd.concat(sensitivity_summaries, ignore_index=True).to_csv(output / "sell_export_sensitivity.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(sensitivity_checks).to_csv(output / "sell_export_sensitivity_checks.csv", index=False, encoding="utf-8-sig")
            if not pd.DataFrame(sensitivity_checks)["passed"].all():
                raise RuntimeError("A sensitivity feasibility check failed.")

    recommended = summary[summary["recommended_labels"].fillna("") != ""]
    log(f"Completed. Output: {output}")
    log(f"Base candidates={len(summary)}, recommended={len(recommended)}, non-dominated={int(summary['is_nondominated'].sum())}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # concise user-facing failure while preserving traceback for diagnosis
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
