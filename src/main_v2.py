"""
main_v2.py
----------
Full-dataset experiment runner for the final airline route-frequency / fleet
planning MILP.

This file is consistent with:
- data_loader.py
- model_v2_final.py

This script runs ONLY FULL-DATASET experiments. It creates 10 meaningful
experiments and exports compact output CSV files.
"""

import os
import sys
import copy
from typing import Dict, Any, List, Tuple

import pandas as pd

# Keep this only if your Gurobi license path is exactly this.
# If Gurobi already works on your computer, this is harmless.
os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\ed024981\gurobi.lic")

# Make sure imports come from the same src folder as this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, print_data_summary
from model_v2_final import build_and_solve


# =============================================================================
# User-facing run settings
# =============================================================================

# Run the entire uploaded dataset. Do not use small/medium/large filters here.
FULL_LOAD_KWARGS = dict(
    n_routes=None,
    n_months=None,
    n_aircraft=None,
    K_ct_value=1,
)

# Solver settings used across experiments.
TIME_LIMIT = 3600
MIP_GAP = 0.00
SHOW_GUROBI_LOG = True

# Realistic default fixed-cost assumptions for the final model.
# The data files do not contain these costs, so they are scenario assumptions.
BASE_AIRCRAFT_FC = 50_000.0       # per deployed physical aircraft per period
BASE_ROUTE_RC = 25_000.0          # per active route-period
BASE_ROUTE_SC = 150_000.0         # one-time route opening/startup cost

# Output folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(OUTPUT_DIR, "full_dataset_10_experiment_summary.csv")
FLIGHT_CSV = os.path.join(OUTPUT_DIR, "full_dataset_10_experiment_flight_schedule.csv")
Q_CSV = os.path.join(OUTPUT_DIR, "full_dataset_10_experiment_aircraft_deployment.csv")

# =============================================================================
# Helper functions
# =============================================================================

def clone_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy data so each experiment can safely modify costs/parameters."""
    return copy.deepcopy(data)


def apply_cost_assumptions(
    data: Dict[str, Any],
    aircraft_fc: float = BASE_AIRCRAFT_FC,
    route_rc: float = BASE_ROUTE_RC,
    route_sc: float = BASE_ROUTE_SC,
) -> Dict[str, Any]:
    """
    Apply fixed-cost assumptions consistently.

    FC[a]    : cost per deployed physical aircraft of type a in a period.
    RC[r,t]  : recurring fixed cost for keeping route r active in period t.
    SC[r]    : startup/opening cost when route r is newly opened.
    """
    routes = data["routes"]
    periods = data["periods"]
    aircraft = data["aircraft"]

    data["FC"] = {a: float(aircraft_fc) for a in aircraft}
    data["RC"] = {(r, t): float(route_rc) for r in routes for t in periods}
    data["SC"] = {r: float(route_sc) for r in routes}
    return data


def set_oversupply_factor(data: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """Set eta[r,t], the maximum allowed seat oversupply multiplier."""
    data["oversupply_factor"] = float(factor)
    data["eta"] = {(r, t): float(factor) for r in data["routes"] for t in data["periods"]}
    return data


def scale_hub_slots(data: Dict[str, Any], multiplier: float) -> Dict[str, Any]:
    """Scale hub slot capacity, useful for airport capacity stress tests."""
    data["HubSlot"] = {
        t: max(0, int(round(v * float(multiplier))))
        for t, v in data["HubSlot"].items()
    }
    return data

def run_case(data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Consistent wrapper around build_and_solve()."""
    defaults = dict(
        delta=0.0,
        alpha=1.0,
        N_min=2,
        maintenance_window=5,
        max_active_periods=3,
        time_limit=TIME_LIMIT,
        mip_gap=MIP_GAP,
        verbose=SHOW_GUROBI_LOG,
    )
    defaults.update(kwargs)
    return build_and_solve(data, **defaults)


def obj_text(res: Dict[str, Any]) -> str:
    if res.get("obj_value") is None:
        return res.get("status", "NO_RESULT")
    return f"${res['obj_value']:,.2f}"

def summarize_result(
    experiment_id: int,
    experiment_name: str,
    description: str,
    res: Dict[str, Any],
    baseline_obj: float = None,
) -> Dict[str, Any]:
    """Create one compact summary row for an experiment."""
    total_flights = sum(res.get("x_vals", {}).values())
    active_route_periods = len(res.get("y_vals", {}))
    route_openings = len(res.get("z_vals", {}))
    sum_q = sum(res.get("q_vals", {}).values())
    total_aircraft_hours = sum(res.get("aircraft_hours_vals", {}).values())

    obj_value = res.get("obj_value")
    obj_change = None
    obj_change_pct = None
    if baseline_obj is not None and obj_value is not None:
        obj_change = obj_value - baseline_obj
        obj_change_pct = 100.0 * obj_change / abs(baseline_obj) if baseline_obj != 0 else None

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "description": description,
        "status": res.get("status"),
        "objective_usd": obj_value,
        "objective_change_vs_baseline_usd": obj_change,
        "objective_change_vs_baseline_pct": obj_change_pct,
        "runtime_sec": res.get("runtime"),
        "n_vars": res.get("n_vars"),
        "n_constrs": res.get("n_constrs"),
        "delta": res.get("delta"),
        "alpha": res.get("alpha"),
        "N_min": res.get("N_min"),
        "maintenance_window": res.get("maintenance_window"),
        "max_active_periods": res.get("max_active_periods"),
        "n_neg_pi_combos": res.get("n_neg_pi_combos"),
        "total_flights": total_flights,
        "active_route_periods": active_route_periods,
        "route_openings": route_openings,
        "sum_q": sum_q,
        "total_aircraft_hours": total_aircraft_hours,
    }


def build_flight_rows(experiment_id: int, experiment_name: str, res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for (route, month, aircraft), flights in sorted(res.get("x_vals", {}).items()):
        rows.append({
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "route": route,
            "month": month,
            "aircraft": aircraft,
            "flights": flights,
        })
    return rows


def build_q_rows(experiment_id: int, experiment_name: str, res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for (aircraft, month), q_val in sorted(res.get("q_vals", {}).items()):
        rows.append({
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "aircraft": aircraft,
            "month": month,
            "q_deployed": q_val,
            "assigned_hours": res.get("aircraft_hours_vals", {}).get((aircraft, month), 0.0),
        })
    return rows


def print_summary_table(summary_rows: List[Dict[str, Any]]) -> None:
    """Print compact full-dataset experiment table."""
    print()
    print("=" * 138)
    print("FULL DATASET - 10 MEANINGFUL EXPERIMENTS")
    print("=" * 138)
    print(
        f"{'ID':>2}  {'Experiment':<32} {'Status':>10} {'Objective (USD)':>20} "
        f"{'Change %':>9} {'Flights':>8} {'Rts':>5} {'Open':>5} {'sum_q':>6} "
        f"{'Vars':>6} {'Constr':>7} {'Time':>8}"
    )
    print("-" * 138)

    for row in summary_rows:
        obj = "N/A" if row["objective_usd"] is None else f"${row['objective_usd']:,.2f}"
        pct = "-" if row["objective_change_vs_baseline_pct"] is None else f"{row['objective_change_vs_baseline_pct']:+.2f}%"
        print(
            f"{row['experiment_id']:>2}  {row['experiment_name']:<32} {row['status']:>10} {obj:>20} "
            f"{pct:>9} {row['total_flights']:>8} {row['active_route_periods']:>5} "
            f"{row['route_openings']:>5} {row['sum_q']:>6} {row['n_vars']:>6} "
            f"{row['n_constrs']:>7} {row['runtime_sec']:>7.3f}s"
        )
    print("=" * 138)


def print_top_route_periods(res: Dict[str, Any], top_n: int = 20) -> None:
    """Print a compact view of the largest route-period decisions for one solution."""
    route_period_totals: Dict[Tuple[str, int], int] = {}
    for (r, t, _a), v in res.get("x_vals", {}).items():
        route_period_totals[(r, t)] = route_period_totals.get((r, t), 0) + v

    top_items = sorted(route_period_totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    print()
    print(f"Top {top_n} route-period flight totals for baseline:")
    print("-" * 70)
    for (r, t), total in top_items:
        print(f"  {r:<15} Month {t:>2} -> {total:>4} flights")


# =============================================================================
# Main experiment runner
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 90)
    print("# FULL DATASET EXPERIMENT RUNNER")
    print("# This script uses the full uploaded dataset for every experiment.")
    print("#" * 90)

    base_full = load_data(**FULL_LOAD_KWARGS)
    base_full = apply_cost_assumptions(
        base_full,
        aircraft_fc=BASE_AIRCRAFT_FC,
        route_rc=BASE_ROUTE_RC,
        route_sc=BASE_ROUTE_SC,
    )
    print_data_summary(base_full)

    # ------------------------------------------------------------------
    # Define 10 full-dataset experiments.
    # Each experiment uses the full dataset; only parameters/scenario settings change.
    # ------------------------------------------------------------------
    experiments = [
        {
            "name": "Realistic baseline",
            "description": "Baseline with positive aircraft, recurring route, and startup costs.",
            "data_fn": lambda: clone_data(base_full),
            "solve_kwargs": dict(delta=0.0, alpha=1.0, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "Fuel shock +10%",
            "description": "Moderate fuel-price shock.",
            "data_fn": lambda: clone_data(base_full),
            "solve_kwargs": dict(delta=0.10, alpha=1.0, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "Fuel shock +25%",
            "description": "Severe fuel-price shock.",
            "data_fn": lambda: clone_data(base_full),
            "solve_kwargs": dict(delta=0.25, alpha=1.0, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "Capacity shortage alpha=0.80",
            "description": "Reduced available aircraft-hour capacity.",
            "data_fn": lambda: clone_data(base_full),
            "solve_kwargs": dict(delta=0.0, alpha=0.80, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "Capacity expansion alpha=1.20",
            "description": "Expanded aircraft-hour capacity.",
            "data_fn": lambda: clone_data(base_full),
            "solve_kwargs": dict(delta=0.0, alpha=1.20, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "High aircraft fixed cost",
            "description": "Aircraft deployment cost increased to 100,000 per aircraft-period.",
            "data_fn": lambda: apply_cost_assumptions(
                clone_data(base_full),
                aircraft_fc=100_000.0,
                route_rc=BASE_ROUTE_RC,
                route_sc=BASE_ROUTE_SC,
            ),
            "solve_kwargs": dict(delta=0.0, alpha=1.0, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "High route fixed costs",
            "description": "Recurring route cost and startup cost are increased.",
            "data_fn": lambda: apply_cost_assumptions(
                clone_data(base_full),
                aircraft_fc=BASE_AIRCRAFT_FC,
                route_rc=75_000.0,
                route_sc=300_000.0,
            ),
            "solve_kwargs": dict(delta=0.0, alpha=1.0, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "Tight demand cap eta=1.05",
            "description": "Offered seats are limited to 105% of estimated demand.",
            "data_fn": lambda: set_oversupply_factor(clone_data(base_full), 1.05),
            "solve_kwargs": dict(delta=0.0, alpha=1.0, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "Category coverage K=2",
            "description": "Each route category requires two active routes per period when possible.",
            "data_fn": lambda: apply_cost_assumptions(
                load_data(K_ct_value=2),
                aircraft_fc=BASE_AIRCRAFT_FC,
                route_rc=BASE_ROUTE_RC,
                route_sc=BASE_ROUTE_SC,
            ),
            "solve_kwargs": dict(delta=0.0, alpha=1.0, N_min=2, maintenance_window=5, max_active_periods=3),
        },
        {
            "name": "Strict maintenance 2-in-5",
            "description": "Rolling maintenance permits only two active deployment periods in each five-period window.",
            "data_fn": lambda: clone_data(base_full),
            "solve_kwargs": dict(delta=0.0, alpha=1.0, N_min=2, maintenance_window=5, max_active_periods=2),
        },
    ]

    summary_rows: List[Dict[str, Any]] = []
    flight_rows: List[Dict[str, Any]] = []
    q_rows: List[Dict[str, Any]] = []
    baseline_obj = None
    baseline_result = None

    for i, exp in enumerate(experiments, start=1):
        print("\n" + "#" * 90)
        print(f"# EXPERIMENT {i:02d}: {exp['name']}")
        print(f"# {exp['description']}")
        print("#" * 90)

        try:
            exp_data = exp["data_fn"]()
            res = run_case(exp_data, **exp["solve_kwargs"])
            print(f"  Result: {res['status']} | Objective: {obj_text(res)} | Runtime: {res['runtime']:.3f}s")

            if i == 1 and res.get("obj_value") is not None:
                baseline_obj = res["obj_value"]
                baseline_result = res

            summary = summarize_result(i, exp["name"], exp["description"], res, baseline_obj)
            summary_rows.append(summary)
            flight_rows.extend(build_flight_rows(i, exp["name"], res))
            q_rows.extend(build_q_rows(i, exp["name"], res))

        except Exception as exc:
            print(f"  ERROR in experiment {i}: {exc}")
            summary_rows.append({
                "experiment_id": i,
                "experiment_name": exp["name"],
                "description": exp["description"],
                "status": "ERROR",
                "objective_usd": None,
                "objective_change_vs_baseline_usd": None,
                "objective_change_vs_baseline_pct": None,
                "runtime_sec": None,
                "n_vars": None,
                "n_constrs": None,
                "delta": exp["solve_kwargs"].get("delta"),
                "alpha": exp["solve_kwargs"].get("alpha"),
                "N_min": exp["solve_kwargs"].get("N_min"),
                "maintenance_window": exp["solve_kwargs"].get("maintenance_window"),
                "max_active_periods": exp["solve_kwargs"].get("max_active_periods"),
                "n_neg_pi_combos": None,
                "total_flights": None,
                "active_route_periods": None,
                "route_openings": None,
                "sum_q": None,
                "total_aircraft_hours": None,
                "error_message": str(exc),
            })

    # ------------------------------------------------------------------
    # Print and export outputs
    # ------------------------------------------------------------------
    print_summary_table(summary_rows)

    if baseline_result is not None:
        print_top_route_periods(baseline_result, top_n=20)

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(flight_rows).to_csv(FLIGHT_CSV, index=False)
    pd.DataFrame(q_rows).to_csv(Q_CSV, index=False)

    print()
    print("Output files written:")
    print(f"  Summary table       : {SUMMARY_CSV}")
    print(f"  Flight schedule     : {FLIGHT_CSV}")
    print(f"  Aircraft deployment : {Q_CSV}")
    print()
    print("Done. Full-dataset 10-experiment run completed.")
