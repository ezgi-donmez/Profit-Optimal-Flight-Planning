"""
main_ga_v3_improved.py
----------------------
Runner for the improved GA model.

Put this file in your src/ folder together with:
    data_loader.py
    ga_model_v3_improved.py

Run:
    python main_ga_v3_improved.py
"""

import os
import sys
import copy
from typing import Dict, Any, List

import pandas as pd

# Make sure imports come from the same src folder as this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, print_data_summary
from ga_model_v2 import build_and_solve_ga_improved


# =============================================================================
# Full-dataset settings
# =============================================================================

FULL_LOAD_KWARGS = dict(
    n_routes=None,
    n_months=None,
    n_aircraft=None,
    K_ct_value=1,
)

# Use the same fixed-cost assumptions as your main_v2.py baseline.
BASE_AIRCRAFT_FC = 50_000.0
BASE_ROUTE_RC = 25_000.0
BASE_ROUTE_SC = 150_000.0

# Improved GA settings for full dataset.
# If this is too slow, lower population_size/generations first.
GA_SETTINGS = dict(
    delta=0.0,
    alpha=1.0,
    N_min=2,
    maintenance_window=5,
    max_active_periods=3,
    population_size=160,
    generations=300,
    tournament_size=5,
    crossover_rate=0.90,
    mutation_rate=0.10,
    elitism_count=5,
    stall_generations=140,
    active_probability=0.45,
    greedy_seed_fraction=0.35,
    use_greedy_fill=True,
    greedy_fill_passes=10,
    route_mutations_per_child=4,
    x_mutations_per_child=20,
    seed=42,
    verbose=True,
    print_every=25,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ga_v3_improved")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Helper functions
# =============================================================================

def apply_cost_assumptions(
    data: Dict[str, Any],
    aircraft_fc: float = BASE_AIRCRAFT_FC,
    route_rc: float = BASE_ROUTE_RC,
    route_sc: float = BASE_ROUTE_SC,
) -> Dict[str, Any]:
    """Apply the same fixed-cost assumptions used in the Gurobi baseline."""
    routes = data["routes"]
    periods = data["periods"]
    aircraft = data["aircraft"]

    data["FC"] = {a: float(aircraft_fc) for a in aircraft}
    data["RC"] = {(r, t): float(route_rc) for r in routes for t in periods}
    data["SC"] = {r: float(route_sc) for r in routes}
    return data


def build_flight_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for (route, month, aircraft), flights in sorted(res.get("x_vals", {}).items()):
        rows.append({
            "route": route,
            "month": month,
            "aircraft": aircraft,
            "flights": flights,
        })
    return rows


def build_active_route_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for (route, month), val in sorted(res.get("y_vals", {}).items()):
        rows.append({
            "route": route,
            "month": month,
            "active": val,
        })
    return rows


def build_opening_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for (route, month), val in sorted(res.get("z_vals", {}).items()):
        rows.append({
            "route": route,
            "month": month,
            "opened": val,
        })
    return rows


def build_q_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for (aircraft, month), q_val in sorted(res.get("q_vals", {}).items()):
        rows.append({
            "aircraft": aircraft,
            "month": month,
            "q_deployed": q_val,
            "assigned_hours": res.get("aircraft_hours_vals", {}).get((aircraft, month), 0.0),
        })
    return rows


def build_violation_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"constraint": name, "violation": amount}
        for name, amount in res.get("violated_constraints", [])
    ]


def summarize(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": res.get("status"),
        "objective_value": res.get("obj_value"),
        "fitness": res.get("fitness"),
        "penalty": res.get("penalty"),
        "total_violation": res.get("total_violation"),
        "runtime_sec": res.get("runtime"),
        "best_generation": res.get("best_generation"),
        "total_flights": sum(res.get("x_vals", {}).values()),
        "active_route_periods": len(res.get("y_vals", {})),
        "route_openings": len(res.get("z_vals", {})),
        "sum_q": sum(res.get("q_vals", {}).values()),
        "total_aircraft_hours": sum(res.get("aircraft_hours_vals", {}).values()),
        "population_size": res.get("population_size"),
        "generations_requested": res.get("generations_requested"),
        "delta": res.get("delta"),
        "alpha": res.get("alpha"),
        "N_min": res.get("N_min"),
        "maintenance_window": res.get("maintenance_window"),
        "max_active_periods": res.get("max_active_periods"),
        "n_valid_combos": res.get("n_valid_combos"),
        "n_x_genes": res.get("n_x_genes"),
        "n_y_genes": res.get("n_y_genes"),
        "n_q_genes": res.get("n_q_genes"),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 90)
    print("# IMPROVED GENETIC ALGORITHM RUNNER")
    print("# Full dataset, no Gurobi used inside GA")
    print("#" * 90)

    data = load_data(**FULL_LOAD_KWARGS)
    data = apply_cost_assumptions(data)
    print_data_summary(data)

    res = build_and_solve_ga_improved(data, **GA_SETTINGS)

    summary_row = summarize(res)

    pd.DataFrame([summary_row]).to_csv(os.path.join(OUTPUT_DIR, "ga_v3_summary.csv"), index=False)
    pd.DataFrame(build_flight_rows(res)).to_csv(os.path.join(OUTPUT_DIR, "ga_v3_flight_schedule.csv"), index=False)
    pd.DataFrame(build_active_route_rows(res)).to_csv(os.path.join(OUTPUT_DIR, "ga_v3_active_routes.csv"), index=False)
    pd.DataFrame(build_opening_rows(res)).to_csv(os.path.join(OUTPUT_DIR, "ga_v3_route_openings.csv"), index=False)
    pd.DataFrame(build_q_rows(res)).to_csv(os.path.join(OUTPUT_DIR, "ga_v3_aircraft_deployment.csv"), index=False)
    pd.DataFrame(build_violation_rows(res)).to_csv(os.path.join(OUTPUT_DIR, "ga_v3_violated_constraints.csv"), index=False)
    pd.DataFrame(res.get("history", [])).to_csv(os.path.join(OUTPUT_DIR, "ga_v3_history.csv"), index=False)

    print("\n" + "=" * 90)
    print("IMPROVED GA RUN COMPLETED")
    print(f"Status             : {res.get('status')}")
    print(f"Objective value    : {res.get('obj_value'):,.2f}")
    print(f"Penalized fitness  : {res.get('fitness'):,.2f}")
    print(f"Total violation    : {res.get('total_violation'):.6f}")
    print(f"Runtime seconds    : {res.get('runtime'):.2f}")
    print(f"Best generation    : {res.get('best_generation')}")
    print(f"Total flights      : {summary_row['total_flights']:,}")
    print(f"Active route-periods: {summary_row['active_route_periods']:,}")
    print(f"Route openings     : {summary_row['route_openings']:,}")
    print(f"sum_q              : {summary_row['sum_q']:,}")
    print(f"Output folder      : {OUTPUT_DIR}")
    print("=" * 90)

    if res.get("status") == "FEASIBLE_GA":
        print("\nOK: The improved GA found a feasible solution.")
        print("Compare its objective value, flights, runtime, and route openings with Gurobi.")
    else:
        print("\nWARNING: The improved GA still has constraint violations.")
        print("Open ga_v3_violated_constraints.csv and inspect the remaining violation type.")
