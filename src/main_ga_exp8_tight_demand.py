# -*- coding: utf-8 -*-
"""
Created on Sun May  3 09:31:35 2026

@author: ed024981
"""

# main_ga_exp8_tight_demand.py
# ------------------------------------------------------------
# GA runner for Experiment 8: Tight demand cap eta = 1.05
# This matches the Gurobi Experiment 8 scenario.
# ------------------------------------------------------------

import os
import sys
from typing import Dict, Any, List

import pandas as pd

# Make sure imports come from the same src folder as this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, print_data_summary

# IMPORTANT:
# If your improved GA code is saved as ga_model_v2.py, keep this:
from ga_model_v2 import build_and_solve_ga

# If your improved GA code is saved as ga_model_v3_improved.py instead,
# use this import and comment out the previous one:
# from ga_model_v3_improved import build_and_solve_ga_improved


# =============================================================================
# Full-dataset settings
# =============================================================================

FULL_LOAD_KWARGS = dict(
    n_routes=None,
    n_months=None,
    n_aircraft=None,
    K_ct_value=1,
)

# Same fixed-cost assumptions as main_v2.py baseline
BASE_AIRCRAFT_FC = 50_000.0
BASE_ROUTE_RC = 25_000.0
BASE_ROUTE_SC = 150_000.0

# Experiment 8 setting
TIGHT_DEMAND_ETA = 1.05

GA_SETTINGS = dict(
    delta=0.0,
    alpha=1.0,
    N_min=2,
    maintenance_window=5,
    max_active_periods=3,

    # You can keep your improved GA settings
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ga_exp8_tight_demand")
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
    """
    Apply the same fixed-cost assumptions used in the Gurobi baseline.
    """
    routes = data["routes"]
    periods = data["periods"]
    aircraft = data["aircraft"]

    data["FC"] = {a: float(aircraft_fc) for a in aircraft}
    data["RC"] = {(r, t): float(route_rc) for r in routes for t in periods}
    data["SC"] = {r: float(route_sc) for r in routes}
    return data


def set_oversupply_factor(data: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """
    This matches Experiment 8 in main_v2.py:
    eta[r,t] = 1.05 for all route-period pairs.
    """
    data["oversupply_factor"] = float(factor)
    data["eta"] = {
        (r, t): float(factor)
        for r in data["routes"]
        for t in data["periods"]
    }
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
        "scenario": "Experiment 8 - Tight demand cap eta=1.05",
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
        "eta": TIGHT_DEMAND_ETA,
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
    print("# GA EXPERIMENT 8: TIGHT DEMAND CAP eta = 1.05")
    print("# Full dataset, no Gurobi used inside GA")
    print("#" * 90)

    # 1. Load full dataset
    data = load_data(**FULL_LOAD_KWARGS)

    # 2. Apply the same fixed-cost assumptions as Gurobi baseline
    data = apply_cost_assumptions(data)

    # 3. Apply Experiment 8 demand cap: eta = 1.05
    data = set_oversupply_factor(data, TIGHT_DEMAND_ETA)

    print_data_summary(data)

    # 4. Run improved GA
    res = build_and_solve_ga(data, **GA_SETTINGS)

    summary_row = summarize(res)

    # 5. Export outputs
    pd.DataFrame([summary_row]).to_csv(
        os.path.join(OUTPUT_DIR, "ga_exp8_summary.csv"),
        index=False
    )

    pd.DataFrame(build_flight_rows(res)).to_csv(
        os.path.join(OUTPUT_DIR, "ga_exp8_flight_schedule.csv"),
        index=False
    )

    pd.DataFrame(build_active_route_rows(res)).to_csv(
        os.path.join(OUTPUT_DIR, "ga_exp8_active_routes.csv"),
        index=False
    )

    pd.DataFrame(build_opening_rows(res)).to_csv(
        os.path.join(OUTPUT_DIR, "ga_exp8_route_openings.csv"),
        index=False
    )

    pd.DataFrame(build_q_rows(res)).to_csv(
        os.path.join(OUTPUT_DIR, "ga_exp8_aircraft_deployment.csv"),
        index=False
    )

    pd.DataFrame(build_violation_rows(res)).to_csv(
        os.path.join(OUTPUT_DIR, "ga_exp8_violated_constraints.csv"),
        index=False
    )

    pd.DataFrame(res.get("history", [])).to_csv(
        os.path.join(OUTPUT_DIR, "ga_exp8_history.csv"),
        index=False
    )

    print("\n" + "=" * 90)
    print("GA EXPERIMENT 8 COMPLETED")
    print(f"Scenario           : Tight demand cap eta = {TIGHT_DEMAND_ETA}")
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
        print("\nOK: GA found a feasible solution for Experiment 8.")
        print("Compare this with Gurobi Experiment 8: Tight demand cap eta=1.05.")
    else:
        print("\nWARNING: GA has constraint violations.")
        print("Open ga_exp8_violated_constraints.csv and inspect the remaining violation type.")