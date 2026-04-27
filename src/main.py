"""
main.py
----------------
Complete experiment runner for the Profit-Optimal Flight Planning MILP (v2).
DS502 Path B

Scenarios
─────────
  1. Small baseline        10 routes × 3 months × 3 aircraft
  2. Fuel-shock            delta  ∈ {0, 0.10, 0.25}
  3. Capacity              alpha  ∈ {0.30, 0.60, 1.00}
  4. Min-up-time           N_min  ∈ {1, 2, 3}
  5. Maintenance window    N_maint ∈ {1, 2, 3}
  6. Maintenance intensity gamma  ∈ {0.0, 0.10, 0.20, 0.30}
  7. Activation cost       FC_a   ∈ {0, 50 000, 100 000}
  8. Medium baseline       20 routes × 8 months × 4 aircraft
  9. Large baseline        16 routes × 12 months × 6 aircraft  ← max within restricted license
 10. Large fuel-shock      delta ∈ {0, 0.10, 0.25} on large instance
 11. Large maintenance     gamma ∈ {0.0, 0.20, 0.40} on large instance
 12. Full dataset baseline 30 routes × 12 months × 6 aircraft  ← optional, requires full license

NOTE ON LICENSE
───────────────
  Restricted Gurobi license: max ~2000 constraints.
  Full dataset (30 × 12 × 6) requires ~2900 constraints → exceeds limit.
  16 routes × 12 months × 6 aircraft stays under the restricted license limit.
  To run the full 30-route dataset: obtain a free Gurobi academic license
  at gurobi.com/academia and set RUN_FULL_DATASET = True.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data, print_data_summary
from model import build_and_solve


# ─────────────────────────────────────────────────────────────────────────────
# Optional full-dataset switch
# ─────────────────────────────────────────────────────────────────────────────
# Set this to True only if you have a full Gurobi academic license.
# The restricted license may fail because the full dataset can exceed
# the maximum allowed model size.
RUN_FULL_DATASET = True


# ─────────────────────────────────────────────────────────────────────────────
# Helper: print a single result
# ─────────────────────────────────────────────────────────────────────────────
def print_results(res, title=""):
    print()
    print("=" * 72)
    print(f"  {title}")
    print(
        f"  delta={res['delta']:.2f} | alpha={res['alpha']:.2f} | "
        f"N_min={res['N_min']} | N_maint={res['N_maint']} | "
        f"gamma={res['gamma_maint']:.2f}"
    )
    print("=" * 72)
    print(f"  Status      : {res['status']}")
    print(f"  Runtime     : {res['runtime']:.3f} s")
    print(f"  Variables   : {res['n_vars']}  |  Constraints: {res['n_constrs']}")

    if res["obj_value"] is None:
        print("  *** No feasible solution found. ***")
        return

    print(f"  Objective   : ${res['obj_value']:>20,.2f}  (USD)")
    print()

    served = sorted(res["y_vals"].keys())
    print(f"  Active route-periods : {len(served)}")

    for (r, t) in served:
        flights = {
            (rr, tt, aa): v
            for (rr, tt, aa), v in res["x_vals"].items()
            if rr == r and tt == t
        }

        total = sum(flights.values())
        detail = "  ".join(f"{aa}:{v}" for (_, _, aa), v in flights.items())
        new = " [NEW]" if (r, t) in res["z_vals"] else ""

        print(f"    {r:<15}  Month {t:>2}  | total={total:>4}  [{detail}]{new}")

    if res["z_vals"]:
        print()
        print(f"  Route openings : {len(res['z_vals'])}")
        for (r, t) in sorted(res["z_vals"]):
            print(f"    {r}  opened Month {t}")

    print()
    print("  Aircraft deployments (u=1):")

    for a in sorted({a for (a, t) in res["u_vals"]}):
        periods_used = sorted(t for (aa, t) in res["u_vals"] if aa == a)
        detail = ", ".join(
            f"M{t}(W={res['W_vals'].get((a, t), 0)})" for t in periods_used
        )
        print(f"    {a:<20}  ->  {detail if detail else 'not deployed'}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compact scenario table
# ─────────────────────────────────────────────────────────────────────────────
def print_table(results, label=""):
    print()
    print("=" * 100)
    print(f"  SCENARIO TABLE  [{label}]")
    print("=" * 100)
    print(
        f"  {'delta':>5}  {'alpha':>5}  {'Nmin':>4}  {'Nm':>3}  "
        f"{'gamma':>5}  {'neg':>4}  {'Status':>10}  "
        f"{'Objective (USD)':>22}  {'Rts':>4}  {'Open':>4}  "
        f"{'u=1':>4}  {'Time':>7}"
    )
    print("  " + "-" * 98)

    for r in results:
        obj = (
            f"${r['obj_value']:>20,.2f}"
            if r["obj_value"] is not None
            else "                   N/A"
        )

        nrts = len(r["y_vals"])
        nopn = len(r["z_vals"])
        ndep = len(r["u_vals"])

        print(
            f"  {r['delta']:>5.2f}  {r['alpha']:>5.2f}  "
            f"{r['N_min']:>4}  {r['N_maint']:>3}  "
            f"{r['gamma_maint']:>5.2f}  {r['n_neg_pi_combos']:>4}  "
            f"{r['status']:>10}  {obj}  "
            f"{nrts:>4}  {nopn:>4}  {ndep:>4}  {r['runtime']:>6.3f}s"
        )

    print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 1. SMALL BASELINE ────────────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 1 — SMALL BASELINE  (10 routes, 3 months, 3 aircraft)")
    print("#" * 72)

    small = load_data(n_routes=10, n_months=3, n_aircraft=3, K_ct_value=1)
    print_data_summary(small)

    res_s = build_and_solve(
        small,
        delta=0.0,
        alpha=1.0,
        N_min=2,
        N_maint=2,
        gamma_maint=0.20,
        time_limit=120,
        verbose=True,
    )
    print_results(res_s, "Small — Baseline")

    # ── 2. FUEL-SHOCK ────────────────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 2 — FUEL-SHOCK  (delta ∈ {0, 0.10, 0.25})")
    print("#" * 72)

    fuel = []
    for d in [0.0, 0.10, 0.25]:
        r = build_and_solve(
            small,
            delta=d,
            alpha=1.0,
            N_min=2,
            N_maint=2,
            gamma_maint=0.20,
            time_limit=120,
            verbose=False,
        )
        fuel.append(r)
        tag = f"${r['obj_value']:,.2f}" if r["obj_value"] is not None else r["status"]
        print(f"  delta={d:.2f}  neg-pi={r['n_neg_pi_combos']:>3}  ->  {tag}")

    print_table(fuel, "Fuel-shock — Small")

    # ── 3. CAPACITY ──────────────────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 3 — CAPACITY  (alpha ∈ {0.30, 0.60, 1.00})")
    print("#" * 72)

    cap = []
    for a in [0.30, 0.60, 1.00]:
        r = build_and_solve(
            small,
            delta=0.0,
            alpha=a,
            N_min=2,
            N_maint=2,
            gamma_maint=0.20,
            time_limit=120,
            verbose=False,
        )
        cap.append(r)
        tag = f"${r['obj_value']:,.2f}" if r["obj_value"] is not None else r["status"]
        print(f"  alpha={a:.2f}  ->  {tag}")

    print_table(cap, "Capacity — Small")

    # ── 4. MIN-UP-TIME ───────────────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 4 — MIN-UP-TIME  (N_min ∈ {1, 2, 3})")
    print("#" * 72)

    nmin = []
    for n in [1, 2, 3]:
        r = build_and_solve(
            small,
            delta=0.0,
            alpha=1.0,
            N_min=n,
            N_maint=2,
            gamma_maint=0.20,
            time_limit=120,
            verbose=False,
        )
        nmin.append(r)
        tag = f"${r['obj_value']:,.2f}" if r["obj_value"] is not None else r["status"]
        print(f"  N_min={n}  opens={len(r['z_vals'])}  ->  {tag}")

    print_table(nmin, "Min-up-time — Small")

    # ── 5. MAINTENANCE WINDOW ────────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 5 — MAINTENANCE WINDOW  (N_maint ∈ {1, 2, 3})")
    print("#" * 72)

    nmnt = []
    for nm in [1, 2, 3]:
        r = build_and_solve(
            small,
            delta=0.0,
            alpha=1.0,
            N_min=2,
            N_maint=nm,
            gamma_maint=0.20,
            time_limit=120,
            verbose=False,
        )
        nmnt.append(r)
        tag = f"${r['obj_value']:,.2f}" if r["obj_value"] is not None else r["status"]
        print(f"  N_maint={nm}  ->  {tag}")

    print_table(nmnt, "Maintenance window — Small")

    # ── 6. MAINTENANCE INTENSITY ─────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 6 — MAINTENANCE INTENSITY  (gamma ∈ {0.0, 0.10, 0.20, 0.30})")
    print("# Shows how much profit drops as maintenance consumes more capacity.")
    print("#" * 72)

    gam = []
    for g in [0.0, 0.10, 0.20, 0.30]:
        r = build_and_solve(
            small,
            delta=0.0,
            alpha=1.0,
            N_min=2,
            N_maint=2,
            gamma_maint=g,
            time_limit=120,
            verbose=False,
        )
        gam.append(r)
        tag = f"${r['obj_value']:,.2f}" if r["obj_value"] is not None else r["status"]
        print(f"  gamma={g:.2f}  ->  {tag}")

    print_table(gam, "Maintenance intensity — Small")

    # ── 7. ACTIVATION COST ───────────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 7 — AIRCRAFT ACTIVATION COST  (FC_a ∈ {0, 50K, 100K})")
    print("# FC_a is the fixed cost charged each time an aircraft type is deployed.")
    print("#" * 72)

    act = []
    for fc in [0, 50_000, 100_000]:
        small_fc = dict(small)
        small_fc["FC"] = {a: float(fc) for a in small["aircraft"]}

        r = build_and_solve(
            small_fc,
            delta=0.0,
            alpha=1.0,
            N_min=2,
            N_maint=2,
            gamma_maint=0.20,
            time_limit=120,
            verbose=False,
        )
        act.append(r)
        tag = f"${r['obj_value']:,.2f}" if r["obj_value"] is not None else r["status"]
        print(f"  FC_a={fc:>7,}  ->  {tag}  (u=1 count: {len(r['u_vals'])})")

    print_table(act, "Activation cost — Small")

    # ── 8. MEDIUM BASELINE ───────────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 8 — MEDIUM BASELINE  (20 routes, 8 months, 4 aircraft)")
    print("#" * 72)

    medium = load_data(n_routes=20, n_months=8, n_aircraft=4, K_ct_value=1)
    print_data_summary(medium)

    res_m = build_and_solve(
        medium,
        delta=0.0,
        alpha=1.0,
        N_min=2,
        N_maint=2,
        gamma_maint=0.20,
        time_limit=300,
        verbose=True,
    )
    print_results(res_m, "Medium — Baseline")

    # ── 9. LARGE BASELINE (max within restricted license) ────────────────────
    print("\n" + "#" * 72)
    print("# STEP 9 — LARGE BASELINE  (16 routes, 12 months, 6 aircraft)")
    print("# This is the largest instance solvable within the restricted")
    print("# Gurobi license limit.")
    print("# For the full 30-route dataset: obtain a full Gurobi academic")
    print("# license and set RUN_FULL_DATASET = True.")
    print("#" * 72)

    large = load_data(n_routes=16, n_months=12, n_aircraft=6, K_ct_value=1)
    print_data_summary(large)

    res_l = build_and_solve(
        large,
        delta=0.0,
        alpha=1.0,
        N_min=2,
        N_maint=2,
        gamma_maint=0.20,
        time_limit=300,
        verbose=True,
    )
    print_results(res_l, "Large — Baseline (16r × 12m × 6a)")

    # ── 10. LARGE FUEL-SHOCK ─────────────────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 10 — LARGE FUEL-SHOCK  (delta ∈ {0, 0.10, 0.25})")
    print("#" * 72)

    lfuel = []
    for d in [0.0, 0.10, 0.25]:
        r = build_and_solve(
            large,
            delta=d,
            alpha=1.0,
            N_min=2,
            N_maint=2,
            gamma_maint=0.20,
            time_limit=300,
            verbose=False,
        )
        lfuel.append(r)
        tag = f"${r['obj_value']:,.2f}" if r["obj_value"] is not None else r["status"]
        print(f"  delta={d:.2f}  neg-pi={r['n_neg_pi_combos']:>3}  ->  {tag}")

    print_table(lfuel, "Fuel-shock — Large")

    # ── 11. LARGE MAINTENANCE INTENSITY ──────────────────────────────────────
    print("\n" + "#" * 72)
    print("# STEP 11 — LARGE MAINTENANCE INTENSITY  (gamma ∈ {0.0, 0.20, 0.40})")
    print("# 12-month horizon makes maintenance capacity reduction visible.")
    print("#" * 72)

    lgam = []
    for g in [0.0, 0.20, 0.40]:
        r = build_and_solve(
            large,
            delta=0.0,
            alpha=1.0,
            N_min=2,
            N_maint=2,
            gamma_maint=g,
            time_limit=300,
            verbose=False,
        )
        lgam.append(r)
        tag = f"${r['obj_value']:,.2f}" if r["obj_value"] is not None else r["status"]
        print(f"  gamma={g:.2f}  ->  {tag}")

    print_table(lgam, "Maintenance intensity — Large")

    # ── 12. FULL DATASET BASELINE ────────────────────────────────────────────
    # Optional: run only with a full Gurobi academic license.
    # The restricted license may fail because the full dataset can exceed
    # the allowed model size.
    if RUN_FULL_DATASET:
        print("\n" + "#" * 72)
        print("# STEP 12 — FULL DATASET BASELINE  (30 routes, 12 months, 6 aircraft)")
        print("# This scenario uses the full available route set.")
        print("# It may exceed the restricted Gurobi license limit.")
        print("#" * 72)

        full = load_data(n_routes=30, n_months=12, n_aircraft=6, K_ct_value=1)
        print_data_summary(full)

        try:
            res_full = build_and_solve(
                full,
                delta=0.0,
                alpha=1.0,
                N_min=2,
                N_maint=2,
                gamma_maint=0.20,
                time_limit=600,
                verbose=True,
            )
            print_results(res_full, "Full Dataset — Baseline (30r × 12m × 6a)")

        except Exception as e:
            print()
            print("FULL DATASET RUN FAILED.")
            print("This is most likely due to the restricted Gurobi license size limit.")
            print("To run the full dataset, use a full Gurobi academic license.")
            print(f"Error message: {e}")

    else:
        print("\n" + "#" * 72)
        print("# STEP 12 — FULL DATASET BASELINE  (SKIPPED)")
        print("# Set RUN_FULL_DATASET = True to run the 30 routes × 12 months × 6 aircraft case.")
        print("# This may require a full Gurobi academic license.")
        print("#" * 72)

    print("\nDone. All scenarios completed.")
