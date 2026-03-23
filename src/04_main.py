"""
main.py
-------
Runs: A SMALL test instance (10 routes, 3 months, 3 aircraft)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data, print_data_summary
from model import build_and_solve

# print results
def print_results(results, title=""):
    print()
    print("=" * 65)
    print(f"  {title}")
    print(f"  delta={results['delta']:.2f}  |  alpha={results['alpha']:.2f}")
    print("=" * 65)
    print(f"  Solver status : {results['status']}")
    print(f"  Runtime       : {results['runtime']:.3f} s")
    print(f"  Variables     : {results['n_vars']}")
    print(f"  Constraints   : {results['n_constrs']}")

    if results['obj_value'] is None:
        print("  *** No feasible solution found. ***")
        return

    print(f"  Objective z*  : ${results['obj_value']:>18,.2f}  (USD)")
    print()

# active route summary  
    served = sorted(results['y_vals'].keys())
    print(f"  Active routes : {len(served)}")
    for (r, t) in served:
        flights_this_rt = {
            (rr, tt, aa): v
            for (rr, tt, aa), v in results['x_vals'].items()
            if rr == r and tt == t
        }
        total_flights = sum(flights_this_rt.values())
        detail = "  ".join(f"{aa}:{v}" for (_, _, aa), v in flights_this_rt.items())
        print(f"    {r}  Month {t:>2}  | total={total_flights:>3}  [{detail}]")

    print()

# run in small instance
if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("# SMALL TEST INSTANCE ONLY")
    print("#" * 65)

    small_data = load_data(
        n_routes=10,
        n_months=3,
        n_aircraft=3,
        K_ct_value=1,
        alpha=1.0,
    )

    print_data_summary(small_data)

    res_small = build_and_solve(
        small_data,
        delta=0.0,
        alpha=1.0,
        time_limit=120,
        verbose=True,
    )

    print_results(res_small, title="Small Instance - Baseline")

    print("\nDone. Small instance completed.")
