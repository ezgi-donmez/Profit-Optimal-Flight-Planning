"""
ga_model_v3_improved.py
-----------------------
Improved Genetic Algorithm (GA) for the final airline route-frequency /
fleet-planning model.

This version keeps the SAME data dictionary structure used by data_loader.py and
model_v2_final.py, but it does NOT import or call Gurobi.

Main function
-------------
    build_and_solve_ga_improved(data, ...)

Main improvements compared with the repaired GA
-----------------------------------------------
1. Correct q repair: q[a,t] is recalculated as the minimum aircraft deployment
   required by the assigned aircraft hours, instead of keeping unnecessary
   random aircraft.
2. Temporal route repair: route openings are repaired to satisfy minimum
   up-time and no-late-opening logic.
3. Category repair: missing category coverage is repaired by activating
   feasible route blocks.
4. Greedy fill local search: after feasibility repair, profitable flights are
   added while route, demand, hub, fleet-hour, and rolling-maintenance limits
   remain satisfied.
5. Route-based crossover: children inherit complete route schedules from
   parents instead of using only flat one-point crossover.
6. Route-block mutation: route activation is mutated in blocks rather than
   single random month flips.
7. Semi-greedy initial population: a part of the population is seeded with
   high-profit route-period patterns.

Objective and fitness
---------------------
The original MILP objective is maximized. Constraint violations are penalized:
    fitness = objective - penalty_weight * total_violation

If total_violation = 0, the penalized fitness equals the raw objective.
"""

import time
import math
import random
from typing import Dict, Any, List, Tuple, Optional


# =============================================================================
# Backward-compatible default helpers
# =============================================================================

def _guess_seats(aircraft_name: str) -> int:
    name = str(aircraft_name).lower()
    if "a380" in name:
        return 500
    if "777" in name:
        return 360
    if "a350" in name:
        return 325
    if "787" in name:
        return 290
    if "a320" in name:
        return 180
    if "737" in name:
        return 170
    return 220


def _add_backward_compatible_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """Add final-model keys if an older data_loader output is passed."""
    routes = list(data["routes"])
    periods = list(data["periods"])
    aircraft = list(data["aircraft"])
    combos = sorted(list(data["combos"]))
    M = data["M"]

    data.setdefault("SC", {r: 0.0 for r in routes})
    data.setdefault("RC", {(r, t): 0.0 for r in routes for t in periods})
    data.setdefault("FC", {a: 0.0 for a in aircraft})
    data.setdefault("Seats", {a: _guess_seats(a) for a in aircraft})

    if "compat" not in data:
        compatible_pairs = {(r, a) for (r, _, a) in combos}
        data["compat"] = {
            (r, a): 1 if (r, a) in compatible_pairs else 0
            for r in routes for a in aircraft
        }

    if "FleetSize" not in data:
        H_bar = data.get("H_bar", {})
        data["FleetSize"] = {}
        for a in aircraft:
            max_h = max([float(H_bar.get((a, t), 0.0)) for t in periods] + [0.0])
            data["FleetSize"][a] = max(1, int(math.ceil(max_h / 120.0))) if max_h > 0 else 1

    if "H_unit" not in data:
        H_bar = data.get("H_bar", data.get("H", {}))
        data["H_unit"] = {}
        for a in aircraft:
            fs = max(1, int(data["FleetSize"].get(a, 1)))
            for t in periods:
                data["H_unit"][(a, t)] = float(H_bar.get((a, t), 0.0)) / fs

    if "U_route" not in data:
        data["U_route"] = {}
        for r in routes:
            for t in periods:
                total_m = sum(int(M.get((r, t, a), 0)) for a in aircraft)
                data["U_route"][(r, t)] = max(int(data.get("L", {}).get((r, t), 0)), total_m)

    if "HubSlot" not in data:
        data["HubSlot"] = {
            t: sum(data["U_route"].get((r, t), 0) for r in routes)
            for t in periods
        }

    if "Demand" not in data:
        data["Demand"] = {}
        for r in routes:
            for t in periods:
                offered = sum(
                    data["Seats"][a] * int(M.get((r, t, a), 0))
                    for a in aircraft
                )
                data["Demand"][(r, t)] = max(1, int(math.ceil(0.85 * offered)))

    if "eta" not in data:
        factor = data.get("oversupply_factor", 1.20)
        data["eta"] = {(r, t): float(factor) for r in routes for t in periods}

    data.setdefault("Y0", {r: 0 for r in routes})
    data.setdefault("nfoc", {c: 0.0 for c in combos})

    return data


def _require_keys(data: Dict[str, Any], required_keys: List[str]) -> None:
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            "data dictionary is missing required keys: "
            + ", ".join(missing)
            + ". Check that your data_loader.py returns the final-model data structure."
        )


def _clone_chrom(chrom: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return {"x": chrom["x"][:], "y": chrom["y"][:], "q": chrom["q"][:]}


# =============================================================================
# Main improved GA solver
# =============================================================================

def build_and_solve_ga_improved(
    data: Dict[str, Any],
    delta: float = 0.0,
    alpha: float = 1.0,
    N_min: int = 2,
    maintenance_window: int = 5,
    max_active_periods: int = 3,
    Omega: Optional[float] = None,
    B_nfoc: Optional[float] = None,
    population_size: int = 160,
    generations: int = 300,
    tournament_size: int = 5,
    crossover_rate: float = 0.90,
    mutation_rate: float = 0.10,
    elitism_count: int = 5,
    stall_generations: int = 140,
    penalty_weight: Optional[float] = None,
    active_probability: float = 0.45,
    greedy_seed_fraction: float = 0.35,
    use_greedy_fill: bool = True,
    greedy_fill_passes: int = 10,
    route_mutations_per_child: int = 4,
    x_mutations_per_child: int = 20,
    seed: Optional[int] = 42,
    verbose: bool = True,
    print_every: int = 25,
) -> Dict[str, Any]:
    """
    Approximate the final MILP using an improved GA.

    This function is compatible with the same data dictionary used by the
    Gurobi model, but it does not use Gurobi.
    """

    if seed is not None:
        random.seed(seed)

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Read and validate data
    # -------------------------------------------------------------------------
    data = _add_backward_compatible_defaults(dict(data))
    required = [
        "routes", "periods", "aircraft", "categories", "R_c", "combos",
        "pi", "f", "h", "M", "L", "K",
        "SC", "RC", "FC",
        "FleetSize", "H_unit",
        "Demand", "Seats", "eta",
        "compat", "HubSlot", "U_route",
    ]
    _require_keys(data, required)

    routes = list(data["routes"])
    periods = sorted(list(data["periods"]))
    aircraft = list(data["aircraft"])
    categories = list(data["categories"])
    R_c = data["R_c"]
    raw_combos = sorted(list(data["combos"]))

    T = periods
    n_T = len(T)
    t_idx = {t: i for i, t in enumerate(T)}

    pi = data["pi"]
    f = data["f"]
    h = data["h"]
    M = data["M"]
    L = data["L"]
    K = data["K"]
    SC = data["SC"]
    RC = data["RC"]
    FC = data["FC"]
    FleetSize = data["FleetSize"]
    H_unit = data["H_unit"]
    Demand = data["Demand"]
    Seats = data["Seats"]
    eta = data["eta"]
    compat = data["compat"]
    HubSlot = data["HubSlot"]
    U_route = data["U_route"]
    Y0 = data.get("Y0", {r: 0 for r in routes})
    nfoc = data.get("nfoc", {c: 0.0 for c in raw_combos})

    combos = [
        (r, t, a)
        for (r, t, a) in raw_combos
        if r in routes and t in periods and a in aircraft
        and compat.get((r, a), 0) == 1
        and (r, t, a) in M
        and (r, t, a) in pi
        and (r, t, a) in f
        and (r, t, a) in h
    ]
    if not combos:
        raise ValueError("No valid compatible (route, period, aircraft) combinations exist.")

    pi_delta = {
        key: float(pi[key]) - float(delta) * float(f[key])
        for key in combos
    }

    route_period_keys = [(r, t) for r in routes for t in periods]
    aircraft_period_keys = [(a, t) for a in aircraft for t in periods]

    x_index = {key: i for i, key in enumerate(combos)}
    y_index = {key: i for i, key in enumerate(route_period_keys)}
    q_index = {key: i for i, key in enumerate(aircraft_period_keys)}

    combos_by_route_period = {
        (r, t): [(r, t, a) for a in aircraft if (r, t, a) in x_index]
        for r in routes for t in periods
    }
    combos_by_aircraft_period = {
        (a, t): [(r, t, a) for r in routes if (r, t, a) in x_index]
        for a in aircraft for t in periods
    }
    combos_by_period = {
        t: [(r, t, a) for (r, tt, a) in combos if tt == t]
        for t in periods
    }
    combos_by_route = {
        r: [key for key in combos if key[0] == r]
        for r in routes
    }

    # Potential score used by greedy seeding and category repair.
    route_period_potential = {}
    for r in routes:
        for t in periods:
            route_period_potential[(r, t)] = sum(
                max(0.0, pi_delta[key]) * int(M[key])
                for key in combos_by_route_period[(r, t)]
            )

    if penalty_weight is None:
        values = [abs(float(v)) for v in pi_delta.values()]
        values += [abs(float(v)) for v in SC.values()]
        values += [abs(float(v)) for v in RC.values()]
        values += [abs(float(v)) for v in FC.values()]
        profit_scale = max(values + [1.0])
        penalty_weight = 1000.0 * profit_scale

    # -------------------------------------------------------------------------
    # Decoding and accounting helpers
    # -------------------------------------------------------------------------
    def derive_z_from_y(y_gene: List[int]) -> List[int]:
        """Derive opening variable z from y and Y0."""
        z_gene = [0 for _ in route_period_keys]
        for r in routes:
            for t in periods:
                i = t_idx[t]
                current = y_gene[y_index[(r, t)]]
                if i == 0:
                    prev_active = int(Y0.get(r, 0))
                else:
                    prev_active = y_gene[y_index[(r, T[i - 1])]]
                z_gene[y_index[(r, t)]] = 1 if current == 1 and prev_active == 0 else 0
        return z_gene

    def route_total(x_gene: List[int], r: str, t: int) -> int:
        return sum(x_gene[x_index[key]] for key in combos_by_route_period[(r, t)])

    def period_total(x_gene: List[int], t: int) -> int:
        return sum(x_gene[x_index[key]] for key in combos_by_period[t])

    def route_seats(x_gene: List[int], r: str, t: int) -> float:
        return sum(float(Seats[key[2]]) * x_gene[x_index[key]] for key in combos_by_route_period[(r, t)])

    def aircraft_hours(x_gene: List[int], a: str, t: int) -> float:
        return sum(float(h[key]) * x_gene[x_index[key]] for key in combos_by_aircraft_period[(a, t)])

    def recompute_minimum_q(x_gene: List[int]) -> List[int]:
        """
        Correct q repair.

        q is not allowed to remain randomly high. It is set to the minimum
        number of physical aircraft needed for the current assigned flight hours.
        """
        q_gene = [0 for _ in aircraft_period_keys]
        for a in aircraft:
            for t in periods:
                hours_used = aircraft_hours(x_gene, a, t)
                unit_hours = float(alpha) * float(H_unit.get((a, t), 0.0))
                if hours_used <= 1e-9:
                    required_q = 0
                elif unit_hours <= 1e-9:
                    required_q = int(FleetSize[a])
                else:
                    required_q = int(math.ceil(hours_used / unit_hours))
                q_gene[q_index[(a, t)]] = max(0, min(required_q, int(FleetSize[a])))
        return q_gene

    def rolling_maintenance_ok(q_gene: List[int], a: str) -> bool:
        if maintenance_window is None or max_active_periods is None:
            return True
        mw = int(maintenance_window)
        map_ = int(max_active_periods)
        if mw <= 0 or map_ <= 0 or n_T < mw:
            return True
        rhs = map_ * int(FleetSize[a])
        for start_idx in range(0, n_T - mw + 1):
            window = [T[j] for j in range(start_idx, start_idx + mw)]
            lhs = sum(q_gene[q_index[(a, tau)]] for tau in window)
            if lhs > rhs:
                return False
        return True

    def all_rolling_maintenance_ok(q_gene: List[int]) -> bool:
        return all(rolling_maintenance_ok(q_gene, a) for a in aircraft)

    # -------------------------------------------------------------------------
    # Repair helpers
    # -------------------------------------------------------------------------
    def repair_temporal_y(y_gene: List[int]) -> List[int]:
        """Repair min-up-time and no-late-opening constraints in y."""
        y_fixed = [1 if int(v) >= 1 else 0 for v in y_gene]
        N = int(N_min)
        if N <= 1 or n_T == 0:
            return y_fixed

        for r in routes:
            # Remove openings too late to fit N_min.
            for i in range(n_T):
                t = T[i]
                curr = y_fixed[y_index[(r, t)]]
                prev = int(Y0.get(r, 0)) if i == 0 else y_fixed[y_index[(r, T[i - 1])]]
                if curr == 1 and prev == 0 and i + N > n_T:
                    for j in range(i, n_T):
                        y_fixed[y_index[(r, T[j])]] = 0
                    break

            # Extend every opening for N_min periods when feasible.
            i = 0
            while i < n_T:
                t = T[i]
                curr = y_fixed[y_index[(r, t)]]
                prev = int(Y0.get(r, 0)) if i == 0 else y_fixed[y_index[(r, T[i - 1])]]
                if curr == 1 and prev == 0:
                    if i + N <= n_T:
                        window = [T[j] for j in range(i, i + N)]
                        feasible_window = all(len(combos_by_route_period[(r, tau)]) > 0 for tau in window)
                        if feasible_window:
                            for tau in window:
                                y_fixed[y_index[(r, tau)]] = 1
                            i += N
                            continue
                        y_fixed[y_index[(r, t)]] = 0
                    else:
                        y_fixed[y_index[(r, t)]] = 0
                i += 1
        return y_fixed

    def activate_block_containing_period(y_gene: List[int], r: str, t: int) -> None:
        """Activate a route block of length at least N_min that contains period t."""
        N = max(1, int(N_min))
        i = t_idx[t]
        if n_T == 0:
            return
        if N > n_T:
            return

        # Prefer a block starting at t; if too late, shift the block earlier.
        start = min(i, n_T - N)
        end = start + N
        window = [T[j] for j in range(start, end)]
        if all(len(combos_by_route_period[(r, tau)]) > 0 for tau in window):
            for tau in window:
                y_gene[y_index[(r, tau)]] = 1

    def repair_category_coverage(y_gene: List[int]) -> List[int]:
        """Repair category coverage by activating high-potential route blocks."""
        y_fixed = y_gene[:]
        for c in categories:
            for t in periods:
                required_k = int(K.get((c, t), 0))
                if required_k <= 0:
                    continue
                category_routes = [r for r in R_c.get(c, []) if (r, t) in y_index]
                active_count = sum(y_fixed[y_index[(r, t)]] for r in category_routes)
                if active_count >= required_k:
                    continue

                candidates = [
                    r for r in category_routes
                    if y_fixed[y_index[(r, t)]] == 0 and len(combos_by_route_period[(r, t)]) > 0
                ]
                candidates.sort(key=lambda r_: route_period_potential.get((r_, t), 0.0), reverse=True)
                needed = required_k - active_count
                for r in candidates[:needed]:
                    activate_block_containing_period(y_fixed, r, t)

        return repair_temporal_y(y_fixed)

    def add_minimum_service(x_gene: List[int], y_gene: List[int]) -> None:
        """For active route-periods, add enough flights to satisfy L[r,t] when possible."""
        for r in routes:
            for t in periods:
                if y_gene[y_index[(r, t)]] == 0:
                    continue
                relevant = combos_by_route_period[(r, t)]
                if not relevant:
                    y_gene[y_index[(r, t)]] = 0
                    continue
                required = int(L.get((r, t), 0))
                current = route_total(x_gene, r, t)
                if current >= required:
                    continue

                ordered = sorted(relevant, key=lambda key: pi_delta.get(key, 0.0), reverse=True)
                need = required - current
                for key in ordered:
                    if need <= 0:
                        break
                    idx = x_index[key]
                    addable = max(0, int(M[key]) - x_gene[idx])
                    # Do not exceed route-period upper bound or demand cap if avoidable.
                    route_remaining = max(0, int(U_route.get((r, t), 0)) - route_total(x_gene, r, t))
                    demand_remaining = int(math.floor(
                        max(0.0, float(eta.get((r, t), 1.20)) * float(Demand.get((r, t), 1.0)) - route_seats(x_gene, r, t))
                        / max(1.0, float(Seats[key[2]]))
                    ))
                    hub_remaining = max(0, int(HubSlot.get(t, 0)) - period_total(x_gene, t))
                    add = min(addable, need, route_remaining, demand_remaining, hub_remaining)
                    if add > 0:
                        x_gene[idx] += add
                        need -= add

    def trim_route_and_demand_excess(x_gene: List[int]) -> None:
        """Remove low-profit flights if route, demand, or hub limits are exceeded."""
        # Route and demand excess.
        for r in routes:
            for t in periods:
                relevant = combos_by_route_period[(r, t)]
                if not relevant:
                    continue
                ordered_low_profit = sorted(relevant, key=lambda key: pi_delta.get(key, 0.0))

                # Route maximum frequency.
                route_ub = int(U_route.get((r, t), sum(int(M.get(key, 0)) for key in relevant)))
                while route_total(x_gene, r, t) > route_ub:
                    removed = False
                    for key in ordered_low_profit:
                        idx = x_index[key]
                        if x_gene[idx] > 0:
                            x_gene[idx] -= 1
                            removed = True
                            break
                    if not removed:
                        break

                # Demand / seat cap.
                seat_cap = float(eta.get((r, t), 1.20)) * float(Demand.get((r, t), 1.0))
                while route_seats(x_gene, r, t) > seat_cap + 1e-7:
                    removed = False
                    for key in ordered_low_profit:
                        idx = x_index[key]
                        if x_gene[idx] > 0:
                            x_gene[idx] -= 1
                            removed = True
                            break
                    if not removed:
                        break

        # Hub slot excess by period.
        for t in periods:
            relevant = combos_by_period[t]
            ordered_low_profit = sorted(relevant, key=lambda key: pi_delta.get(key, 0.0))
            hub_cap = int(HubSlot.get(t, sum(int(M.get(key, 0)) for key in relevant)))
            while period_total(x_gene, t) > hub_cap:
                removed = False
                for key in ordered_low_profit:
                    idx = x_index[key]
                    if x_gene[idx] > 0:
                        x_gene[idx] -= 1
                        removed = True
                        break
                if not removed:
                    break

    def trim_aircraft_capacity_excess(x_gene: List[int]) -> None:
        """Remove low-profit flights if aircraft-hour capacity is exceeded."""
        for a in aircraft:
            for t in periods:
                unit_hours = float(alpha) * float(H_unit.get((a, t), 0.0))
                max_hours = unit_hours * int(FleetSize[a])
                relevant = combos_by_aircraft_period[(a, t)]
                ordered_low_profit = sorted(relevant, key=lambda key: pi_delta.get(key, 0.0))
                while aircraft_hours(x_gene, a, t) > max_hours + 1e-7:
                    removed = False
                    for key in ordered_low_profit:
                        idx = x_index[key]
                        if x_gene[idx] > 0:
                            x_gene[idx] -= 1
                            removed = True
                            break
                    if not removed:
                        break

    def trim_rolling_maintenance_excess(x_gene: List[int]) -> None:
        """
        If rolling maintenance is violated, remove low-profit flights from the
        violating aircraft-window until the recomputed q satisfies the window.
        """
        if maintenance_window is None or max_active_periods is None:
            return
        mw = int(maintenance_window)
        map_ = int(max_active_periods)
        if mw <= 0 or map_ <= 0 or n_T < mw:
            return

        # Limit iterations to avoid very long loops on difficult instances.
        for _ in range(2000):
            q_gene = recompute_minimum_q(x_gene)
            found_violation = False

            for a in aircraft:
                rhs = map_ * int(FleetSize[a])
                for start_idx in range(0, n_T - mw + 1):
                    window = [T[j] for j in range(start_idx, start_idx + mw)]
                    lhs = sum(q_gene[q_index[(a, tau)]] for tau in window)
                    if lhs <= rhs:
                        continue

                    found_violation = True
                    window_combos = []
                    for tau in window:
                        window_combos.extend(combos_by_aircraft_period[(a, tau)])
                    # Remove first from lowest profit per hour.
                    ordered = sorted(
                        window_combos,
                        key=lambda key: pi_delta.get(key, 0.0) / max(1e-9, float(h[key]))
                    )
                    removed = False
                    for key in ordered:
                        idx = x_index[key]
                        if x_gene[idx] > 0:
                            x_gene[idx] -= 1
                            removed = True
                            break
                    if not removed:
                        return
                    break
                if found_violation:
                    break

            if not found_violation:
                return

    def deactivate_empty_routes(x_gene: List[int], y_gene: List[int]) -> None:
        """Deactivate y[r,t] if no flights remain after repairs."""
        for r in routes:
            for t in periods:
                if route_total(x_gene, r, t) <= 0:
                    y_gene[y_index[(r, t)]] = 0

    def repair_chromosome(chrom: Dict[str, List[int]], run_greedy_fill: bool = False) -> Dict[str, List[int]]:
        """
        Main model-aware repair function.

        It repairs bounds, temporal route logic, category coverage, minimum
        service, excessive capacity usage, and q deployment.
        """
        x_gene = chrom["x"][:]
        y_gene = chrom["y"][:]

        # Bound x.
        for key, i in x_index.items():
            x_gene[i] = max(0, min(int(round(x_gene[i])), int(M[key])))

        # Bound and repair y.
        y_gene = [1 if int(v) >= 1 else 0 for v in y_gene]
        y_gene = repair_temporal_y(y_gene)
        y_gene = repair_category_coverage(y_gene)

        # Force x = 0 when y = 0.
        for (r, t), relevant in combos_by_route_period.items():
            if y_gene[y_index[(r, t)]] == 0:
                for key in relevant:
                    x_gene[x_index[key]] = 0

        # Add minimum flights where routes are active.
        add_minimum_service(x_gene, y_gene)

        # Remove obvious excesses before computing q.
        trim_route_and_demand_excess(x_gene)
        trim_aircraft_capacity_excess(x_gene)
        trim_rolling_maintenance_excess(x_gene)

        # Deactivate empty routes, then repair temporal/category again because
        # trimming may have removed all flights from a route-period.
        deactivate_empty_routes(x_gene, y_gene)
        y_gene = repair_temporal_y(y_gene)
        y_gene = repair_category_coverage(y_gene)
        add_minimum_service(x_gene, y_gene)
        trim_route_and_demand_excess(x_gene)
        trim_aircraft_capacity_excess(x_gene)
        trim_rolling_maintenance_excess(x_gene)

        q_gene = recompute_minimum_q(x_gene)

        if run_greedy_fill and use_greedy_fill:
            chrom2 = improve_flight_frequencies({"x": x_gene, "y": y_gene, "q": q_gene})
            x_gene = chrom2["x"]
            y_gene = chrom2["y"]
            q_gene = chrom2["q"]

        return {"x": x_gene, "y": y_gene, "q": q_gene}

    # -------------------------------------------------------------------------
    # Greedy local search: fill profitable flights
    # -------------------------------------------------------------------------
    def improve_flight_frequencies(chrom: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """
        Greedily add profitable flights to already-active route-periods.

        This directly targets the weakness seen in the previous GA: it was
        feasible but scheduled too few flights compared with Gurobi.
        """
        x_gene = chrom["x"][:]
        y_gene = chrom["y"][:]
        q_gene = recompute_minimum_q(x_gene)

        candidate_combos = sorted(
            combos,
            key=lambda key: pi_delta.get(key, 0.0) / max(1.0, float(h[key])),
            reverse=True,
        )

        for _pass in range(max(0, int(greedy_fill_passes))):
            added_any = False

            for key in candidate_combos:
                r, t, a = key
                idx = x_index[key]

                if y_gene[y_index[(r, t)]] == 0:
                    continue
                if pi_delta[key] <= 0:
                    continue
                if x_gene[idx] >= int(M[key]):
                    continue

                route_remaining = int(U_route.get((r, t), 0)) - route_total(x_gene, r, t)
                hub_remaining = int(HubSlot.get(t, 0)) - period_total(x_gene, t)
                demand_remaining = int(math.floor(
                    max(0.0, float(eta.get((r, t), 1.20)) * float(Demand.get((r, t), 1.0)) - route_seats(x_gene, r, t))
                    / max(1.0, float(Seats[a]))
                ))
                bound_remaining = int(M[key]) - x_gene[idx]

                unit_hours = float(alpha) * float(H_unit.get((a, t), 0.0))
                if unit_hours <= 1e-9:
                    continue
                current_hours = aircraft_hours(x_gene, a, t)
                hour_remaining = int(math.floor(
                    max(0.0, unit_hours * int(FleetSize[a]) - current_hours)
                    / max(1e-9, float(h[key]))
                ))

                max_add = min(bound_remaining, route_remaining, hub_remaining, demand_remaining, hour_remaining)
                if max_add <= 0:
                    continue

                # Try the largest possible add first, then reduce if rolling
                # maintenance or marginal fixed aircraft cost makes it unattractive.
                chosen_add = 0
                chosen_q = None
                chosen_gain = None

                for add in range(max_add, 0, -1):
                    old_q_val = q_gene[q_index[(a, t)]]
                    new_hours = current_hours + add * float(h[key])
                    new_q_val = int(math.ceil(new_hours / unit_hours))
                    if new_q_val > int(FleetSize[a]):
                        continue

                    q_test = q_gene[:]
                    q_test[q_index[(a, t)]] = new_q_val
                    if not rolling_maintenance_ok(q_test, a):
                        continue

                    marginal_gain = add * float(pi_delta[key])
                    if marginal_gain > 1e-7:
                        chosen_add = add
                        chosen_q = q_test
                        chosen_gain = marginal_gain
                        break

                if chosen_add > 0 and chosen_q is not None and chosen_gain is not None:
                    x_gene[idx] += chosen_add
                    q_gene = chosen_q
                    added_any = True

            if not added_any:
                break

        return {"x": x_gene, "y": y_gene, "q": recompute_minimum_q(x_gene)}

    # -------------------------------------------------------------------------
    # Fitness evaluation
    # -------------------------------------------------------------------------
    def evaluate(chrom: Dict[str, List[int]], collect_violations: bool = False) -> Dict[str, Any]:
        """Compute objective, penalty, fitness, and violation details."""
        chrom = repair_chromosome(chrom, run_greedy_fill=False)
        x_gene = chrom["x"]
        y_gene = chrom["y"]
        q_gene = chrom["q"]
        z_gene = derive_z_from_y(y_gene)

        revenue_profit = sum(float(pi_delta[key]) * x_gene[x_index[key]] for key in combos)
        startup_cost = sum(
            float(SC.get(r, 0.0)) * z_gene[y_index[(r, t)]]
            for r in routes for t in periods
        )
        route_cost = sum(
            float(RC.get((r, t), 0.0)) * y_gene[y_index[(r, t)]]
            for r in routes for t in periods
        )
        fleet_cost = sum(
            float(FC.get(a, 0.0)) * q_gene[q_index[(a, t)]]
            for a in aircraft for t in periods
        )
        objective = revenue_profit - startup_cost - route_cost - fleet_cost

        total_violation = 0.0
        violations = []

        def add_violation(name: str, amount: float) -> None:
            nonlocal total_violation
            amount = float(amount)
            if amount > 1e-7:
                total_violation += amount
                if collect_violations:
                    violations.append((name, amount))

        # (1) Link x and y.
        for key in combos:
            r, t, a = key
            lhs = x_gene[x_index[key]]
            rhs = int(M[key]) * y_gene[y_index[(r, t)]]
            add_violation(f"link_x_y[{r},{t},{a}]", max(0.0, lhs - rhs))

        # (2) Minimum service and (3) route max.
        for r in routes:
            for t in periods:
                relevant = combos_by_route_period[(r, t)]
                y_val = y_gene[y_index[(r, t)]]
                total_x = sum(x_gene[x_index[key]] for key in relevant)
                if relevant:
                    add_violation(f"min_service[{r},{t}]", max(0.0, int(L.get((r, t), 0)) * y_val - total_x))
                    route_ub = int(U_route.get((r, t), sum(int(M.get(key, 0)) for key in relevant)))
                    add_violation(f"route_max_freq[{r},{t}]", max(0.0, total_x - route_ub * y_val))
                else:
                    add_violation(f"no_feasible_aircraft[{r},{t}]", y_val)

        # (4) Fleet hours.
        for a in aircraft:
            for t in periods:
                hours_used = aircraft_hours(x_gene, a, t)
                capacity = float(alpha) * float(H_unit.get((a, t), 0.0)) * q_gene[q_index[(a, t)]]
                add_violation(f"physical_fleet_hours[{a},{t}]", max(0.0, hours_used - capacity))

        # (5) Fleet size bounds.
        for a in aircraft:
            for t in periods:
                q_val = q_gene[q_index[(a, t)]]
                add_violation(f"fleet_size[{a},{t}]", max(0.0, q_val - int(FleetSize[a])))
                add_violation(f"q_nonnegative[{a},{t}]", max(0.0, -q_val))

        # (6) Rolling maintenance.
        if maintenance_window is not None and max_active_periods is not None:
            mw = int(maintenance_window)
            map_ = int(max_active_periods)
            if mw > 0 and map_ > 0 and n_T >= mw:
                for a in aircraft:
                    for start_idx in range(0, n_T - mw + 1):
                        window = [T[j] for j in range(start_idx, start_idx + mw)]
                        lhs = sum(q_gene[q_index[(a, tau)]] for tau in window)
                        rhs = map_ * int(FleetSize[a])
                        add_violation(f"rolling_maint[{a},{T[start_idx]}]", max(0.0, lhs - rhs))

        # (7) Demand / seat cap.
        for r in routes:
            for t in periods:
                seats_used = route_seats(x_gene, r, t)
                cap = float(eta.get((r, t), 1.20)) * float(Demand.get((r, t), 1.0))
                add_violation(f"demand_seat_cap[{r},{t}]", max(0.0, seats_used - cap))

        # (8) Hub slots.
        for t in periods:
            total_x = period_total(x_gene, t)
            hub_cap = int(HubSlot.get(t, sum(int(M.get(key, 0)) for key in combos_by_period[t])))
            add_violation(f"hub_slot_cap[{t}]", max(0.0, total_x - hub_cap))

        # (9) Category coverage.
        for c in categories:
            for t in periods:
                routes_in_category = [r for r in R_c.get(c, []) if (r, t) in y_index]
                lhs = sum(y_gene[y_index[(r, t)]] for r in routes_in_category)
                rhs = int(K.get((c, t), 0))
                add_violation(f"category_coverage[{c},{t}]", max(0.0, rhs - lhs))

        # (11) Minimum up-time.
        N = int(N_min)
        if N > 0:
            for r in routes:
                for start_idx in range(0, n_T - N + 1):
                    t = T[start_idx]
                    window = [T[j] for j in range(start_idx, start_idx + N)]
                    lhs = sum(y_gene[y_index[(r, tau)]] for tau in window)
                    rhs = N * z_gene[y_index[(r, t)]]
                    add_violation(f"min_up_time[{r},{t}]", max(0.0, rhs - lhs))

        # (12) No late opening.
        if N_min > 1 and n_T >= int(N_min):
            late_start_idx = n_T - int(N_min) + 1
            for r in routes:
                for idx in range(late_start_idx, n_T):
                    t = T[idx]
                    z_val = z_gene[y_index[(r, t)]]
                    add_violation(f"no_late_open[{r},{t}]", z_val)

        # (13) Optional loss-risk cap.
        if Omega is not None:
            negative_profit_combos = [key for key in combos if pi_delta[key] < 0]
            loss_value = sum(abs(pi_delta[key]) * x_gene[x_index[key]] for key in negative_profit_combos)
            add_violation("loss_risk_cap", max(0.0, loss_value - float(Omega)))

        # (14) Optional non-fuel operating cost budget.
        if B_nfoc is not None:
            nfoc_value = sum(float(nfoc.get(key, 0.0)) * x_gene[x_index[key]] for key in combos)
            add_violation("nfoc_budget", max(0.0, nfoc_value - float(B_nfoc)))

        penalty = float(penalty_weight) * total_violation
        fitness = objective - penalty

        return {
            "chromosome": chrom,
            "z_gene": z_gene,
            "objective": objective,
            "penalty": penalty,
            "fitness": fitness,
            "total_violation": total_violation,
            "violations": violations,
        }

    # -------------------------------------------------------------------------
    # Initial population
    # -------------------------------------------------------------------------
    def create_empty_chromosome() -> Dict[str, List[int]]:
        return {
            "x": [0 for _ in combos],
            "y": [0 for _ in route_period_keys],
            "q": [0 for _ in aircraft_period_keys],
        }

    def create_random_chromosome() -> Dict[str, List[int]]:
        """Random but model-aware chromosome."""
        chrom = create_empty_chromosome()
        y_gene = chrom["y"]
        x_gene = chrom["x"]

        # Activate route-periods randomly.
        for r in routes:
            for t in periods:
                if combos_by_route_period[(r, t)] and random.random() < active_probability:
                    y_gene[y_index[(r, t)]] = 1

        # Random initial frequencies for active route-periods.
        for r in routes:
            for t in periods:
                if y_gene[y_index[(r, t)]] == 0:
                    continue
                relevant = combos_by_route_period[(r, t)]
                if not relevant:
                    continue
                min_required = int(L.get((r, t), 0))
                max_possible = min(
                    int(U_route.get((r, t), 0)),
                    sum(int(M[key]) for key in relevant),
                )
                if max_possible <= 0:
                    continue
                total_flights = random.randint(min(min_required, max_possible), max_possible)
                ordered = relevant[:]
                random.shuffle(ordered)
                remaining = total_flights
                for key in ordered:
                    if remaining <= 0:
                        break
                    add = random.randint(0, min(int(M[key]), remaining))
                    x_gene[x_index[key]] += add
                    remaining -= add
                for key in ordered:
                    if remaining <= 0:
                        break
                    idx = x_index[key]
                    add = min(int(M[key]) - x_gene[idx], remaining)
                    x_gene[idx] += add
                    remaining -= add

        return repair_chromosome(chrom, run_greedy_fill=True)

    def create_greedy_chromosome(diversity: float = 0.15) -> Dict[str, List[int]]:
        """
        Semi-greedy chromosome seed.

        It activates required category routes and additional high-profit route
        periods, then uses greedy fill to add profitable flight frequencies.
        """
        chrom = create_empty_chromosome()
        y_gene = chrom["y"]

        # First satisfy category coverage using high-potential routes.
        for c in categories:
            for t in periods:
                required_k = int(K.get((c, t), 0))
                if required_k <= 0:
                    continue
                candidates = [r for r in R_c.get(c, []) if combos_by_route_period[(r, t)]]
                candidates.sort(key=lambda r_: route_period_potential.get((r_, t), 0.0), reverse=True)
                # Add some randomization among good candidates.
                top_pool = candidates[:max(required_k, min(len(candidates), required_k + 3))]
                random.shuffle(top_pool)
                selected = sorted(top_pool[:required_k], key=lambda r_: route_period_potential.get((r_, t), 0.0), reverse=True)
                for r in selected:
                    activate_block_containing_period(y_gene, r, t)

        # Then add profitable route-periods with semi-greedy randomized threshold.
        scored_route_periods = sorted(
            route_period_keys,
            key=lambda rt: route_period_potential.get(rt, 0.0),
            reverse=True,
        )
        for r, t in scored_route_periods:
            potential = route_period_potential.get((r, t), 0.0)
            if potential <= 0:
                continue
            # Stronger potential -> higher activation probability.
            if random.random() < max(0.05, min(0.85, active_probability + diversity)):
                activate_block_containing_period(y_gene, r, t)

        return repair_chromosome(chrom, run_greedy_fill=True)

    n_greedy = int(round(float(greedy_seed_fraction) * int(population_size)))
    n_greedy = max(0, min(n_greedy, int(population_size)))
    population = []
    for _ in range(n_greedy):
        population.append(create_greedy_chromosome(diversity=random.uniform(0.0, 0.25)))
    while len(population) < int(population_size):
        population.append(create_random_chromosome())

    # -------------------------------------------------------------------------
    # GA operators
    # -------------------------------------------------------------------------
    def tournament_select(evaluated_population: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        k = max(1, min(int(tournament_size), len(evaluated_population)))
        competitors = random.sample(evaluated_population, k)
        winner = max(competitors, key=lambda item: item["fitness"])
        return _clone_chrom(winner["chromosome"])

    def route_based_crossover(
        parent_1: Dict[str, List[int]],
        parent_2: Dict[str, List[int]],
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """
        Route-based crossover.

        A child inherits the complete route schedule from one parent route by
        route. This preserves temporal route patterns better than flat cuts.
        """
        if random.random() > crossover_rate:
            return _clone_chrom(parent_1), _clone_chrom(parent_2)

        child_1 = create_empty_chromosome()
        child_2 = create_empty_chromosome()

        for r in routes:
            source_for_child_1 = parent_1 if random.random() < 0.5 else parent_2
            source_for_child_2 = parent_2 if source_for_child_1 is parent_1 else parent_1

            # Copy y route pattern.
            for t in periods:
                yi = y_index[(r, t)]
                child_1["y"][yi] = source_for_child_1["y"][yi]
                child_2["y"][yi] = source_for_child_2["y"][yi]

            # Copy x decisions for this route.
            for key in combos_by_route[r]:
                xi = x_index[key]
                child_1["x"][xi] = source_for_child_1["x"][xi]
                child_2["x"][xi] = source_for_child_2["x"][xi]

        # q is derived by repair, so no need to inherit random q values.
        return (
            repair_chromosome(child_1, run_greedy_fill=True),
            repair_chromosome(child_2, run_greedy_fill=True),
        )

    def mutate_route_blocks(chrom: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Mutation using route blocks instead of independent y flips."""
        child = _clone_chrom(chrom)
        y_gene = child["y"]
        x_gene = child["x"]

        N = max(1, int(N_min))

        # Mutate route activation in blocks.
        for _ in range(max(1, int(route_mutations_per_child))):
            r = random.choice(routes)
            op = random.choice(["open", "close", "extend"])

            if op == "open":
                if n_T >= N:
                    start = random.randint(0, n_T - N)
                    length = random.randint(N, min(n_T - start, N + 3))
                    for j in range(start, start + length):
                        if combos_by_route_period[(r, T[j])]:
                            y_gene[y_index[(r, T[j])]] = 1

            elif op == "close":
                active_positions = [i for i, t in enumerate(T) if y_gene[y_index[(r, t)]] == 1]
                if active_positions:
                    start = random.choice(active_positions)
                    length = random.randint(1, min(n_T - start, max(N, 3)))
                    for j in range(start, start + length):
                        t = T[j]
                        y_gene[y_index[(r, t)]] = 0
                        for key in combos_by_route_period[(r, t)]:
                            x_gene[x_index[key]] = 0

            else:  # extend
                active_positions = [i for i, t in enumerate(T) if y_gene[y_index[(r, t)]] == 1]
                if active_positions:
                    i = random.choice(active_positions)
                    for j in [i - 1, i + 1]:
                        if 0 <= j < n_T and combos_by_route_period[(r, T[j])]:
                            y_gene[y_index[(r, T[j])]] = 1

        # Mutate flight frequencies locally.
        for _ in range(max(0, int(x_mutations_per_child))):
            key = random.choice(combos)
            r, t, _a = key
            idx = x_index[key]
            if y_gene[y_index[(r, t)]] == 0:
                continue
            if random.random() < mutation_rate:
                if random.random() < 0.75:
                    x_gene[idx] += random.choice([-3, -2, -1, 1, 2, 3])
                else:
                    x_gene[idx] = random.randint(0, max(0, int(M[key])))

        return repair_chromosome(child, run_greedy_fill=True)

    # -------------------------------------------------------------------------
    # Evolution loop
    # -------------------------------------------------------------------------
    best_eval = None
    best_generation = 0
    generations_without_improvement = 0
    history = []

    for gen in range(int(generations)):
        evaluated = [evaluate(chrom, collect_violations=False) for chrom in population]
        evaluated.sort(key=lambda item: item["fitness"], reverse=True)

        current_best = evaluated[0]
        current_avg_fitness = sum(item["fitness"] for item in evaluated) / len(evaluated)
        history.append({
            "generation": gen,
            "best_fitness": current_best["fitness"],
            "best_objective": current_best["objective"],
            "best_total_violation": current_best["total_violation"],
            "avg_fitness": current_avg_fitness,
        })

        if best_eval is None or current_best["fitness"] > best_eval["fitness"]:
            best_eval = current_best
            best_generation = gen
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if verbose and (gen == 0 or (print_every > 0 and gen % print_every == 0) or gen == int(generations) - 1):
            print(
                f"Generation {gen:4d} | "
                f"Best fitness: {current_best['fitness']:.2f} | "
                f"Objective: {current_best['objective']:.2f} | "
                f"Violation: {current_best['total_violation']:.4f}"
            )

        if generations_without_improvement >= int(stall_generations):
            if verbose:
                print(f"Stopping early at generation {gen}: no improvement for {stall_generations} generations.")
            break

        # Elitism: keep the best solutions exactly.
        elite_n = max(0, min(int(elitism_count), len(evaluated)))
        next_population = [_clone_chrom(evaluated[i]["chromosome"]) for i in range(elite_n)]

        # Generate children.
        while len(next_population) < int(population_size):
            parent_1 = tournament_select(evaluated)
            parent_2 = tournament_select(evaluated)
            child_1, child_2 = route_based_crossover(parent_1, parent_2)
            child_1 = mutate_route_blocks(child_1)
            child_2 = mutate_route_blocks(child_2)
            next_population.append(child_1)
            if len(next_population) < int(population_size):
                next_population.append(child_2)

        population = next_population

    # Final detailed evaluation.
    final_eval = evaluate(best_eval["chromosome"], collect_violations=True)
    final_chrom = final_eval["chromosome"]
    final_z = final_eval["z_gene"]

    x_vals = {
        key: int(final_chrom["x"][idx])
        for key, idx in x_index.items()
        if int(final_chrom["x"][idx]) > 0
    }
    y_vals = {
        key: 1
        for key, idx in y_index.items()
        if int(final_chrom["y"][idx]) > 0
    }
    z_vals = {
        key: 1
        for key, idx in y_index.items()
        if int(final_z[idx]) > 0
    }
    q_vals = {
        key: int(final_chrom["q"][idx])
        for key, idx in q_index.items()
        if int(final_chrom["q"][idx]) > 0
    }
    aircraft_hours_vals = {}
    for a in aircraft:
        for t in periods:
            hours_used = aircraft_hours(final_chrom["x"], a, t)
            if hours_used > 1e-6:
                aircraft_hours_vals[(a, t)] = float(hours_used)

    runtime = time.time() - start_time
    status = "FEASIBLE_GA" if final_eval["total_violation"] <= 1e-6 else "PENALIZED_INFEASIBLE_GA"

    if verbose:
        total_flights = sum(x_vals.values())
        print("\n========== IMPROVED GA RESULT ==========")
        print(f"Status: {status}")
        print(f"Best generation: {best_generation}")
        print(f"Best raw objective value: {final_eval['objective']:.4f}")
        print(f"Best penalized fitness: {final_eval['fitness']:.4f}")
        print(f"Total violation: {final_eval['total_violation']:.6f}")
        print(f"Total flights: {total_flights}")
        print(f"Active route-periods: {len(y_vals)}")
        print(f"Route openings: {len(z_vals)}")
        print(f"sum_q: {sum(q_vals.values())}")
        print(f"Runtime: {runtime:.2f} seconds")
        print("Violated constraints:")
        if final_eval["violations"]:
            for name, amount in final_eval["violations"][:50]:
                print(f"  {name}: violation = {amount:.6f}")
            if len(final_eval["violations"]) > 50:
                print(f"  ... {len(final_eval['violations']) - 50} more violations not printed")
        else:
            print("  None")

    return {
        "status": status,
        "obj_value": final_eval["objective"],
        "fitness": final_eval["fitness"],
        "penalty": final_eval["penalty"],
        "total_violation": final_eval["total_violation"],
        "violated_constraints": final_eval["violations"],
        "runtime": runtime,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "z_vals": z_vals,
        "q_vals": q_vals,
        "aircraft_hours_vals": aircraft_hours_vals,
        "delta": delta,
        "alpha": alpha,
        "N_min": N_min,
        "maintenance_window": maintenance_window,
        "max_active_periods": max_active_periods,
        "Omega": Omega,
        "B_nfoc": B_nfoc,
        "population_size": population_size,
        "generations_requested": generations,
        "best_generation": best_generation,
        "history": history,
        "penalty_weight": penalty_weight,
        "n_valid_combos": len(combos),
        "n_x_genes": len(combos),
        "n_y_genes": len(route_period_keys),
        "n_q_genes": len(aircraft_period_keys),
        "algorithm_notes": {
            "q_repair": "q[a,t] is recalculated as the minimum aircraft deployment required by assigned hours.",
            "crossover": "Route-based crossover preserving full route schedules.",
            "mutation": "Route-block mutation plus local flight-frequency mutation.",
            "local_search": "Greedy fill adds profitable flights while respecting key capacity constraints.",
            "feasibility": "Remaining violations are penalized and reported.",
        },
    }


# Backward-compatible alias if your runner imports build_and_solve_ga.
def build_and_solve_ga(*args, **kwargs):
    return build_and_solve_ga_improved(*args, **kwargs)


if __name__ == "__main__":
    print(
        "This file defines build_and_solve_ga_improved(data, ...).\n"
        "Example:\n"
        "    from data_loader import load_data\n"
        "    from ga_model_v3_improved import build_and_solve_ga_improved\n"
        "    data = load_data()\n"
        "    results = build_and_solve_ga_improved(data)\n"
    )
