"""
model_v2_final.py
 Gurobi MILP for the airline route-frequency / fleet planning project.
"""

import time
import math
from typing import Dict, Any, Tuple, List

import gurobipy as gp
from gurobipy import GRB


STATUS_MAP = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
}


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
    """
    Add final-model parameters if an older data_loader.py output is passed.

    This prevents KeyError and keeps the model usable even if Spyder imports an
    old loader by mistake. The corrected data_loader.py already returns these
    keys, so this function usually does nothing.
    """
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
        # Use old key if available, otherwise 1.20
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
            + ". Check that main_v2.py imports the corrected data_loader.py."
        )


def build_and_solve(
    data: Dict[str, Any],
    delta: float = 0.0,
    alpha: float = 1.0,
    N_min: int = 2,
    maintenance_window: int = 5,
    max_active_periods: int = 3,
    Omega=None,
    B_nfoc=None,
    time_limit: int = 3600,
    mip_gap: float = 0.00,
    verbose: bool = True,
    compute_iis: bool = False,
) -> Dict[str, Any]:
    """
    Build and solve the final strategic airline route-frequency MILP.

    Important:
    gamma_maint is intentionally removed. Maintenance is modeled as:
        sum_{tau in a rolling window} q[a,tau]
        <= max_active_periods * FleetSize[a]
    """

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

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    routes = list(data["routes"])
    periods = sorted(list(data["periods"]))
    aircraft = list(data["aircraft"])
    categories = list(data["categories"])
    R_c = data["R_c"]
    raw_combos = sorted(list(data["combos"]))

    T = periods
    n_T = len(T)
    t_idx = {t: i for i, t in enumerate(T)}

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
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

    # Valid set G: only existing and compatible combinations
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

    # Scenario-adjusted profit
    pi_delta = {
        (r, t, a): float(pi[(r, t, a)]) - float(delta) * float(f[(r, t, a)])
        for (r, t, a) in combos
    }
    n_neg_pi_combos = sum(1 for key in combos if pi_delta[key] < 0)

    # ------------------------------------------------------------------
    # Build Gurobi model
    # ------------------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()

    m = gp.Model("Final_Strategic_Fleet_Planning_MILP", env=env)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    x = {
        (r, t, a): m.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            ub=max(0, int(M[(r, t, a)])),
            name=f"x[{r},{t},{a}]",
        )
        for (r, t, a) in combos
    }

    y = {
        (r, t): m.addVar(vtype=GRB.BINARY, name=f"y[{r},{t}]")
        for r in routes for t in periods
    }

    z = {
        (r, t): m.addVar(vtype=GRB.BINARY, name=f"z[{r},{t}]")
        for r in routes for t in periods
    }

    q = {
        (a, t): m.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            ub=max(0, int(FleetSize[a])),
            name=f"q[{a},{t}]",
        )
        for a in aircraft for t in periods
    }

    m.update()

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    m.setObjective(
        gp.quicksum(pi_delta[(r, t, a)] * x[(r, t, a)] for (r, t, a) in combos)
        - gp.quicksum(float(SC.get(r, 0.0)) * z[(r, t)] for r in routes for t in periods)
        - gp.quicksum(float(RC.get((r, t), 0.0)) * y[(r, t)] for r in routes for t in periods)
        - gp.quicksum(float(FC.get(a, 0.0)) * q[(a, t)] for a in aircraft for t in periods),
        GRB.MAXIMIZE,
    )

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    # (1) Route activation linking
    for (r, t, a) in combos:
        m.addConstr(
            x[(r, t, a)] <= int(M[(r, t, a)]) * y[(r, t)],
            name=f"link_x_y[{r},{t},{a}]",
        )

    # (2) Minimum service and (3) route-period max frequency
    for r in routes:
        for t in periods:
            relevant = [(r, t, a) for a in aircraft if (r, t, a) in x]

            if relevant:
                m.addConstr(
                    gp.quicksum(x[(r, t, a)] for (r, t, a) in relevant)
                    >= int(L.get((r, t), 0)) * y[(r, t)],
                    name=f"min_service[{r},{t}]",
                )

                m.addConstr(
                    gp.quicksum(x[(r, t, a)] for (r, t, a) in relevant)
                    <= int(U_route.get((r, t), sum(M.get((r, t, a), 0) for a in aircraft))) * y[(r, t)],
                    name=f"route_max_freq[{r},{t}]",
                )
            else:
                # No feasible aircraft for this route-period, so it cannot be active.
                m.addConstr(y[(r, t)] == 0, name=f"no_feasible_aircraft[{r},{t}]")

    # (4) Physical fleet-hour capacity using q[a,t]
    for a in aircraft:
        for t in periods:
            relevant = [(r, t, a) for r in routes if (r, t, a) in x]

            if relevant:
                m.addConstr(
                    gp.quicksum(float(h[(r, t, a)]) * x[(r, t, a)] for (r, t, a) in relevant)
                    <= float(alpha) * float(H_unit.get((a, t), 0.0)) * q[(a, t)],
                    name=f"physical_fleet_hours[{a},{t}]",
                )
            else:
                m.addConstr(q[(a, t)] == 0, name=f"no_aircraft_needed[{a},{t}]")

    # (5) q[a,t] <= FleetSize[a] is enforced by q upper bound.

    # (6) Rolling-window maintenance using q[a,t].
    # If the horizon is shorter than maintenance_window, this constraint is skipped.
    if maintenance_window is not None and max_active_periods is not None:
        maintenance_window = int(maintenance_window)
        max_active_periods = int(max_active_periods)

        if maintenance_window > 0 and max_active_periods > 0 and n_T >= maintenance_window:
            for a in aircraft:
                for start_idx in range(0, n_T - maintenance_window + 1):
                    window = [T[j] for j in range(start_idx, start_idx + maintenance_window)]
                    m.addConstr(
                        gp.quicksum(q[(a, tau)] for tau in window)
                        <= max_active_periods * int(FleetSize[a]),
                        name=f"rolling_maint[{a},{T[start_idx]}]",
                    )

    # (7) Demand / seat capacity
    for r in routes:
        for t in periods:
            relevant = [(r, t, a) for a in aircraft if (r, t, a) in x]
            if relevant:
                m.addConstr(
                    gp.quicksum(float(Seats[a]) * x[(r, t, a)] for (r, t, a) in relevant)
                    <= float(eta.get((r, t), 1.20)) * float(Demand.get((r, t), 1.0)),
                    name=f"demand_seat_cap[{r},{t}]",
                )

    # (8) Hub slot capacity
    for t in periods:
        relevant = [(r, t, a) for (r, tt, a) in combos if tt == t]
        if relevant:
            m.addConstr(
                gp.quicksum(x[(r, t, a)] for (r, t, a) in relevant)
                <= int(HubSlot.get(t, sum(M.get((r, t, a), 0) for (r, t, a) in relevant))),
                name=f"hub_slot_cap[{t}]",
            )

    # (9) Category coverage
    for c in categories:
        for t in periods:
            routes_in_category = [r for r in R_c.get(c, []) if (r, t) in y]
            if routes_in_category:
                m.addConstr(
                    gp.quicksum(y[(r, t)] for r in routes_in_category)
                    >= int(K.get((c, t), 0)),
                    name=f"category_coverage[{c},{t}]",
                )
            else:
                # If category has no routes in the filtered set, require zero.
                if int(K.get((c, t), 0)) > 0:
                    raise ValueError(f"Category {c} has K[{c},{t}] > 0 but no routes.")

    # (10) Opening detection
    for r in routes:
        for t in periods:
            i = t_idx[t]
            if i == 0:
                prev_active = int(Y0.get(r, 0))
            else:
                prev_t = T[i - 1]
                prev_active = y[(r, prev_t)]

            m.addConstr(z[(r, t)] >= y[(r, t)] - prev_active, name=f"open_lb[{r},{t}]")
            m.addConstr(z[(r, t)] <= y[(r, t)], name=f"open_ub_active[{r},{t}]")
            m.addConstr(z[(r, t)] <= 1 - prev_active, name=f"open_ub_prev[{r},{t}]")

    # (11) Minimum up-time after opening
    N_min = int(N_min)
    if N_min > 0:
        for r in routes:
            for start_idx in range(0, n_T - N_min + 1):
                t = T[start_idx]
                window = [T[j] for j in range(start_idx, start_idx + N_min)]
                m.addConstr(
                    gp.quicksum(y[(r, tau)] for tau in window)
                    >= N_min * z[(r, t)],
                    name=f"min_up_time[{r},{t}]",
                )

    # (12) No late openings if the minimum-up-time window cannot fit
    if N_min > 1 and n_T >= N_min:
        late_start_idx = n_T - N_min + 1
        for r in routes:
            for idx in range(late_start_idx, n_T):
                t = T[idx]
                m.addConstr(z[(r, t)] == 0, name=f"no_late_open[{r},{t}]")

    # (13) Optional loss-risk cap
    if Omega is not None:
        negative_profit_combos = [key for key in combos if pi_delta[key] < 0]
        if negative_profit_combos:
            m.addConstr(
                gp.quicksum(abs(pi_delta[key]) * x[key] for key in negative_profit_combos)
                <= float(Omega),
                name="loss_risk_cap",
            )

    # (14) Optional non-fuel operating cost budget
    if B_nfoc is not None:
        m.addConstr(
            gp.quicksum(float(nfoc.get(key, 0.0)) * x[key] for key in combos)
            <= float(B_nfoc),
            name="nfoc_budget",
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    start = time.time()
    m.optimize()
    runtime = time.time() - start

    status_code = m.Status
    status_str = STATUS_MAP.get(status_code, str(status_code))

    if status_code == GRB.INFEASIBLE and compute_iis:
        try:
            m.computeIIS()
            m.write("infeasible_model.ilp")
        except gp.GurobiError:
            pass

    results = {
        "status": status_str,
        "obj_value": None,
        "runtime": runtime,
        "x_vals": {},
        "y_vals": {},
        "z_vals": {},
        "q_vals": {},
        "aircraft_hours_vals": {},
        "model": m,
        "delta": delta,
        "alpha": alpha,
        "N_min": N_min,
        "maintenance_window": maintenance_window,
        "max_active_periods": max_active_periods,
        "Omega": Omega,
        "B_nfoc": B_nfoc,
        "n_vars": m.NumVars,
        "n_constrs": m.NumConstrs,
        "n_neg_pi_combos": n_neg_pi_combos,
    }

    if status_code in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        try:
            results["obj_value"] = m.ObjVal

            for key in combos:
                val = x[key].X
                if val > 0.5:
                    results["x_vals"][key] = int(round(val))

            for r in routes:
                for t in periods:
                    if y[(r, t)].X > 0.5:
                        results["y_vals"][(r, t)] = 1
                    if z[(r, t)].X > 0.5:
                        results["z_vals"][(r, t)] = 1

            for a in aircraft:
                for t in periods:
                    q_val = q[(a, t)].X
                    if q_val > 0.5:
                        results["q_vals"][(a, t)] = int(round(q_val))

                    hours = sum(
                        float(h[(r, t, a)]) * x[(r, t, a)].X
                        for r in routes
                        if (r, t, a) in x
                    )
                    if hours > 1e-6:
                        results["aircraft_hours_vals"][(a, t)] = float(hours)

        except gp.GurobiError:
            pass

    return results
