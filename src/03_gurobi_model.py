"""
model.py
--------
Gurobi implementation of the Profit-Optimal Flight Planning MILP.

Sets  : R (routes), T (periods), A (aircraft), C (categories), R_c ⊆ R
Params: pi_rta, h_rta, H_at, M_rta, L_rt, K_ct, f_rta
Vars  : x_rta ∈ Z+,  y_rt ∈ {0,1}

"""

import time
import gurobipy as gp
from gurobipy import GRB

# Status code mapping for better readability in results
STATUS_MAP = {
    GRB.OPTIMAL:    'OPTIMAL',
    GRB.INFEASIBLE: 'INFEASIBLE',
    GRB.INF_OR_UNBD:'INF_OR_UNBD',
    GRB.UNBOUNDED:  'UNBOUNDED',
    GRB.TIME_LIMIT: 'TIME_LIMIT',
    GRB.SUBOPTIMAL: 'SUBOPTIMAL',
}

def build_and_solve(data, delta=0.0, alpha=1.0, time_limit=300, mip_gap=0.01,
                    verbose=True):
    """
    Build and solve the MILP for a given scenario (delta, alpha).
    Parameters
    ----------
    data       : dict  returned by data_loader.load_data()
    delta      : float fuel-price shock factor ∈ {0, 0.10, 0.25}
    alpha      : float capacity multiplier (overrides data['H'] if != 1.0)
    time_limit : int   Gurobi TimeLimit in seconds
    mip_gap    : float Gurobi MIPGap tolerance
    verbose    : bool  print Gurobi log

    Returns
    -------
    results : dict  {status, obj_value, runtime, x_vals, y_vals, model}
    """
    routes     = data['routes']
    periods    = data['periods']
    aircraft   = data['aircraft']
    categories = data['categories']
    R_c        = data['R_c']
    combos     = data['combos']        # set of (r, t, a) in data

    pi   = data['pi']
    h    = data['h']
    f    = data['f']
    M    = data['M']
    L    = data['L']
    K    = data['K']

    # Recalculate H_at with the given alpha (allows scenario override)
    H_bar = data['H_bar']
    H_at  = {key: alpha * val for key, val in H_bar.items()}

  
    # Scenario-adjusted profit:  π^(δ)_rta = π_rta - δ · f_rta        
    pi_delta = {(r, t, a): pi[(r, t, a)] - delta * f[(r, t, a)]
                for (r, t, a) in combos}

    # Build Gurobi model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1 if verbose else 0)
    env.start()

    m = gp.Model("FlightPlanning_MILP", env=env)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap',    mip_gap)

    # Decision variables
    # x_rta ∈ Z+ (number of flights on route r in period t with aircraft a)
    # y_rt  ∈ {0,1} (binary: is route r served in period t?)

    x = {}
    for (r, t, a) in combos:
        x[(r, t, a)] = m.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            ub=M[(r, t, a)],   # constraint (5): x_rta ≤ M_rta
            name=f"x[{r},{t},{a}]"
        )

    y = {}
    for r in routes:
        for t in periods:
            y[(r, t)] = m.addVar(
                vtype=GRB.BINARY,
                name=f"y[{r},{t}]"
            )

    m.update()

    # Objective: Maximize total expected profit (1)
    m.setObjective(
        gp.quicksum(pi_delta[(r, t, a)] * x[(r, t, a)]
                    for (r, t, a) in combos),
        GRB.MAXIMIZE
    )

    # Constraint (2): Fleet-hour capacity per aircraft type per period
    #   Σ_r  h_rta · x_rta  ≤  H_at    ∀ a, t
    for a in aircraft:
        for t in periods:
            if (a, t) not in H_at:
                continue
            rhs = H_at[(a, t)]
            relevant = [(r, t, a) for r in routes if (r, t, a) in combos]
            if relevant:
                m.addConstr(
                    gp.quicksum(h[(r, t, a)] * x[(r, t, a)]
                                for (r, t, a) in relevant) <= rhs,
                    name=f"cap[{a},{t}]"
                )

    # Constraint (3): Linking – x_rta ≤ M_rta · y_rt    ∀ r, t, a
    # (x can only be positive if the route is activated)
    for (r, t, a) in combos:
        m.addConstr(
            x[(r, t, a)] <= M[(r, t, a)] * y[(r, t)],
            name=f"link[{r},{t},{a}]"
        )

    # Constraint (4): Minimum service – Σ_a x_rta ≥ L_rt · y_rt  ∀ r, t
    # (if route is served, at least L_rt flights must be scheduled)
    for r in routes:
        for t in periods:
            aircraft_for_rt = [(r, t, a) for a in aircraft if (r, t, a) in combos]
            if aircraft_for_rt:
                m.addConstr(
                    gp.quicksum(x[(r, t, a)] for (r, t, a) in aircraft_for_rt)
                    >= L.get((r, t), 1) * y[(r, t)],
                    name=f"minsvc[{r},{t}]"
                )

    # Constraint (6): Category coverage – Σ_{r ∈ R_c} y_rt ≥ K_ct  ∀ c, t
    # (at least K_ct routes per category per period must be served)
    for c in categories:
        for t in periods:
            routes_in_cat = [r for r in R_c[c] if (r, t) in y]
            k = K.get((c, t), 0)
            if routes_in_cat and k > 0:
                m.addConstr(
                    gp.quicksum(y[(r, t)] for r in routes_in_cat) >= k,
                    name=f"catcov[{c},{t}]"
                )

    # Solve
    start_time = time.time()
    m.optimize()
    runtime = time.time() - start_time

    # Extract results
    status_code = m.Status
    status_str  = STATUS_MAP.get(status_code, str(status_code))

    results = {
        'status':     status_str,
        'obj_value':  None,
        'runtime':    runtime,
        'x_vals':     {},
        'y_vals':     {},
        'model':      m,
        'delta':      delta,
        'alpha':      alpha,
        'n_vars':     m.NumVars,
        'n_constrs':  m.NumConstrs,
    }

    if status_code in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        try:
            results['obj_value'] = m.ObjVal
            # Active flights
            for (r, t, a) in combos:
                val = x[(r, t, a)].X
                if val > 0.5:
                    results['x_vals'][(r, t, a)] = int(round(val))
            # Active routes
            for r in routes:
                for t in periods:
                    if y[(r, t)].X > 0.5:
                        results['y_vals'][(r, t)] = 1
        except gp.GurobiError:
            pass

    return results
