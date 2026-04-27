"""
model.py
-----------------
Gurobi – Profit-Optimal Flight Planning MILP.
DS502 Path B  |  Model v2 (fixed, MDP-ready)

"""
import os
os.environ["GRB_LICENSE_FILE"] = r"C:\Users\ed024981\gurobi.lic"

import time
import gurobipy as gp
from gurobipy import GRB


STATUS_MAP = {
    GRB.OPTIMAL:     'OPTIMAL',
    GRB.INFEASIBLE:  'INFEASIBLE',
    GRB.INF_OR_UNBD: 'INF_OR_UNBD',
    GRB.UNBOUNDED:   'UNBOUNDED',
    GRB.TIME_LIMIT:  'TIME_LIMIT',
    GRB.SUBOPTIMAL:  'SUBOPTIMAL',
}


def build_and_solve(
    data,
    delta        = 0.0,
    alpha        = 1.0,
    N_min        = 2,
    Omega        = None,
    B_nfoc       = None,
    N_maint      = 2,      # how many past periods carry a maintenance penalty
    gamma_maint  = 0.20,   # fraction of H_at consumed per prior deployment
                           # 0.0 → no maintenance effect (backward-compatible)
    time_limit   = 300,
    mip_gap      = 0.01,
    verbose      = True,
):
    """
    Build and solve the fixed extended MILP.

    Key parameters
    --------------
    N_maint     : int    – lookback window for maintenance capacity reduction.
    gamma_maint : float  – fraction of H_at lost per deployment in the
                           preceding N_maint periods.  Must satisfy
                           N_maint * gamma_maint < 1 to guarantee positive
                           remaining capacity.  Default 0.20 with N_maint=2
                           → at most 40 % reduction if deployed every period.
    """

    # ── Unpack ───────────────────────────────────────────────────────────────
    routes     = data['routes']
    periods    = data['periods']
    aircraft   = data['aircraft']
    categories = data['categories']
    R_c        = data['R_c']
    combos     = data['combos']

    pi   = data['pi']
    h    = data['h']
    f    = data['f']
    M    = data['M']
    L    = data['L']
    K    = data['K']

    SC   = data.get('SC',   {r: 0.0 for r in routes})
    FC   = data.get('FC',   {a: 0.0 for a in aircraft})
    nfoc = data.get('nfoc', {c: 0.0 for c in combos})

    # Validate gamma_maint to guarantee feasibility
    if N_maint * gamma_maint >= 1.0:
        raise ValueError(
            f"N_maint * gamma_maint = {N_maint * gamma_maint:.2f} ≥ 1.0. "
            f"This would reduce available capacity to ≤ 0. "
            f"Reduce gamma_maint to < {1.0/N_maint:.3f}."
        )

    # ── Ordered periods ───────────────────────────────────────────────────────
    T     = sorted(periods)
    n_T   = len(T)
    t_idx = {t: i for i, t in enumerate(T)}

    T_open = [T[i] for i in range(1, n_T - N_min + 1)] if n_T > N_min else []
    T_late = [T[i] for i in range(max(0, n_T - N_min + 1), n_T)]

    # ── Capacity ──────────────────────────────────────────────────────────────
    H_bar = data['H_bar']
    H_at  = {key: alpha * val for key, val in H_bar.items()}

    # ── Scenario-adjusted profit ──────────────────────────────────────────────
    pi_delta = {(r, t, a): pi[(r, t, a)] - delta * f[(r, t, a)]
                for (r, t, a) in combos}

    # ── BigM for u_at linking ─────────────────────────────────────────────────
    BigM_at = {
        (a, t): sum(M[(r, t, a)] for r in routes if (r, t, a) in combos)
        for a in aircraft for t in periods
    }

    # ── Build Gurobi model ────────────────────────────────────────────────────
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1 if verbose else 0)
    env.start()

    m = gp.Model("FlightPlanning_v2_final", env=env)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap',    mip_gap)

    # ═════════════════════════════════════════════════════════════════════════
    # DECISION VARIABLES
    # ═════════════════════════════════════════════════════════════════════════

    x = {(r, t, a): m.addVar(vtype=GRB.INTEGER, lb=0, ub=M[(r, t, a)],
                              name=f"x[{r},{t},{a}]")
         for (r, t, a) in combos}

    y = {(r, t): m.addVar(vtype=GRB.BINARY, name=f"y[{r},{t}]")
         for r in routes for t in periods}

    z = {(r, t): m.addVar(vtype=GRB.BINARY, name=f"z[{r},{t}]")
         for r in routes for t in T_open}

    # u_at ∈ {0,1} — aircraft deployment indicator
    u = {(a, t): m.addVar(vtype=GRB.BINARY, name=f"u[{a},{t}]")
         for a in aircraft for t in periods}

    # W_at ∈ Z+ — utilisation tracker
    W = {(a, t): m.addVar(vtype=GRB.INTEGER, lb=0,
                           ub=BigM_at.get((a, t), 0),
                           name=f"W[{a},{t}]")
         for a in aircraft for t in periods}

    m.update()

    # ═════════════════════════════════════════════════════════════════════════
    # OBJECTIVE (1')
    # max Σ π^δ·x  −  Σ SC·z  −  Σ FC·u
    # ═════════════════════════════════════════════════════════════════════════
    m.setObjective(
        gp.quicksum(pi_delta[(r, t, a)] * x[(r, t, a)] for (r, t, a) in combos)
        - gp.quicksum(SC.get(r, 0.0) * z[(r, t)] for r in routes for t in T_open)
        - gp.quicksum(FC.get(a, 0.0) * u[(a, t)] for a in aircraft for t in periods),
        GRB.MAXIMIZE,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # ORIGINAL CONSTRAINTS (2)–(14)
    # ═════════════════════════════════════════════════════════════════════════

    # (2) Fleet-hour capacity — also absorbs maintenance reduction (see C19)
    cap_constrs = {}
    for a in aircraft:
        for t in periods:
            if (a, t) not in H_at:
                continue
            relevant = [(r, t, a) for r in routes if (r, t, a) in combos]
            if relevant:
                cap_constrs[(a, t)] = m.addConstr(
                    gp.quicksum(h[(r, t, a)] * x[(r, t, a)] for (r, t, a) in relevant)
                    <= H_at[(a, t)],
                    name=f"cap[{a},{t}]",
                )

    # (3) Route-activation linking
    for (r, t, a) in combos:
        m.addConstr(x[(r, t, a)] <= M[(r, t, a)] * y[(r, t)],
                    name=f"link[{r},{t},{a}]")

    # (4) Minimum service
    for r in routes:
        for t in periods:
            at = [(r, t, a) for a in aircraft if (r, t, a) in combos]
            if at:
                m.addConstr(
                    gp.quicksum(x[(r, t, a)] for (r, t, a) in at)
                    >= L.get((r, t), 1) * y[(r, t)],
                    name=f"minsvc[{r},{t}]")

    # (6) Category coverage
    for c in categories:
        for t in periods:
            rc = [r for r in R_c[c] if (r, t) in y]
            k  = K.get((c, t), 0)
            if rc and k > 0:
                m.addConstr(gp.quicksum(y[(r, t)] for r in rc) >= k,
                            name=f"catcov[{c},{t}]")

    # (10a-c) Opening detection
    for r in routes:
        for t in T_open:
            t_prev = T[t_idx[t] - 1]
            m.addConstr(z[(r, t)] >= y[(r, t)] - y[(r, t_prev)],
                        name=f"newopen_lb[{r},{t}]")
            m.addConstr(z[(r, t)] <= y[(r, t)],
                        name=f"newopen_ub1[{r},{t}]")
            m.addConstr(z[(r, t)] <= 1 - y[(r, t_prev)],
                        name=f"newopen_ub2[{r},{t}]")

    # (11) Minimum-up-time
    for r in routes:
        for t in T_open:
            i0  = t_idx[t]
            win = [T[i] for i in range(i0, min(i0 + N_min, n_T))]
            if len(win) == N_min:
                m.addConstr(gp.quicksum(y[(r, tau)] for tau in win)
                            >= N_min * z[(r, t)],
                            name=f"minup[{r},{t}]")

    # (12) No late openings
    for r in routes:
        for t in T_late:
            m.addConstr(y[(r, t)] <= y[(r, T[t_idx[t] - 1])],
                        name=f"nolateopen[{r},{t}]")

    # (13) Loss-risk cap (optional)
    if Omega is not None:
        neg = [(r, t, a) for (r, t, a) in combos if pi_delta[(r, t, a)] < 0]
        if neg:
            m.addConstr(
                gp.quicksum(abs(pi_delta[(r, t, a)]) * x[(r, t, a)]
                            for (r, t, a) in neg) <= Omega,
                name="loss_risk_cap")

    # (14) NFOC budget (optional)
    if B_nfoc is not None:
        m.addConstr(
            gp.quicksum(nfoc.get((r, t, a), 0.0) * x[(r, t, a)]
                        for (r, t, a) in combos) <= B_nfoc,
            name="nfoc_budget")

    # ═════════════════════════════════════════════════════════════════════════
    # NEW CONSTRAINTS (16)–(19)
    # ═════════════════════════════════════════════════════════════════════════

    for a in aircraft:
        for t in periods:
            flights = [(r, t, a) for r in routes if (r, t, a) in combos]
            bm      = BigM_at.get((a, t), 0)

            # (16) W_at definition
            m.addConstr(
                W[(a, t)] == gp.quicksum(x[(r, t, a)] for (r, t, a) in flights),
                name=f"W_def[{a},{t}]")

            if bm > 0:
                # (17) Activation upper bound: no flights without deployment
                m.addConstr(W[(a, t)] <= bm * u[(a, t)], name=f"u_ub[{a},{t}]")
                # (18) Activation lower bound: u=0 when no flights assigned
                m.addConstr(u[(a, t)] <= W[(a, t)],      name=f"u_lb[{a},{t}]")
            else:
                m.addConstr(u[(a, t)] == 0, name=f"u_zero[{a},{t}]")
                m.addConstr(W[(a, t)] == 0, name=f"W_zero[{a},{t}]")

    # (19) Maintenance capacity reduction
    #
    #  If aircraft a was deployed in period t−k (u_{a,t−k}=1), a fraction
    #  γ of its monthly hours in period t is consumed by maintenance.
    #  Modifies the fleet-hour capacity constraint (2) additively:
    #
    #    Σ_r h_rta · x_rta  +  Σ_{k=1}^{N_maint} γ · H_at · u_{a,t−k}  ≤  H_at
    #
    #  ⟺  Σ_r h_rta · x_rta  ≤  H_at · (1 − γ · Σ_k u_{a,t−k})
    #
    #  The aircraft type can still operate — just with reduced capacity.
    #  Setting gamma_maint=0.0 deactivates this constraint.
    if gamma_maint > 0.0 and N_maint > 0:
        for a in aircraft:
            for t in periods:
                if (a, t) not in H_at:
                    continue
                i_t      = t_idx[t]
                H_val    = H_at[(a, t)]
                relevant = [(r, t, a) for r in routes if (r, t, a) in combos]
                if not relevant:
                    continue

                # Collect maintenance terms from the N_maint preceding periods
                maint_expr = gp.LinExpr()
                for k in range(1, N_maint + 1):
                    i_prev = i_t - k
                    if i_prev >= 0:
                        t_prev = T[i_prev]
                        maint_expr += gamma_maint * H_val * u[(a, t_prev)]

                if maint_expr.size() > 0:
                    # Rebuild capacity constraint with maintenance term
                    flight_hours = gp.quicksum(
                        h[(r, t, a)] * x[(r, t, a)] for (r, t, a) in relevant)
                    m.addConstr(
                        flight_hours + maint_expr <= H_val,
                        name=f"maint_cap[{a},{t}]")

    # ═════════════════════════════════════════════════════════════════════════
    # SOLVE
    # ═════════════════════════════════════════════════════════════════════════
    start = time.time()
    m.optimize()
    runtime = time.time() - start

    status_code = m.Status
    status_str  = STATUS_MAP.get(status_code, str(status_code))
    n_neg       = sum(1 for (r, t, a) in combos if pi_delta[(r, t, a)] < 0)

    results = {
        'status':          status_str,
        'obj_value':       None,
        'runtime':         runtime,
        'x_vals':          {},
        'y_vals':          {},
        'z_vals':          {},
        'u_vals':          {},
        'W_vals':          {},
        'model':           m,
        'delta':           delta,
        'alpha':           alpha,
        'N_min':           N_min,
        'N_maint':         N_maint,
        'gamma_maint':     gamma_maint,
        'Omega':           Omega,
        'n_vars':          m.NumVars,
        'n_constrs':       m.NumConstrs,
        'n_neg_pi_combos': n_neg,
    }

    if status_code in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        try:
            results['obj_value'] = m.ObjVal
            for (r, t, a) in combos:
                val = x[(r, t, a)].X
                if val > 0.5:
                    results['x_vals'][(r, t, a)] = int(round(val))
            for r in routes:
                for t in periods:
                    if y[(r, t)].X > 0.5:
                        results['y_vals'][(r, t)] = 1
            for r in routes:
                for t in T_open:
                    if z[(r, t)].X > 0.5:
                        results['z_vals'][(r, t)] = 1
            for a in aircraft:
                for t in periods:
                    if u[(a, t)].X > 0.5:
                        results['u_vals'][(a, t)] = 1
                    w_val = int(round(W[(a, t)].X))
                    if w_val > 0:
                        results['W_vals'][(a, t)] = w_val
        except gp.GurobiError:
            pass

    return results
