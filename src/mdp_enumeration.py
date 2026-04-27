"""
mdp_enumeration.py
------------------
Explicit MDP state-enumeration demonstration for the Profit-Optimal
Flight Planning project.

Important:
This file is NOT intended to replace the MILP implementation.
The main project still solves the deterministic multi-period MILP using Gurobi.

This file explicitly represents:
    - state
    - action
    - transition
    - reward
    - finite-horizon dynamic programming recursion

Because exact MDP enumeration grows very quickly, this script should only be
run on a very small instance.

The flight-count decision x_rta is optimized inside each stage using a small
single-period MILP after enumerating binary route/activity/deployment actions.
This keeps the example computationally possible while still making the MDP
state-transition structure explicit.
"""

import os
import sys
import itertools
from dataclasses import dataclass
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

# Make local imports work when running from src/
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data, print_data_summary


# ---------------------------------------------------------------------------
# MDP state and action definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MDPState:
    """
    Explicit MDP state.

    period_pos:
        Index of the current period in the ordered period list.
        For example, period_pos = 0 means the first planning period.

    y_prev:
        Previous period route activity vector.
        Tuple of 0/1 values with length |R|.

    route_commit:
        Remaining minimum-up-time commitment for each route.
        Tuple of integers with length |R|.

    maint_memory:
        Aircraft maintenance-memory vector.
        Tuple of tuples. For each aircraft type a, it stores whether
        that aircraft was deployed in the previous N_maint periods.
        Example for one aircraft and N_maint=2:
            (1, 0) means deployed in the immediately previous period,
            not deployed two periods ago.
    """
    period_pos: int
    y_prev: tuple
    route_commit: tuple
    maint_memory: tuple


@dataclass(frozen=True)
class MDPAction:
    """
    Explicit stage action.

    y:
        Current route activity vector.

    z:
        Current route-opening vector.

    u:
        Current aircraft deployment vector.

    x:
        Optimized flight assignment dictionary for the current period.
        This is obtained by a single-period MILP after y, z, and u are fixed.

    reward:
        Immediate reward of the stage action.
    """
    y: tuple
    z: tuple
    u: tuple
    x: tuple
    reward: float


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_open_and_late_periods(periods, N_min):
    """
    Reproduce the route-opening period logic used in the MILP.
    """
    T = sorted(periods)
    n_T = len(T)

    if n_T > N_min:
        T_open = [T[i] for i in range(1, n_T - N_min + 1)]
    else:
        T_open = []

    T_late = [T[i] for i in range(max(0, n_T - N_min + 1), n_T)]

    return T_open, T_late


def initial_state(data, N_maint):
    """
    Initial state before the first planning period.
    No previous route activity, no route commitments, and no aircraft
    maintenance-memory.
    """
    n_routes = len(data["routes"])
    n_aircraft = len(data["aircraft"])

    return MDPState(
        period_pos=0,
        y_prev=tuple(0 for _ in range(n_routes)),
        route_commit=tuple(0 for _ in range(n_routes)),
        maint_memory=tuple(tuple(0 for _ in range(N_maint)) for _ in range(n_aircraft)),
    )


def category_coverage_ok(y_tuple, data, current_period):
    """
    Check route-category coverage:
        sum_{r in R_c} y_rt >= K_ct
    """
    routes = data["routes"]
    route_to_idx = {r: i for i, r in enumerate(routes)}

    for c in data["categories"]:
        required = data["K"].get((c, current_period), 0)
        if required <= 0:
            continue

        active_count = 0
        for r in data["R_c"].get(c, []):
            if r in route_to_idx:
                active_count += y_tuple[route_to_idx[r]]

        if active_count < required:
            return False

    return True


def derive_route_openings(state, y_tuple, data, current_period, N_min):
    """
    Derive z_rt from previous route activity and current route activity.

    z_rt = 1 if the route was inactive in the previous period and
    becomes active in the current period.

    For the first period, z is set to zero because the opening detector
    requires a previous period.
    """
    routes = data["routes"]
    periods = sorted(data["periods"])
    T_open, T_late = get_open_and_late_periods(periods, N_min)

    z = []

    for i, r in enumerate(routes):
        y_now = y_tuple[i]
        y_prev = state.y_prev[i]

        # First period: no opening detector
        if state.period_pos == 0:
            z.append(0)
            continue

        # Late periods: no new openings allowed
        if current_period in T_late and y_now > y_prev:
            return None

        # Opening periods
        if current_period in T_open:
            z.append(1 if (y_now == 1 and y_prev == 0) else 0)
        else:
            z.append(0)

    return tuple(z)


def route_commitments_ok(state, y_tuple):
    """
    If a route still has remaining minimum-up-time commitment,
    it must stay active.
    """
    for i, commit in enumerate(state.route_commit):
        if commit > 0 and y_tuple[i] == 0:
            return False
    return True


def update_route_commitments(state, z_tuple, N_min):
    """
    Update route commitment counters.

    If route r is newly opened at t:
        m_r^{t+1} = N_min - 1

    If it already has commitment:
        m_r^{t+1} = max(m_r^t - 1, 0)
    """
    next_commit = []

    for i, old_commit in enumerate(state.route_commit):
        if z_tuple[i] == 1:
            next_commit.append(max(N_min - 1, 0))
        elif old_commit > 0:
            next_commit.append(max(old_commit - 1, 0))
        else:
            next_commit.append(0)

    return tuple(next_commit)


def update_maintenance_memory(state, u_tuple, N_maint):
    """
    Update aircraft maintenance-memory.

    q_{a,1}^{t+1} = u_at
    q_{a,k}^{t+1} = q_{a,k-1}^{t}, for k = 2,...,N_maint
    """
    if N_maint == 0:
        return tuple(tuple() for _ in u_tuple)

    next_memory = []

    for a_idx, u_val in enumerate(u_tuple):
        old_mem = state.maint_memory[a_idx]
        new_mem = (u_val,) + old_mem[:-1]
        next_memory.append(new_mem)

    return tuple(next_memory)


def transition(state, y_tuple, z_tuple, u_tuple, data, N_min, N_maint):
    """
    Deterministic MDP transition.
    """
    next_period_pos = state.period_pos + 1

    next_state = MDPState(
        period_pos=next_period_pos,
        y_prev=tuple(y_tuple),
        route_commit=update_route_commitments(state, z_tuple, N_min),
        maint_memory=update_maintenance_memory(state, u_tuple, N_maint),
    )

    return next_state


# ---------------------------------------------------------------------------
# Single-period action evaluation
# ---------------------------------------------------------------------------

def solve_stage_flight_assignment(
    state,
    y_tuple,
    z_tuple,
    u_tuple,
    data,
    delta=0.0,
    alpha=1.0,
    N_maint=2,
    gamma_maint=0.20,
    FC=None,
    SC=None,
    verbose=False,
):
    """
    Given a binary MDP action (y,z,u), solve the current-period flight
    assignment x_rta.

    This is a small one-period MILP. It evaluates the immediate reward
    for the action.

    Returns:
        reward, x_solution

    If infeasible:
        returns None, None
    """
    routes = data["routes"]
    aircraft = data["aircraft"]
    periods = sorted(data["periods"])
    combos = data["combos"]

    pi = data["pi"]
    h = data["h"]
    f = data["f"]
    M = data["M"]
    L = data["L"]
    H_bar = data["H_bar"]

    current_period = periods[state.period_pos]

    if FC is None:
        FC = {a: 0.0 for a in aircraft}
    if SC is None:
        SC = {r: 0.0 for r in routes}

    route_to_idx = {r: i for i, r in enumerate(routes)}
    aircraft_to_idx = {a: i for i, a in enumerate(aircraft)}

    # Current-period route-aircraft combinations
    current_combos = [
        (r, current_period, a)
        for (r, t, a) in combos
        if t == current_period
    ]

    # Build one-period MILP
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()

    m = gp.Model("stage_action_eval", env=env)
    m.setParam("OutputFlag", 1 if verbose else 0)

    x = {}

    for (r, t, a) in current_combos:
        r_idx = route_to_idx[r]
        a_idx = aircraft_to_idx[a]

        # If route is inactive or aircraft is not deployed, force no variable
        if y_tuple[r_idx] == 0 or u_tuple[a_idx] == 0:
            continue

        x[(r, t, a)] = m.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            ub=M[(r, t, a)],
            name=f"x[{r},{t},{a}]"
        )

    m.update()

    # Minimum service constraints
    for r in routes:
        r_idx = route_to_idx[r]
        if y_tuple[r_idx] == 0:
            continue

        relevant = [
            (rr, tt, aa)
            for (rr, tt, aa) in x
            if rr == r and tt == current_period
        ]

        if not relevant:
            return None, None

        m.addConstr(
            gp.quicksum(x[key] for key in relevant) >= L.get((r, current_period), 1),
            name=f"min_service[{r}]"
        )

    # Capacity + maintenance-memory constraints
    for a in aircraft:
        a_idx = aircraft_to_idx[a]

        if (a, current_period) not in H_bar:
            continue

        H_at = alpha * H_bar[(a, current_period)]

        relevant = [
            (rr, tt, aa)
            for (rr, tt, aa) in x
            if aa == a and tt == current_period
        ]

        flight_hours = gp.quicksum(h[key] * x[key] for key in relevant)

        # Maintenance memory from previous deployments
        maint_used = 0.0
        if N_maint > 0 and gamma_maint > 0:
            for memory_value in state.maint_memory[a_idx]:
                maint_used += gamma_maint * H_at * memory_value

        m.addConstr(
            flight_hours + maint_used <= H_at,
            name=f"capacity_with_maintenance[{a}]"
        )

    # Objective: immediate reward
    profit_expr = gp.quicksum(
        (pi[key] - delta * f[key]) * x[key]
        for key in x
    )

    startup_cost = sum(
        SC.get(r, 0.0) * z_tuple[route_to_idx[r]]
        for r in routes
    )

    activation_cost = sum(
        FC.get(a, 0.0) * u_tuple[aircraft_to_idx[a]]
        for a in aircraft
    )

    m.setObjective(profit_expr - startup_cost - activation_cost, GRB.MAXIMIZE)

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None, None

    x_solution = {}
    for key, var in x.items():
        val = int(round(var.X))
        if val > 0:
            x_solution[key] = val

    reward = float(m.ObjVal)

    # Convert x_solution to a hashable tuple for storage in MDPAction
    x_tuple = tuple(sorted(x_solution.items()))

    return reward, x_tuple


# ---------------------------------------------------------------------------
# Action enumeration
# ---------------------------------------------------------------------------

def enumerate_feasible_actions(
    state,
    data,
    delta=0.0,
    alpha=1.0,
    N_min=2,
    N_maint=2,
    gamma_maint=0.20,
    FC=None,
    SC=None,
    max_actions_per_state=None,
):
    """
    Enumerate feasible MDP actions for a given state.

    This explicitly enumerates binary route activity y_t and aircraft
    deployment u_t. Then z_t is derived from y_{t-1} and y_t.
    For each binary action, the current-period flight assignment x_t is
    solved by a small single-period MILP.
    """
    routes = data["routes"]
    aircraft = data["aircraft"]
    periods = sorted(data["periods"])

    if state.period_pos >= len(periods):
        return []

    current_period = periods[state.period_pos]

    actions = []

    # Enumerate route activity y_t
    for y_tuple in itertools.product([0, 1], repeat=len(routes)):

        # Route commitment feasibility
        if not route_commitments_ok(state, y_tuple):
            continue

        # Category coverage feasibility
        if not category_coverage_ok(y_tuple, data, current_period):
            continue

        # Derive route openings z_t
        z_tuple = derive_route_openings(state, y_tuple, data, current_period, N_min)
        if z_tuple is None:
            continue

        # Enumerate aircraft deployment u_t
        for u_tuple in itertools.product([0, 1], repeat=len(aircraft)):

            # At least one aircraft should be deployed if any route is active
            if sum(y_tuple) > 0 and sum(u_tuple) == 0:
                continue

            reward, x_tuple = solve_stage_flight_assignment(
                state=state,
                y_tuple=y_tuple,
                z_tuple=z_tuple,
                u_tuple=u_tuple,
                data=data,
                delta=delta,
                alpha=alpha,
                N_maint=N_maint,
                gamma_maint=gamma_maint,
                FC=FC,
                SC=SC,
                verbose=False,
            )

            if reward is None:
                continue

            action = MDPAction(
                y=tuple(y_tuple),
                z=tuple(z_tuple),
                u=tuple(u_tuple),
                x=x_tuple,
                reward=reward,
            )

            actions.append(action)

            if max_actions_per_state is not None and len(actions) >= max_actions_per_state:
                return actions

    return actions


# ---------------------------------------------------------------------------
# Explicit reachable-state enumeration and backward DP
# ---------------------------------------------------------------------------

def enumerate_reachable_mdp(
    data,
    delta=0.0,
    alpha=1.0,
    N_min=2,
    N_maint=2,
    gamma_maint=0.20,
    FC=None,
    SC=None,
    max_actions_per_state=None,
):
    """
    Explicitly enumerate reachable states and transitions over the finite horizon.
    """
    periods = sorted(data["periods"])
    horizon = len(periods)

    s0 = initial_state(data, N_maint)

    states_by_time = defaultdict(set)
    states_by_time[0].add(s0)

    edges = defaultdict(list)

    for t_pos in range(horizon):
        print(f"\nEnumerating period position {t_pos + 1}/{horizon} ...")
        print(f"Current number of states: {len(states_by_time[t_pos])}")

        for state in list(states_by_time[t_pos]):
            actions = enumerate_feasible_actions(
                state=state,
                data=data,
                delta=delta,
                alpha=alpha,
                N_min=N_min,
                N_maint=N_maint,
                gamma_maint=gamma_maint,
                FC=FC,
                SC=SC,
                max_actions_per_state=max_actions_per_state,
            )

            for action in actions:
                next_state = transition(
                    state=state,
                    y_tuple=action.y,
                    z_tuple=action.z,
                    u_tuple=action.u,
                    data=data,
                    N_min=N_min,
                    N_maint=N_maint,
                )

                edges[state].append((action, next_state))
                states_by_time[t_pos + 1].add(next_state)

        print(f"Next number of states: {len(states_by_time[t_pos + 1])}")

    return states_by_time, edges


def solve_by_backward_dp(states_by_time, edges, horizon):
    """
    Solve the explicitly enumerated finite-horizon deterministic MDP
    using backward dynamic programming.
    """
    V = {}
    policy = {}

    # Terminal values
    for s in states_by_time[horizon]:
        V[s] = 0.0

    # Backward recursion
    for t_pos in reversed(range(horizon)):
        for state in states_by_time[t_pos]:
            best_value = -float("inf")
            best_action = None
            best_next_state = None

            for action, next_state in edges.get(state, []):
                value = action.reward + V.get(next_state, -float("inf"))

                if value > best_value:
                    best_value = value
                    best_action = action
                    best_next_state = next_state

            V[state] = best_value
            policy[state] = (best_action, best_next_state)

    return V, policy


def print_policy_path(initial, policy, periods, routes, aircraft):
    """
    Print one optimal policy path from the initial state.
    """
    state = initial

    print("\n" + "=" * 80)
    print("OPTIMAL MDP POLICY PATH FOR ENUMERATED SMALL INSTANCE")
    print("=" * 80)

    for step in range(len(periods)):
        if state not in policy or policy[state][0] is None:
            print("No feasible action from state.")
            break

        action, next_state = policy[state]
        period = periods[state.period_pos]

        active_routes = [routes[i] for i, val in enumerate(action.y) if val == 1]
        opened_routes = [routes[i] for i, val in enumerate(action.z) if val == 1]
        deployed_aircraft = [aircraft[i] for i, val in enumerate(action.u) if val == 1]

        print(f"\nPeriod {period}")
        print(f"  Reward: {action.reward:,.2f}")
        print(f"  Active routes: {active_routes}")
        print(f"  Opened routes: {opened_routes}")
        print(f"  Deployed aircraft: {deployed_aircraft}")
        print("  Flight assignments:")

        for key, value in action.x:
            print(f"    {key}: {value}")

        state = next_state

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main test run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Keep this very small.
    # Full 30-route enumeration is not practical.
    data = load_data(
    n_routes=6,
    n_months=3,
    n_aircraft=3,
    K_ct_value=0,
)

    N_min = 2
    N_maint = 2
    gamma_maint = 0.20
    delta = 0.0
    alpha = 1.0

    # Optional fixed costs. Set to zero unless your data dictionary contains them.
    FC = {a: 0.0 for a in data["aircraft"]}
    SC = {r: 0.0 for r in data["routes"]}

    states_by_time, edges = enumerate_reachable_mdp(
        data=data,
        delta=delta,
        alpha=alpha,
        N_min=N_min,
        N_maint=N_maint,
        gamma_maint=gamma_maint,
        FC=FC,
        SC=SC,
        max_actions_per_state=None,
    )

    horizon = len(data["periods"])
    V, policy = solve_by_backward_dp(states_by_time, edges, horizon)

    s0 = initial_state(data, N_maint)

    print("\n" + "=" * 80)
    print("MDP ENUMERATION SUMMARY")
    print("=" * 80)

    for t_pos in range(horizon + 1):
        print(f"States at time {t_pos}: {len(states_by_time[t_pos])}")

    total_edges = sum(len(v) for v in edges.values())
    print(f"Total enumerated transitions: {total_edges}")

    print(f"\nInitial state value: {V.get(s0, None):,.2f}")

    print_policy_path(
        initial=s0,
        policy=policy,
        periods=sorted(data["periods"]),
        routes=data["routes"],
        aircraft=data["aircraft"],
    )
