# MDP Notes - Profit-Optimal Flight Planning

This file summarizes the MDP reformulation of the Profit-Optimal Flight Planning project. The original model is a Mixed-Integer Linear Programming (MILP) model. The MDP version does not replace the MILP; it explains the same multi-period planning problem as a sequential decision-making process.

---

## 1. Original MILP View

The original model decides how many flights should be operated for each route, period, and aircraft type. The main objective is to maximize total scenario-adjusted expected profit while satisfying operational and financial constraints.

Main MILP decision variables:

- `x_rta`: number of flights on route `r`, period `t`, aircraft type `a`
- `y_rt`: 1 if route `r` is active in period `t`, 0 otherwise
- `z_rt`: 1 if route `r` is newly opened in period `t`, 0 otherwise
- `u_at`: 1 if aircraft type `a` is deployed in period `t`, 0 otherwise
- `W_at`: total utilization of aircraft type `a` in period `t`

The objective is:

```text
max adjusted flight profit - route startup costs - aircraft activation costs
```

The model includes fleet-hour capacity, route activation, minimum service, category coverage, route-opening logic, minimum up-time, aircraft deployment, aircraft utilization, optional financial controls, and maintenance-related capacity reduction.

---

## 2. MDP Interpretation

In the MDP view, each planning period is one decision epoch. At the beginning of each period, the airline observes the current system state, chooses a feasible operating plan, receives immediate reward, and then moves to the next period.

This interpretation is useful because decisions in one period affect future periods. For example, opening a route creates a minimum-up-time commitment, and deploying an aircraft type may reduce future available capacity through maintenance requirements.

---

## 3. State Space

A state at period `t` is represented as:

```text
s_t = (t, b_t, m_t, q_t, y_{t-1}, Omega_t_rem, B_t_nfoc_rem)
```

where:

- `t` is the current planning period.
- `b_t` is the vector of effective fleet-hour capacities.
- `m_t` is the vector of remaining minimum-up-time commitments for routes.
- `q_t` is the aircraft maintenance-memory vector, recording whether each aircraft type was deployed in the previous `N_maint` periods.
- `y_{t-1}` is the previous period route-activity vector.
- `Omega_t_rem` is the remaining loss-risk allowance, if the optional loss-risk cap is active.
- `B_t_nfoc_rem` is the remaining non-fuel operating cost budget, if the optional NFOC budget is active.

The last two components are optional and are only needed when the financial control constraints are active.

---

## 4. Action Space

At state `s_t`, the action is the current-period operating plan:

```text
a_t = (x_t, y_t, z_t, u_t)
```

where:

- `x_t` gives the flight assignment decisions.
- `y_t` gives the route activation decisions.
- `z_t` gives the route-opening decisions.
- `u_t` gives the aircraft deployment decisions.

A feasible action must satisfy the MILP constraints and the state-dependent rules. In particular:

- Routes with remaining minimum-up-time commitment must stay active.
- New route openings must be consistent with the previous route activity.
- New openings are only allowed in feasible opening periods.
- Aircraft deployment must satisfy available fleet-hour capacity.
- Recent aircraft deployment reduces available capacity through maintenance.
- Optional loss-risk and NFOC budgets cannot be exceeded.

The implemented maintenance-capacity rule is:

```text
sum_r h_rta x_rta + sum_{k=1}^{N_maint} gamma_maint H_at u_{a,t-k} <= H_at
```

This means aircraft are not fully grounded after deployment. Instead, previous deployment consumes part of future available capacity.

---

## 5. Transition Function

The transition function is deterministic:

```text
s_{t+1} = F_t(s_t, a_t)
```

After taking action `a_t`:

- the period advances from `t` to `t+1`,
- `y_t` becomes the previous route-activity vector for the next state,
- route commitment counters are updated,
- aircraft maintenance-memory is updated,
- remaining optional financial budgets are updated if active.

If a route is newly opened, it receives a minimum-up-time commitment. If an aircraft type is deployed, that deployment is recorded in the maintenance-memory state for future capacity calculations.

---

## 6. Reward Function

The immediate reward is the current-period contribution to the original MILP objective:

```text
r_t(s_t, a_t) =
sum_{r,a} pi_rta_delta x_rta
- sum_r SC_r z_rt
- sum_a FC_a u_at
```

The first term is the adjusted profit from flights. The second term subtracts route startup costs. The third term subtracts aircraft activation costs.

Optional loss-risk and NFOC budgets are treated as hard feasibility constraints rather than reward penalties.

---

## 7. Policy

A policy is a rule that selects a feasible action for each state:

```text
mu_t(s_t) = a_t in A(s_t)
```

In this project, a policy tells the airline what operating plan to choose in each period based on current capacity, route commitments, aircraft maintenance-memory, previous route activity, and optional remaining budgets.

---

## 8. Horizon and Terminal Condition

The MDP is finite-horizon because the planning model has a finite set of periods.

The terminal condition is:

```text
V_{|T|+1}(s) = 0
```

This means there is no future reward after the final planning period.

---

## 9. Bellman Equation

The deterministic Bellman recursion is:

```text
V_t(s_t) = max_{a_t in A(s_t)} { r_t(s_t, a_t) + V_{t+1}(F_t(s_t, a_t)) }
```

where:

- `V_t(s_t)` is the maximum total remaining profit from period `t` to the end.
- `A(s_t)` is the feasible action set.
- `r_t(s_t, a_t)` is the immediate reward.
- `F_t(s_t, a_t)` is the deterministic transition function.

The Bellman equation shows that the best current decision is the one that balances immediate profit with future value.

---

## 10. Type of MDP

The current MDP is:

- finite-horizon,
- deterministic,
- fully observable,
- undiscounted.

It is deterministic because the next state is determined by the current state and action once scenario parameters are fixed. It is fully observable because all relevant information is assumed known at the decision epoch. It is undiscounted because the original MILP objective sums profit and costs over the planning horizon without discounting.

---

## 11. MILP-to-MDP Mapping

| MILP Element | MDP Interpretation |
|---|---|
| `x_rta` | Part of the stage action |
| `y_rt` | Part of the stage action |
| `z_rt` | Part of the stage action |
| `u_at` | Part of the stage action |
| `W_at` | Derived aircraft-utilization quantity |
| Previous route status | State component |
| Minimum-up-time logic | Route commitment state and transition |
| Maintenance-capacity logic | Aircraft maintenance-memory state and transition |
| Objective function | Stage reward plus future value |
| MILP constraints | Action feasibility and transition rules |

---

## 12. Current Experiments

The current implementation tests the following scenarios:

- Small baseline: 10 routes, 3 months, 3 aircraft
- Fuel shock: `delta in {0, 0.10, 0.25}`
- Capacity: `alpha in {0.30, 0.60, 1.00}`
- Minimum up-time: `N_min in {1, 2, 3}`
- Maintenance window: `N_maint in {1, 2, 3}`
- Maintenance intensity: `gamma_maint in {0.0, 0.10, 0.20, 0.30}`
- Aircraft activation cost: `FC_a in {0, 50000, 100000}`
- Medium baseline: 20 routes, 8 months, 4 aircraft
- Large baseline: 16 routes, 12 months, 6 aircraft
- Large fuel shock: `delta in {0, 0.10, 0.25}`
- Large maintenance intensity: `gamma_maint in {0.0, 0.20, 0.40}`

---

## 13. Performance Measures

The main output metrics are:

- objective value,
- total flights,
- number of active route-periods,
- number of route openings,
- aircraft deployment count,
- aircraft utilization,
- number of negative-profit combinations,
- solver runtime,
- optimality gap if applicable.

---

## 14. Notes on Results

The current results are consistent with the model logic. Higher fuel shock reduces objective value. Lower capacity reduces objective value. Higher activation cost reduces objective value. Higher maintenance intensity also reduces objective value, especially in the larger instance.

For medium and large instances, the solution should be described as optimal within the selected MIP gap tolerance because the implementation uses a 1% MIP gap.

---

## 15. Summary

The MDP reformulation explains the flight planning problem as a sequential decision process. It clarifies what the airline observes at each period, what decisions it can make, how the system evolves, and how immediate reward connects to future value.

The MDP view is useful because route openings and aircraft deployments have future consequences. Therefore, a good decision should not only maximize current-period profit but also consider future feasibility and capacity.
