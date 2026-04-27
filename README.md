# Profit-Optimal Flight Planning
### DS502 - Path B

## Problem Description

This project develops a profit-optimal flight planning model for an airline operating from a single hub. The model decides how many flights should be operated for each route, planning period, and aircraft type in order to maximize total expected profit while respecting operational and financial constraints.

The project is mainly formulated as a Mixed-Integer Linear Programming (MILP) model and is also interpreted as a Markov Decision Process (MDP). The MILP solves the full multi-period planning problem, while the MDP reformulation explains the same planning logic as a sequential decision-making problem.

The model considers:

- route-period-aircraft flight frequency decisions,
- fleet-hour capacity limits,
- route activation decisions,
- minimum service requirements,
- route-category coverage,
- route-opening detection,
- minimum up-time after route opening,
- fuel-price shock scenarios,
- optional loss-risk and non-fuel operating cost controls,
- aircraft deployment decisions,
- aircraft activation fixed costs,
- aircraft utilization tracking,
- maintenance-related capacity reduction after aircraft deployment.

---

## Repository Structure

```text
├── data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Cleaned and aggregated input files
│       ├── params_rta.csv
│       ├── H_bar_at.csv
│       ├── L_rt.csv
│       └── route_category.csv
│
├── src/
│   ├── read_analyze.py              # Initial data reading and exploration
│   ├── clean_aggregate.py           # Data cleaning and aggregation
│   ├── data_loader.py               # Loads processed model inputs
│   ├── model.py                     # Updated MILP formulation
│   ├── main.py                      # Scenario experiment runner
│   
├── outputs/                         # Solver outputs and experiment summaries
├── mdp_notes.md                     # MDP reformulation notes
└── README.md
```

---

## Mathematical Model Type

The main model is a Mixed-Integer Linear Programming (MILP) model.

It is a MILP because:

- flight frequency decisions are integer variables,
- route activation decisions are binary variables,
- route-opening decisions are binary variables,
- aircraft deployment decisions are binary variables,
- the objective function and constraints are linear.

The model is implemented in Python and solved with Gurobi.

---

## Sets and Indices

| Symbol | Description |
|---|---|
| `r ∈ R` | Set of routes |
| `t ∈ T` | Set of planning periods/months |
| `a ∈ A` | Set of aircraft types |
| `c ∈ C` | Set of route categories |
| `R_c ⊆ R` | Routes belonging to category `c` |
| `T_open` | Periods where a route can be newly opened |
| `T_late` | Late periods where new route openings are not allowed |

---

## Main Parameters

| Symbol | Description |
|---|---|
| `pi_rta` | Base expected profit per flight |
| `f_rta` | Fuel cost per flight |
| `delta` | Fuel-price shock factor |
| `pi_rta_delta` | Scenario-adjusted profit per flight |
| `h_rta` | Flight-hours required per flight |
| `H_bar_at` | Historical available aircraft-hours |
| `alpha` | Capacity multiplier |
| `H_at` | Effective fleet-hour capacity |
| `M_rta` | Maximum allowable flights |
| `L_rt` | Minimum flights if a route is active |
| `K_ct` | Minimum number of active routes in a category |
| `SC_r` | Route startup cost |
| `N_min` | Minimum up-time after route opening |
| `Omega` | Optional loss-risk cap |
| `B_nfoc` | Optional non-fuel operating cost budget |
| `FC_a` | Aircraft activation fixed cost |
| `N_maint` | Maintenance lookback window |
| `gamma_maint` | Maintenance intensity parameter |

---

## Decision Variables

| Symbol | Domain | Description |
|---|---|---|
| `x_rta` | Integer | Number of flights on route `r`, period `t`, aircraft type `a` |
| `y_rt` | Binary | 1 if route `r` is active in period `t`, 0 otherwise |
| `z_rt` | Binary | 1 if route `r` is newly opened in period `t`, 0 otherwise |
| `u_at` | Binary | 1 if aircraft type `a` is deployed in period `t`, 0 otherwise |
| `W_at` | Integer | Total flights assigned to aircraft type `a` in period `t` |

---

## Objective Function

The objective is to maximize total scenario-adjusted profit minus route startup costs and aircraft activation fixed costs:

```text
max Σ_{r,t,a} pi_rta_delta x_rta
    - Σ_{r,t} SC_r z_rt
    - Σ_{a,t} FC_a u_at
```

The first term represents profit from operated flights. The second term penalizes newly opened routes. The third term penalizes aircraft deployment.

---

## Main Constraints

The model includes the following constraint groups:

| Constraint Group | Description |
|---|---|
| Fleet-hour capacity | Total flight-hours cannot exceed available capacity |
| Route activation linking | Flights can only be assigned to active routes |
| Minimum service | Active routes must receive minimum flight service |
| Category coverage | Each route category must satisfy minimum coverage |
| Route-opening detection | Detects whether a route is newly opened |
| Minimum up-time | Newly opened routes must remain active for a required number of periods |
| No late openings | Prevents route openings too close to the end of the horizon |
| Loss-risk cap | Limits total exposure to loss-making flight combinations |
| NFOC budget | Limits total non-fuel operating cost |
| Aircraft utilization | Tracks total flights by aircraft type and period |
| Aircraft activation | Links aircraft deployment to actual usage |
| Maintenance-capacity reduction | Reduces future capacity after aircraft deployment |

---

## Maintenance-Capacity Reduction

In the updated implementation, maintenance is modeled as a capacity-reduction rule rather than a full aircraft grounding rule.

The implemented constraint is:

```text
sum_r h_rta x_rta + sum_{k=1}^{N_maint} gamma_maint H_at u_{a,t-k} <= H_at
```

This means that if aircraft type `a` was deployed in previous periods, a fraction of its current capacity is consumed by maintenance. The aircraft type can still operate, but with reduced available capacity.

If:

```text
gamma_maint = 0
```

then the maintenance effect is deactivated.

---

## MDP Reformulation

For the Week 9 & 10 deliverable, the MILP model is also reformulated as a finite-horizon Markov Decision Process (MDP).

The MDP does not replace the MILP. It explains the same multi-period planning problem as a sequential decision-making process.

### State

A state at period `t` is represented as:

```text
s_t = (t, b_t, m_t, q_t, y_{t-1}, Omega_t_rem, B_t_nfoc_rem)
```

where:

- `t` is the current planning period,
- `b_t` is the vector of effective fleet-hour capacities,
- `m_t` is the vector of remaining minimum-up-time commitments,
- `q_t` is the aircraft maintenance-memory vector,
- `y_{t-1}` is the previous period route-activity vector,
- `Omega_t_rem` is the remaining loss-risk allowance, if active,
- `B_t_nfoc_rem` is the remaining non-fuel operating cost budget, if active.

### Action

At state `s_t`, the action is:

```text
a_t = (x_t, y_t, z_t, u_t)
```

where the action includes current-period flight assignments, route activation decisions, route-opening decisions, and aircraft deployment decisions.

### Transition

The transition is deterministic:

```text
s_{t+1} = F_t(s_t, a_t)
```

The transition updates the planning period, route commitments, aircraft maintenance-memory, previous route activity, and optional remaining financial budgets.

### Reward

The immediate reward is the current-period contribution to the MILP objective:

```text
r_t(s_t, a_t) =
sum_{r,a} pi_rta_delta x_rta
- sum_r SC_r z_rt
- sum_a FC_a u_at
```

### Bellman Equation

The deterministic Bellman recursion is:

```text
V_t(s_t) = max_{a_t in A(s_t)} { r_t(s_t, a_t) + V_{t+1}(F_t(s_t, a_t)) }
```

with terminal condition:

```text
V_{|T|+1}(s) = 0
```

The current MDP is finite-horizon, deterministic, fully observable, and undiscounted.

More details are included in:

```text
mdp_notes.md
```
## Explicit MDP Enumeration Demonstration

In addition to the MILP implementation, the repository includes an explicit MDP enumeration script:

```text
src/mdp_enumeration.py
---

## Current Experiments

The current implementation in `main_v2.py` runs the following scenario groups:

| # | Scenario | Setting |
|---|---|---|
| 1 | Small baseline | 10 routes, 3 months, 3 aircraft |
| 2 | Fuel shock | `delta ∈ {0, 0.10, 0.25}` |
| 3 | Capacity | `alpha ∈ {0.30, 0.60, 1.00}` |
| 4 | Minimum up-time | `N_min ∈ {1, 2, 3}` |
| 5 | Maintenance window | `N_maint ∈ {1, 2, 3}` |
| 6 | Maintenance intensity | `gamma_maint ∈ {0.0, 0.10, 0.20, 0.30}` |
| 7 | Aircraft activation cost | `FC_a ∈ {0, 50000, 100000}` |
| 8 | Medium baseline | 20 routes, 8 months, 4 aircraft |
| 9 | Large baseline | 16 routes, 12 months, 6 aircraft |
| 10 | Large fuel shock | `delta ∈ {0, 0.10, 0.25}` |
| 11 | Large maintenance intensity | `gamma_maint ∈ {0.0, 0.20, 0.40}` |

The large instance uses 16 routes because the restricted Gurobi license has a constraint-size limit. A full 30-route instance would require a larger Gurobi academic license.

---

## Output Metrics

The following metrics are reported across scenarios:

- objective value,
- solver status,
- runtime,
- number of variables,
- number of constraints,
- active route-periods,
- route openings,
- aircraft deployments,
- aircraft utilization,
- number of negative-profit combinations.

---

## Current Results Summary

The current outputs are consistent with the model logic:

- Fuel shock reduces total objective value.
- Capacity reduction reduces total objective value.
- Higher aircraft activation cost reduces total objective value.
- Maintenance window changes may have limited impact in small instances if there is enough capacity slack.
- Higher maintenance intensity reduces objective value, especially in larger instances.
- Medium and large instances are solved within the selected 1% MIP gap tolerance.

Example results from the current runs:

| Scenario | Objective |
|---|---:|
| Small baseline | `$25,604,427.93` |
| Medium baseline | `$127,270,423.45` |
| Large baseline | `$492,242,030.26` |
| Large fuel shock, delta = 0.25 | `$451,810,631.90` |
| Large maintenance intensity, gamma = 0.40 | `$379,629,960.48` |

---

## Solver / Library Requirements

This project requires:

- Python 3.10+
- pandas
- gurobipy
- active Gurobi license

Optional libraries for later analysis or visualization:

- numpy
- matplotlib
- plotly

---

## How to Run the Code

1. Clone the repository:

```bash
git clone https://github.com/ezgi-donmez/Profit-Optimal-Flight-Planning.git
cd Profit-Optimal-Flight-Planning
```

2. Install required libraries:

```bash
pip install pandas numpy gurobipy
```

3. Place the processed input files inside `data/processed/`:

```text
params_rta.csv
H_bar_at.csv
L_rt.csv
route_category.csv
```

4. Run the main experiment script:

```bash
python src/main.py
```

---

## Assumptions

The current model uses the following assumptions:

- Profit per flight and flight-hours are estimated from historical data.
- Scenario-adjusted profit is deterministic once `delta` is selected.
- Aircraft availability is represented using aggregate fleet-hours by aircraft type and period.
- Aircraft maintenance is represented as a capacity reduction, not full grounding.
- Tail-level aircraft rotations are not modeled.
- Crew scheduling is not modeled.
- Airport slot constraints are not modeled.
- Demand uncertainty is not modeled directly.
- Optional financial constraints can be activated or deactivated.

---

## Simplifications

The model is a tactical route-frequency planning model, not a full airline timetable model.

The following details are simplified or excluded:

- individual aircraft tail assignment,
- detailed aircraft rotation feasibility,
- crew pairing,
- airport slot allocation,
- passenger connection structure,
- passenger spill and recapture,
- stochastic demand realization.

---

## Limitations

The model provides decision-support recommendations rather than a directly deployable airline schedule.

Main limitations include:

- public data limitations,
- aggregate capacity representation,
- deterministic scenario treatment,
- simplified maintenance modeling,
- no detailed passenger demand model,
- no full timetable feasibility check.

---

## DS502 Roadmap

### D1 — Topic and Repository Setup

- Created project repository.
- Defined the initial flight planning problem.
- Prepared initial folder structure.

### D2 — Proposal

- Analyzed the route profitability dataset.
- Finalized the scope as route-frequency planning.
- Defined initial assumptions and scenario plan.

### D3 — Mathematical Model

- Defined the MILP formulation.
- Introduced sets, parameters, decision variables, objective, and constraints.

### D4 — Implementation and Baseline Results

- Implemented the baseline MILP.
- Ran small-instance tests.
- Exported solver output and route-level results.

### D6 — MDP Reformulation and Extended Experiments

- Added route-opening and minimum-up-time logic.
- Added aircraft activation variables.
- Added aircraft utilization tracking.
- Added maintenance-related capacity reduction.
- Reformulated the model as a finite-horizon MDP.
- Defined states, actions, transitions, rewards, policy, horizon, and Bellman equation.
- Ran scenario experiments for fuel shock, capacity, minimum up-time, maintenance, activation cost, and instance size.




