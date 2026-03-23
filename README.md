# Profit-Optimal Flight Planning 
### (DS502 - Path B - Ezgi D. S024981)

## Problem Description
A Mixed-Integer Linear Programming (MILP) model that determines optimal flight frequencies across routes, planning periods, and aircraft types to maximize expected profit, subject to fleet-hour capacity and minimum service requirements. An airline operating out of a single hub (DXB) must allocate limited fleet capacity across a set of routes and monthly planning periods. Using historical profitability and cost records, the model selects flight frequencies per route–period–aircraft combination to maximize total expected profit while respecting aircraft-hour limits and optional minimum service levels across route categories (short-, medium-, and long-haul).

## Repository Structure

```
├── data/
│   ├── raw/              # Original Kaggle dataset
│   └── processed/
│       ├── params_rta.csv
│       ├── H_bar_at.csv
│       ├── L_rt.csv
│       └── route_category.csv
├── src/
│   ├── 00_read_analyze.py 
│   ├── 01_clean_aggregate.py   # Data prep → data/processed/
│   ├── 02_data_loader.py       # Load the data
│   └── 03_model.py             # MILP formulation implemented in gurobi
│   └── 04_main.py              # to run the model
├── outputs/              # Solver results, experiment summaries, plots
└── README.md
```

---

## Mathematical Model (MILP)

### Sets & Indices

| Symbol | Description |
|--------|-------------|
| r ∈ R | Routes (Origin–Destination pairs) |
| t ∈ T | Planning periods (months) |
| a ∈ A | Aircraft types |
| c ∈ C | Route categories (Short / Medium / Long Haul) |
| R_c ⊆ R | Subset of routes belonging to category c |

### Parameters

| Symbol | Description |
|--------|-------------|
| π_rta | Expected profit per flight on route r, period t, aircraft a |
| h_rta | Expected aircraft-hours per flight for (r, t, a) |
| H_at | Available aircraft-hours for type a in period t (= α_at · H̄_at) |
| H̄_at | Historical total aircraft-hours for type a in period t |
| M_rta | Maximum allowed flights for (r, t, a) — data-driven upper bound |
| L_rt | Minimum flights required on route r in period t if served |
| K_ct | Minimum number of routes to serve in category c during period t |
| α_at | Capacity multiplier (experiment parameter; baseline = 1.0) |
| f_rta | Average fuel cost per flight for (r, t, a) |
| δ | Fuel price shock factor; adjusted profit: π_rta^(δ) = π_rta − δ · f_rta |

### Decision Variables

| Symbol | Domain | Description |
|--------|--------|-------------|
| x_rta | ℤ₊ | Number of flights on route r, period t, aircraft type a |
| y_rt | {0, 1} | 1 if route r is served in period t, else 0 |

### Objective

Maximize total expected profit:

```
max  Σ_{r,t,a}  π_rta^(δ) · x_rta
```
### Constraints

| # | Constraint | Quantifier | Description |
|---|-----------|------------|-------------|
| (2) | Σ_r h_rta · x_rta ≤ H_at | ∀a, t | Fleet-hour capacity per aircraft type and period |
| (3) | x_rta ≤ M_rta · y_rt | ∀r, t, a | Link flights to route activation; no flights if route not served |
| (4) | Σ_a x_rta ≥ L_rt · y_rt | ∀r, t | Minimum flights when a route is served |
| (5) | 0 ≤ x_rta ≤ M_rta | ∀r, t, a | Non-negativity and data-driven upper bounds |
| (6) | Σ_{r∈R_c} y_rt ≥ K_ct | ∀c, t | Category coverage: at least K_ct routes served per category |
| (7) | x_rta ∈ ℤ₊ |  | Integer flight counts |
| (8) | y_rt ∈ {0, 1} |  | Binary service indicators |

---
## Solver

**Primary:** [Gurobi](https://www.gurobi.com/) (via `gurobipy`)  

## Solver / Library Requirements

This project requires the following environment and libraries:

- **Python 3.10+**
- **pandas**
- **gurobipy**
- an active **Gurobi license** (academic)

The optimization model is solved using **Gurobi Optimizer**. In the current small-instance run, the solver was executed with a **120-second time limit** and a **1% MIP gap tolerance**.

## How to run the code
1. Place the processed input files inside `data/processed/`:
   - `params_rta.csv`
   - `H_bar_at.csv`
   - `L_rt.csv`
   - `route_category.csv`

2. Install the required libraries:
pandas gurobipy

3. Run the main script from the project root directory:
src/main.py

---
## Small Instance Output Explanation
The small test instance was solved successfully to optimality using a reduced real-data subset with 10 routes, 3 months, and 3 aircraft types, corresponding to 36 feasible (r,t,a) combinations and a balanced category mix of 3 Long Haul, 3 Medium Haul, and 4 Short Haul routes. The resulting MILP contained 66 variables and 84 constraints, and Gurobi found the OPTIMAL solution in 0.001 seconds with an objective value of $25,573,426.66, which represents the maximum expected profit for this reduced instance. The solution activated 30 route-month pairs in total, which is consistent with serving all 10 selected routes across 3 months, and the output also shows the corresponding flight allocations by aircraft type for each active route-month pair.  `small_instance_output.txt` contains the full raw terminal output of this first small-instance run, including the data summary, Gurobi solver log, and detailed route-level solution results.

## Planned Experiments (≥ 10 scenarios)

All experiments vary one or more parameters relative to the baseline (α = 1.0, δ = 0).

| # | Scenario | Parameter Change |
|---|----------|-----------------|
| 1 | Baseline | α = 1.0, δ = 0, no min-service |
| 2 | Capacity –20% | α = 0.80 |
| 3 | Capacity –40% | α = 0.60 |
| 4 | Capacity +20% | α = 1.20 |
| 5 | Fuel shock +10% | δ = 0.10 |
| 6 | Fuel shock +25% | δ = 0.25 |
| 7 | Min service ON (loose) | L_rt = 1 per served route |
| 8 | Min service ON (tight) | L_rt = 3 per served route |
| 9 | Category coverage ON | K_ct = 2 per category |
| 10 | Capacity –20% + Fuel shock +10% | α = 0.80, δ = 0.10 |
| 11 | Capacity –40% + Fuel shock +25% | α = 0.60, δ = 0.25 |
| 12 | Seasonal aggregation | Periods = seasons instead of months |

Key output metrics tracked across scenarios: total expected profit, number of routes served, fleet utilization rate, routes dropped per category.

---

## Method
Mixed-Integer Linear Programming (MILP) solved with Gurobi/OR-Tools. 
This project is naturally formulated as a Mixed-Integer Linear Program (MILP) because the key decisions include both discrete and logical choices. The number of flights assigned to each route–period–aircraft combination must be an integer because operating 3.7 flights is not meaningful, and minimum-service/coverage requirements introduce binary “served/not served” decisions for routes. These logical constraints are expressed through standard linking formulations and, while fleet-hour capacity constraints remain linear. Therefore, MILP provides the correct modeling class-capturing integrality and service logic-while allowing efficient solution with solvers such as Gurobi under the project’s scope and timeline.

**Assumptions**
- Profit per flight and flight-hours are estimated from historical data and treated as deterministic.
- Aircraft availability is captured through aggregate fleet-hours per type and period; tail-level rotations are not modeled.
- Routes are independent except for shared fleet-hour capacity (no spill/recapture or competition effects).
- Uncertainty is handled via scenario parameters rather than stochastic optimization.

**Simplifications**
- No crew pairing, maintenance routing, airport slot constraints, or aircraft rotation feasibility.
- Planning is at the route-frequency level (how many flights), not a full timetable.

**Limitations**
- Public dataset only — internal airline constraints (maintenance calendars, slot portfolios) are excluded.
- Results are decision-support recommendations; operational feasibility should be verified separately.

---
## DS502 Roadmap 

### D1 — Topic & Repo Setup (Week 3)
- Repo structure: `data/raw`, `data/processed`, `src`, `outputs`
- README: problem + dataset link + plan + roles

### D2 — Proposal (Week 4)
- Analyze the data
- Finalize scope: monthly flight frequency planning for DXB routes
- Assumptions + data plan: parameter estimation for (route, month, aircraft)
- Baseline MILP: objective + core constraints + scenario list (10+)

### D3 — Mathematical Model v1 (Week 5)
- Write full notation: sets/parameters/variables/objective/constraints
- Define how π_rta, h_rta, c_rta are computed from aggregated data

### D4 — Implementation v1 + Baseline Results (Week 6)
- `src/01_clean_aggregate.py` → `data/processed/params_rta.csv`
- `src/02_data_loader.py` & `src/03_model.py` & `src/04_main.py` → baseline MILP solution + summary metrics
- Baseline outputs: total profit, utilization, route coverage

### D5 — Experiment Plan + Other Components (Week 7)
- Lock experiment matrix (≥10 runs): capacity scaling, fuel shock, min service tightening, budget cap
- Build simplified flow/assignment baseline for comparison (aircraft-hours allocation)

### Week 8 — First Progress Check
- Working pipeline + baseline + at least 3 scenario results

### D6 — Implementation v2 + Extended Results (Week 9)
- Add 1–2 realism constraints (budget and/or route-category service and/or loss-risk limit)
- Run full experiment set and export the results

### D7 — ML/Advanced (Weeks 10–11)
- predict profit-per-flight using route/season/aircraft features and re-optimize

### D8 — Final Report (Week 12)
- Final report: data prep, MILP, flow baseline, experiments, results, limitations
- Slides: model summary + key scenario insights

### D9 — Presentation (Weeks 13–14)
- Final presentation with results, plots, and recommendations
