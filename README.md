# Profit-Optimal Flight Planning (DS502 - Path B - Ezgi D. S024981)

A Mixed-Integer Linear Programming (MILP) model that determines optimal flight frequencies across routes, planning periods, and aircraft types to maximize expected profit, subject to fleet-hour capacity and minimum service requirements.

**Dataset:** [Airline Route Profitability and Cost Analysis – Kaggle](https://www.kaggle.com/datasets/waleedfaheem/airline-route-profitability-and-cost-analysis/data)  
**GitHub:** https://github.com/Jforedd/Profit-Optimal-Flight-Planning.git

## Problem Description

An airline operating out of a single hub (DXB) must allocate limited fleet capacity across a set of routes and monthly planning periods. Using historical profitability and cost records, the model selects flight frequencies per route–period–aircraft combination to maximize total expected profit while respecting aircraft-hour limits and optional minimum service levels across route categories (short-, medium-, and long-haul).

## Repository Structure

```
├── data/
│   ├── raw/              # Original Kaggle dataset
│   └── processed/        # Cleaned & aggregated parameters (params_rta.csv)
├── src/
│   ├── 01_clean_aggregate.py   # Data prep → data/processed/
│   ├── 02_build_model.py       # MILP formulation & baseline solve
│   └── 03_experiments.py       # Scenario runs & result export
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



## Method
Mixed-Integer Linear Programming (MILP) solved with Gurobi/OR-Tools. 
This project is naturally formulated as a Mixed-Integer Linear Program (MILP) because the key decisions include both discrete and logical choices. The number of flights assigned to each route–period–aircraft combination must be an integer because operating 3.7 flights is not meaningful, and minimum-service/coverage requirements introduce binary “served/not served” decisions for routes. These logical constraints are expressed through standard linking formulations and, while fleet-hour capacity constraints remain linear. Therefore, MILP provides the correct modeling class-capturing integrality and service logic-while allowing efficient solution with solvers such as Gurobi under the project’s scope and timeline.

## Repo Structure
- data/raw: raw dataset 
- data/processed: cleaned/aggregated sets/parameters for optimization
- src: python scripts (read/analyze, clean/aggregate, model, experiments)
- outputs: solutions and experiment results

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
- `src/02_build_model_gurobi.py` → baseline MILP solution + summary metrics
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
