## License

This project is licensed under the MIT License for the source code.

The datasets used in this project are provided for academic/course purposes and are not redistributed under this license. If you use this repository, please make sure that you have permission to access and use the original data files.

# Profit-Optimal Flight Planning Using Route Profitability Data

This repository contains a strategic airline route-frequency and fleet planning optimization project. The problem is formulated as a **Mixed-Integer Linear Programming (MILP)** model and solved using **Gurobi**. A **Genetic Algorithm (GA)** is also implemented as a heuristic benchmark.

The project was developed for **DS 502 - Introduction to OR Techniques in Data Science** as part of the **MSc Data Science** program at Özyeğin University.

---

## Project Overview

Airlines need to decide which routes to operate, how frequently to serve them, and which aircraft types to assign. These decisions affect profitability, aircraft utilization, service coverage, and operational feasibility.

This project considers an airline operating from **Dubai International Airport (DXB)** to international destinations over a **12-month planning horizon**.

The model decides:

- Number of flights for each route, period, and aircraft type
- Whether each route is active in each period
- Whether a route is newly opened
- Number of physical aircraft deployed by aircraft type and period

The objective is to **maximize total expected profit** while satisfying operational, demand, capacity, maintenance, compatibility, and service constraints.

---

## Dataset

The project uses the Kaggle dataset:

**Airline Route Profitability and Cost Analysis**  
Published by **waleedfaheem**

Dataset link:  
https://www.kaggle.com/datasets/waleedfaheem/airline-route-profitability-and-cost-analysis

### Instance Summary

| Component | Value |
|---|---:|
| Hub Airport | DXB |
| Routes | 30 |
| Planning Periods | 12 months |
| Aircraft Types | 6 |
| Route Categories | Short, Medium, Long Haul |
| Valid Route-Period-Aircraft Combinations | 840 |
| Gurobi Decision Variables | 1632 |
| Gurobi Constraints | 3528 |

---

## Mathematical Model

The problem is modeled as a **Mixed-Integer Linear Programming (MILP)** problem.

### Decision Variables

| Variable | Description |
|---|---|
| `x[r,t,a]` | Number of flights on route `r`, period `t`, aircraft type `a` |
| `y[r,t]` | 1 if route `r` is active in period `t`, 0 otherwise |
| `z[r,t]` | 1 if route `r` is newly opened in period `t`, 0 otherwise |
| `q[a,t]` | Number of physical aircraft of type `a` deployed in period `t` |

### Objective

The objective maximizes total expected profit by considering:

- Flight-level profit
- Fuel-price shock effects
- Route startup costs
- Recurring route fixed costs
- Aircraft deployment fixed costs

### Main Constraints

The model includes:

- Route activation and flight-frequency linking
- Minimum and maximum service requirements
- Aircraft-hour capacity
- Fleet availability
- Rolling-window maintenance approximation
- Demand-based seat limits
- Hub slot capacity
- Route-aircraft compatibility
- Route category coverage
- Route opening logic
- Minimum up-time for newly opened routes

---

## Solution Approaches

### 1. Gurobi MILP Solver

The MILP model is implemented in Python using `gurobipy`.

Gurobi is used as the exact benchmark solver. It provides proven optimal solutions when the model is solved to optimality.

Solver settings:

| Setting | Value |
|---|---:|
| Time Limit | 3600 seconds |
| MIP Gap | 0.00 |
| Output | Enabled |

### 2. Genetic Algorithm

A Genetic Algorithm is implemented as a heuristic alternative.

The GA includes:

- Model-aware chromosome representation
- Greedy and random initial population
- Tournament selection
- Route-based crossover
- Route-block mutation
- Penalty-based fitness function
- Multi-stage feasibility repair
- Greedy local search
- Elitism

The GA does not guarantee global optimality, but it can produce feasible and high-quality solutions.

---

## Repository Structure

```text
Profit-Optimal-Flight-Planning/
│
├── src/
│   ├── clean_aggregate.py      # Cleans and aggregates raw route-profitability data
│   ├── data_loader.py          # Loads processed data and prepares model inputs
│   ├── model.py                # Builds and solves the Gurobi MILP model
│   ├── main.py                 # Main script for running Gurobi experiments
│   ├── ga_model.py             # Genetic Algorithm implementation
│   ├── main_ga.py              # Main script for running GA experiments
│   ├── read_analyze.py         # Reads and analyzes output files
│   ├── mdp_enumeration.py      # Additional enumeration/analysis script
│   ├── README.md               # Source-level notes
│   └── mdp_notes.md            # Notes related to enumeration/MDP-style analysis
│
├── data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Cleaned input files
│
├── outputs/                    # Gurobi and GA results
├── report/                     # Final project report
├── requirements.txt
└── README.md
```

---

## Experiments

Ten computational experiments are conducted.

| ID | Experiment | Description |
|---|---|---|
| 1 | Baseline | Standard fixed-cost scenario |
| 2 | Fuel Shock +10% | Moderate fuel-price increase |
| 3 | Fuel Shock +25% | Severe fuel-price increase |
| 4 | Capacity Shortage | Aircraft-hour capacity reduced by 20% |
| 5 | Capacity Expansion | Aircraft-hour capacity increased by 20% |
| 6 | High Aircraft Fixed Cost | Aircraft deployment cost doubled |
| 7 | High Route Fixed Costs | Route fixed costs increased |
| 8 | Tight Demand Cap | Seat oversupply factor reduced to 1.05 |
| 9 | Category Coverage K=2 | Minimum category coverage increased |
| 10 | Strict Maintenance | Rolling-window maintenance tightened |

---

## Gurobi Results

| ID | Experiment | Status | Objective ($M) | Runtime (s) | Flights |
|---|---|---:|---:|---:|---:|
| 1 | Baseline | OPTIMAL | 574.83 | 3.88 | 4715 |
| 2 | Fuel +10% | OPTIMAL | 553.70 | 1.73 | 4680 |
| 3 | Fuel +25% | OPTIMAL | 522.41 | 1.43 | 4594 |
| 4 | Capacity Shortage | OPTIMAL | 538.42 | 10.62 | 4058 |
| 5 | Capacity Expansion | OPTIMAL | 596.08 | 0.80 | 5115 |
| 6 | High Aircraft Fixed Cost | OPTIMAL | 559.55 | 2.35 | 4670 |
| 7 | High Route Fixed Costs | OPTIMAL | 558.83 | 2.62 | 4502 |
| 8 | Tight Demand Cap | TIME LIMIT | 529.57 | 3600.70 | 4454 |
| 9 | Category Coverage K=2 | OPTIMAL | 574.83 | 13.89 | 4715 |
| 10 | Strict Maintenance | OPTIMAL | 503.73 | 15.04 | 3678 |

---

## Gurobi vs Genetic Algorithm

### Baseline Comparison

| Metric | Gurobi | GA |
|---|---:|---:|
| Status | OPTIMAL | FEASIBLE GA |
| Objective ($) | 574,833,789.73 | 573,752,765.66 |
| Difference | 0.00% | 0.188% |
| Runtime (s) | 3.88 | 1310.63 |
| Flights | 4715 | 4745 |

The GA produces a feasible solution within **0.188%** of the proven Gurobi optimum. However, Gurobi is much faster for the baseline case.

### Experiment 8 Comparison

| Metric | Gurobi | GA |
|---|---:|---:|
| Status | TIME LIMIT | FEASIBLE GA |
| Objective ($) | 529,572,607.60 | 522,512,862.10 |
| Difference | — | 1.33% |
| Runtime (s) | 2274.26 | 2274.52 |
| Flights | 4454 | 4465 |

For Experiment 8, Gurobi does not prove optimality within the time limit. The GA provides a feasible solution, but its objective is **1.33% lower** than the matched-time Gurobi incumbent.

---

## Key Findings

- The baseline scenario reaches **$574.83 million** objective value.
- Gurobi solves 9 out of 10 experiments to proven optimality.
- Fuel-price increases reduce profitability and slightly reduce flight frequency.
- Capacity shortage decreases both profit and total flights.
- Capacity expansion improves profit, but with diminishing returns.
- Tight demand cap is the most computationally difficult case.
- Category coverage `K = 2` does not change the baseline solution.
- Strict maintenance has the largest negative impact, reducing profit by **12.37%**.
- The GA provides high-quality feasible solutions but does not outperform Gurobi in the tested cases.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/ezgi-donmez/Profit-Optimal-Flight-Planning.git
cd Profit-Optimal-Flight-Planning
```

### 2. Create Environment

```bash
conda create -n flight-planning python=3.10
conda activate flight-planning
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

Main packages:

```text
pandas
numpy
gurobipy
matplotlib
```

A valid Gurobi license is required to run the MILP model.

### 4. Prepare Data

Run the data cleaning script:

```bash
python src/clean_aggregate.py
```

### 5. Run Gurobi Model

```bash
python src/main.py
```

### 6. Run Genetic Algorithm

```bash
python src/main_ga.py
```

---

## Output Interpretation

| Metric | Meaning |
|---|---|
| Objective Value | Total expected profit |
| Runtime | Solution time |
| Flights | Total assigned flights |
| Active Route-Periods | Number of operated route-months |
| Route Openings | Number of newly opened routes |
| Deployed Aircraft-Periods | Total aircraft deployment count |

For Gurobi, `OPTIMAL` means global optimality is proven.  
`TIME LIMIT` means Gurobi found a feasible solution but did not prove optimality.

For GA, `FEASIBLE GA` means the solution has zero recorded constraint violation according to the implemented feasibility checker.

---

## Limitations

This project is a strategic planning model, not a complete airline scheduling system.

Main limitations:

- Some parameters are estimated due to limited operational data.
- The model uses monthly planning periods, not daily schedules.
- Aircraft tail assignment and crew scheduling are not included.
- Passenger connections and transfer effects are not modeled.
- Maintenance is represented using an aggregate rolling-window approximation.
- The GA does not provide an optimality guarantee.
- Experiments are based on one dataset and a limited number of scenarios.

---

## Future Work

Possible extensions include:

- Stochastic demand modeling
- Robust optimization under fuel-price uncertainty
- Daily or weekly flight scheduling
- Aircraft tail assignment
- Crew scheduling constraints
- Passenger connection modeling
- Larger airline network instances
- Hybrid Gurobi-GA approaches
- Dashboard-based scenario visualization

---

## Dataset Reference

waleedfaheem, **Airline Route Profitability and Cost Analysis**, Kaggle Dataset.  
Available at: https://www.kaggle.com/datasets/waleedfaheem/airline-route-profitability-and-cost-analysis

---

## Author

**Ezgi Dönmez**  
MSc Data Science  
Özyeğin University  
DS 502 - Introduction to OR Techniques in Data Science  
May 2026
