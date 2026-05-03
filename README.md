# Profit-Optimal Flight Planning Using Route Profitability Data

This repository contains the implementation and computational analysis of a profit-optimal airline route-frequency and fleet planning problem. The project was developed for **DS 502 - Introduction to OR Techniques in Data Science** as part of the **MSc Data Science** program at Özyeğin University.

The study formulates the airline planning problem as a **Mixed-Integer Linear Programming (MILP)** model and solves it using **Gurobi**. In addition, a **Genetic Algorithm (GA)** is implemented as a heuristic benchmark to evaluate solution quality under computationally challenging scenarios.

---

## Project Overview

Airlines must decide which routes to operate, how frequently to serve them, and which aircraft types to deploy. These decisions affect profitability, fleet utilization, service coverage, and operational feasibility.

This project addresses a strategic airline planning problem for an airline operating from **Dubai International Airport (DXB)** to multiple international destinations over a monthly planning horizon.

The model determines:

- Number of flights operated on each route, period, and aircraft type
- Whether a route is active in each period
- Whether a route is newly opened
- Number of physical aircraft deployed by aircraft type and period

The objective is to **maximize total expected profit** while considering operational and business constraints.

---

## Dataset

The project uses the following dataset:

**Airline Route Profitability and Cost Analysis**  
Published on Kaggle by **waleedfaheem**

Dataset link:  
https://www.kaggle.com/datasets/waleedfaheem/airline-route-profitability-and-cost-analysis

The processed instance used in this project includes:

| Component | Description |
|---|---|
| Hub Airport | Dubai International Airport (DXB) |
| Routes | 30 international destinations |
| Planning Horizon | 12 monthly periods |
| Aircraft Types | 6 aircraft types |
| Route Categories | Short Haul, Medium Haul, Long Haul |
| Valid Route-Period-Aircraft Combinations | 840 |
| Gurobi Decision Variables | 1632 |
| Gurobi Constraints | 3528 |

---

## Mathematical Model

The optimization problem is formulated as a **MILP**.

### Main Decision Variables

| Variable | Description |
|---|---|
| `x[r,t,a]` | Number of flights on route `r`, in period `t`, using aircraft type `a` |
| `y[r,t]` | Binary variable equal to 1 if route `r` is active in period `t` |
| `z[r,t]` | Binary variable equal to 1 if route `r` is newly opened in period `t` |
| `q[a,t]` | Number of physical aircraft of type `a` deployed in period `t` |

### Objective

The objective function maximizes total expected profit by considering:

- Flight-level expected profit
- Fuel-price shock effects
- Route startup costs
- Recurring route fixed costs
- Aircraft deployment fixed costs

### Constraints

The model includes the following constraint groups:

- Route activation and flight-frequency linking
- Minimum and maximum service requirements
- Aircraft-hour capacity constraints
- Aircraft fleet availability constraints
- Rolling-window maintenance approximation
- Demand-based seat capacity limits
- Hub slot capacity limits
- Route-aircraft compatibility
- Route category coverage
- Route opening logic
- Minimum up-time restrictions for newly opened routes

---

## Solution Approaches

Two solution approaches are implemented.

---

### 1. Gurobi MILP Solver

The MILP model is implemented in Python using the `gurobipy` package.

Gurobi is used as the exact benchmark solver. It solves the model using branch-and-bound, presolve, cutting planes, and MILP optimization techniques.

The solver configuration used in the experiments is:

| Setting | Value |
|---|---|
| Time Limit | 3600 seconds |
| MIP Gap Tolerance | 0.00 |
| Output Verbosity | Enabled |

Gurobi provides proven optimal solutions for most experiments. For more difficult cases, it may return a feasible incumbent solution without proving global optimality within the time limit.

---

### 2. Genetic Algorithm

A Genetic Algorithm is implemented as a heuristic alternative.

The GA uses a model-aware chromosome structure that represents:

- Flight-frequency decisions
- Route activation decisions
- Aircraft deployment decisions

The GA includes:

- Greedy-seeded and random initial population generation
- Tournament selection
- Route-based crossover
- Route-block mutation
- Penalty-based fitness evaluation
- Multi-stage feasibility repair
- Greedy local search improvement
- Elitism
- Stall-based termination

The GA does not guarantee global optimality, but it can generate high-quality feasible solutions, especially when the MILP becomes computationally difficult.

---

