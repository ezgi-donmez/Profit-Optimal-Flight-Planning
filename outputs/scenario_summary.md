# Scenario Output Summary

This file summarizes the main computational output of the updated Profit-Optimal Flight Planning MILP. The full raw terminal output is stored separately in `outputs/full_run_output.txt`.

## Solver Setup

The model was solved using Gurobi Optimizer 13.0.1 with an academic license. The solver was run with a time limit and a 1% MIP gap tolerance. Therefore, medium, large, and full-dataset solutions should be interpreted as optimal within the selected MIP gap tolerance.

---

## Baseline Instance Summary

| Instance | Routes | Months | Aircraft | Combos | Variables | Constraints | Objective (USD) | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small baseline | 10 | 3 | 3 | 36 | 94 | 167 | $25,604,427.93 | 0.016 s |
| Medium baseline | 20 | 8 | 4 | 232 | 576 | 1072 | $127,270,423.45 | 0.043 s |
| Large baseline | 16 | 12 | 6 | 504 | 1000 | 1742 | $492,532,011.34 | 0.059 s |
| Full dataset baseline | 30 | 12 | 6 | 840 | 1644 | 2820 | $601,190,383.19 | 0.090 s |

The full dataset baseline was successfully solved using the academic Gurobi license. It includes 30 routes, 12 months, 6 aircraft types, and 840 feasible route-period-aircraft combinations.

---

## Small-Instance Scenario Results

| Scenario | Parameter Setting | Objective (USD) | Active Route-Periods | Route Openings | Aircraft Deployments |
|---|---|---:|---:|---:|---:|
| Baseline | delta = 0.00, alpha = 1.00 | $25,604,427.93 | 29 | 0 | 9 |
| Fuel shock | delta = 0.10 | $24,692,000.76 | 27 | 0 | 9 |
| Fuel shock | delta = 0.25 | $23,324,466.52 | 27 | 0 | 9 |
| Capacity reduction | alpha = 0.30 | $19,734,837.15 | 22 | 1 | 9 |
| Capacity reduction | alpha = 0.60 | $24,975,208.99 | 27 | 0 | 9 |
| Minimum up-time | N_min = 1 | $25,609,775.21 | 28 | 1 | 9 |
| Minimum up-time | N_min = 2 | $25,604,427.93 | 29 | 0 | 9 |
| Minimum up-time | N_min = 3 | $25,569,357.49 | 30 | 0 | 9 |
| Maintenance window | N_maint = 1 | $25,604,427.93 | 29 | 0 | 9 |
| Maintenance window | N_maint = 2 | $25,604,427.93 | 29 | 0 | 9 |
| Maintenance window | N_maint = 3 | $25,604,427.93 | 29 | 0 | 9 |
| Maintenance intensity | gamma = 0.30 | $25,349,489.66 | 27 | 0 | 9 |
| Activation cost | FC_a = 50,000 | $25,154,427.93 | 29 | 0 | 9 |
| Activation cost | FC_a = 100,000 | $24,704,427.93 | 29 | 0 | 9 |

---

## Large-Instance Scenario Results

| Scenario | Parameter Setting | Objective (USD) | Active Route-Periods | Route Openings | Aircraft Deployments |
|---|---|---:|---:|---:|---:|
| Large baseline | delta = 0.00, gamma = 0.20 | $492,532,011.34 | 147 | 4 | 69 |
| Large fuel shock | delta = 0.10 | $476,064,794.23 | 144 | 3 | 69 |
| Large fuel shock | delta = 0.25 | $451,810,631.90 | 143 | 5 | 69 |
| Large maintenance | gamma = 0.00 | $510,633,688.24 | 150 | 4 | 69 |
| Large maintenance | gamma = 0.20 | $492,532,011.34 | 147 | 4 | 69 |
| Large maintenance | gamma = 0.40 | $378,414,046.95 | 136 | 10 | 62 |

---

## Full Dataset Baseline

| Metric | Value |
|---|---:|
| Routes | 30 |
| Months | 12 |
| Aircraft types | 6 |
| Feasible route-period-aircraft combinations | 840 |
| Variables | 1644 |
| Constraints | 2820 |
| Objective value | $601,190,383.19 |
| Active route-periods | 249 |
| Route openings | 13 |
| Runtime | 0.090 s |
| Optimality gap | 0.9966% |

The full dataset baseline confirms that the updated implementation can solve the complete 30-route instance when the academic Gurobi license is active. The model produced an objective value of $601.19 million, with 249 active route-periods and 13 route openings. Since the solver uses a 1% MIP gap tolerance, this result should be reported as optimal within tolerance.

---

## Interpretation of Results

The scenario outputs behave consistently with the model logic. Increasing the fuel-shock factor reduces the objective value because higher fuel cost lowers the adjusted profit per flight. Reducing capacity also decreases the objective value because the model has fewer available fleet-hours to allocate profitable flights. Increasing the aircraft activation cost reduces total profit because each deployed aircraft type becomes more expensive.

The maintenance-window experiment does not change the small-instance objective, which suggests that the small instance has enough capacity slack under gamma = 0.20. However, the maintenance-intensity experiment shows a clearer effect. In the large instance, increasing gamma from 0.00 to 0.40 reduces the objective from $510.63 million to $378.41 million. This shows that maintenance-related capacity reduction becomes more important in larger instances where aircraft are used repeatedly across many periods.

The full dataset result is useful as the strongest computational evidence because it includes all 30 routes, all 12 months, and 6 aircraft types. It demonstrates that the model can scale beyond the small and medium test cases when the academic license is available.
