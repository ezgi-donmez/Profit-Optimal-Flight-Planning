from pathlib import Path
import pandas as pd
import numpy as np

# Paths
DATA_PATH  = Path(r"C:\Users\ed024981\Desktop\DS502project\airline_route_profitability.xlsx")
PROC_DIR   = Path(r"C:\Users\ed024981\Desktop\DS502project\data\processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


# Loading and basic cleaning
df = pd.read_excel(DATA_PATH, engine="openpyxl")
df["Flight_Date"] = pd.to_datetime(df["Flight_Date"], errors="coerce")

# Drop rows with invalid dates (safety check)
n_before = len(df)
df = df.dropna(subset=["Flight_Date"])
print(f"Rows dropped (invalid date): {n_before - len(df)}")

# Add Month index (1–12)
df["Month"] = df["Flight_Date"].dt.month

# Recompute Profit from components to ensure consistency
# (handles any rounding errors in raw data)
df["Profit_clean"] = df["Total_Revenue"] - df["Total_Cost"]

# Flag large discrepancies (>$1) between raw Profit and recomputed
inconsistent = (df["Profit"] - df["Profit_clean"]).abs() > 1.0
print(f"Rows with Profit inconsistency > $1: {inconsistent.sum()} — using recomputed values")
df["Profit"] = df["Profit_clean"]
df.drop(columns=["Profit_clean"], inplace=True)

# Drop exact duplicates on key identity columns
key_cols = ["Flight_Number", "Flight_Date", "Origin", "Destination", "Aircraft_Type"]
n_before = len(df)
df = df.drop_duplicates(subset=key_cols)
print(f"Rows dropped (key duplicates): {n_before - len(df)}")

print(f"\nClean dataset: {df.shape[0]} rows, {df.shape[1]} cols")

# ROUTE–CATEGORY LOOKUP 
# Used for K_ct constraint in the MILP
route_category = (
    df[["Route", "Route_Category"]]
    .drop_duplicates()
    .sort_values("Route")
    .reset_index(drop=True)
)
# each route should map to exactly one category
multi_cat = route_category.groupby("Route")["Route_Category"].nunique()
assert (multi_cat > 1).sum() == 0, "Some routes map to multiple categories — check raw data!"
print(f"\nRoute-category mapping: {len(route_category)} routes")
print(route_category.groupby("Route_Category")["Route"].count().to_string())

route_category.to_csv(PROC_DIR / "route_category.csv", index=False)
print("  Saved → route_category.csv")

# PARAMS_RTA  — per (Route, Month, Aircraft_Type)
#    π_rta : mean profit per flight
#    h_rta : mean flight hours per flight
#    f_rta : mean fuel cost per flight
#    M_rta : historical max flight count  (upper bound for x_rta)
#    L_rt  : min flights observed when route was served (lower bound)

rta = (
    df.groupby(["Route", "Month", "Aircraft_Type"])
    .agg(
        pi_rta        = ("Profit",       "mean"),   # expected profit per flight
        h_rta         = ("Flight_Hours", "mean"),   # expected flight-hours per flight
        f_rta         = ("Fuel_Cost",    "mean"),   # avg fuel cost (for shock scenarios)
        flight_count  = ("Profit",       "count"),  # number of observed flights
        revenue_mean  = ("Total_Revenue","mean"),   # informational
        cost_mean     = ("Total_Cost",   "mean"),   # informational
        load_factor   = ("Load_Factor",  "mean"),   # informational
    )
    .reset_index()
)

# M_rta: use the observed flight count as the upper bound
# (how many times this r,t,a combo was flown historically)
rta["M_rta"] = rta["flight_count"]

# Round continuous parameters to 2 decimal places
for col in ["pi_rta", "h_rta", "f_rta", "revenue_mean", "cost_mean", "load_factor"]:
    rta[col] = rta[col].round(2)

print(f"\nparams_rta: {len(rta)} (r,t,a) triples")
print(rta.describe().round(2).to_string())

rta.to_csv(PROC_DIR / "params_rta.csv", index=False)
print("\n  Saved → params_rta.csv")

# L_RT  — minimum flights per (Route, Month) if served
# Defined as the minimum observed flight count across aircraft types for that route-month (conservative lower bound)

lrt = (
    rta.groupby(["Route", "Month"])["flight_count"]
    .min()
    .reset_index()
    .rename(columns={"flight_count": "L_rt"})
)
# Floor at 1 — if a route was served, at least 1 flight must be scheduled
lrt["L_rt"] = lrt["L_rt"].clip(lower=1)

print(f"\nL_rt (min service levels): {len(lrt)} (r,t) pairs")
print(lrt["L_rt"].describe().round(2).to_string())

lrt.to_csv(PROC_DIR / "L_rt.csv", index=False)
print("  Saved → L_rt.csv")

# H_BAR_AT  — historical total fleet-hours per (Aircraft_Type, Month)
#    H̄_at = sum of Flight_Hours for aircraft a in month t
#    H_at = α_at * H̄_at  (α is the capacity multiplier in experiments)

hat = (
    df.groupby(["Aircraft_Type", "Month"])["Flight_Hours"]
    .sum()
    .reset_index()
    .rename(columns={"Flight_Hours": "H_bar_at"})
)
hat["H_bar_at"] = hat["H_bar_at"].round(2)

print(f"\nH_bar_at: {len(hat)} (a,t) pairs")
print(hat.groupby("Aircraft_Type")["H_bar_at"].agg(["mean","min","max"]).round(2).to_string())

hat.to_csv(PROC_DIR / "H_bar_at.csv", index=False)
print("  Saved → H_bar_at.csv")

# SUMMARY REPORT
print("\n" + "=" * 55)
print("PROCESSED FILES SUMMARY")
print("=" * 55)
files = {
    "params_rta.csv"     : f"{len(rta)} rows — π_rta, h_rta, f_rta, M_rta per (r,t,a)",
    "L_rt.csv"           : f"{len(lrt)} rows — min flights per (r,t) if served",
    "H_bar_at.csv"       : f"{len(hat)} rows — historical fleet-hours per (a,t)",
    "route_category.csv" : f"{len(route_category)} rows — route → category mapping",
}
for fname, desc in files.items():
    print(f"  {fname:<22}: {desc}")

# Model dimension check
R = rta["Route"].nunique()
T = rta["Month"].nunique()
A = rta["Aircraft_Type"].nunique()
print(f"\nModel dimensions:")
print(f"  |R| = {R} routes")
print(f"  |T| = {T} months")
print(f"  |A| = {A} aircraft types")
print(f"  x_rta variables (upper bound): {R*T*A}")
print(f"  y_rt  variables (upper bound): {R*T}")
print(f"  Observed (r,t,a) triples    : {len(rta)}  ({len(rta)/(R*T*A)*100:.1f}% of possible)")

print("\n 01_clean_aggregate.py complete.")
