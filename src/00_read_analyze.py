from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file path 
DATA_PATH = Path(r"C:\Users\ed024981\Desktop\DS502project\airline_route_profitability.xlsx")
# Output folder 
OUT_DIR = Path(r"C:\Users\ed024981\Desktop\DS502project\outputs\eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Read Excel
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Data info
print("Shape:", df.shape) # dimension preview
display(df.head(5)) # first rows
display(pd.DataFrame({"columns": df.columns})) # columns name
display(df.dtypes) # data types

# Check Dates 
df["Flight_Date"] = pd.to_datetime(df["Flight_Date"], errors="coerce") # Convert Flight_Date to datetime 
invalid_dates = df["Flight_Date"].isna().sum() # check invalid dates
print("Invalid Flight_Date rows:", invalid_dates)
print("Date min:", df["Flight_Date"].min()) # data start date
print("Date max:", df["Flight_Date"].max()) # data end date

# count uniques in the dataset
summary = {
    "rows": len(df),
    "cols": df.shape[1],
    "unique_origins": df["Origin"].nunique(),
    "unique_destinations": df["Destination"].nunique(),
    "unique_routes": df["Route"].nunique(),
    "unique_aircraft_types": df["Aircraft_Type"].nunique(),
    "unique_seasons": df["Season"].nunique(),
    "unique_demand_levels": df["Demand_Level"].nunique(),
    "unique_route_categories": df["Route_Category"].nunique(),
}
print(summary)

# Count missing values per column 
# Create a clean table for non-zero missing columns
missing = df.isna().sum()
missing_tbl = (
    pd.DataFrame({
        "missing_count": missing,
        "missing_rate_%": missing / len(df) * 100 })
    .query("missing_count > 0")
    .sort_values("missing_count", ascending=False)
)
print(missing_tbl)

# Check duplicates across all columns and for flight identity
dup_all = df.duplicated().sum()
key_cols = ["Flight_Number", "Flight_Date", "Origin", "Destination", "Aircraft_Type"]
dup_key = df.duplicated(subset=key_cols).sum()
print("Duplicates (all columns):", dup_all)
print("Duplicates (key cols):", dup_key)

# Check if the dataset is internally consistent
profit_err = (df["Profit"] - (df["Total_Revenue"] - df["Total_Cost"])).abs() # Profit should equal Total_Revenue - Total_Cost
print("Profit identity max abs error:", profit_err.max())
print("Profit identity mean abs error:", profit_err.mean())

# Total_Revenue should equal Ticket_Revenue + Ancillary_Revenue when ancillary exists
mask = df["Ancillary_Revenue"].notna()
rev_err = (df.loc[mask, "Total_Revenue"] - (df.loc[mask, "Ticket_Revenue"] + df.loc[mask, "Ancillary_Revenue"])).abs()
print("Revenue identity max abs error (where ancillary present):", rev_err.max())
print("Revenue identity mean abs error (where ancillary present):", rev_err.mean())


