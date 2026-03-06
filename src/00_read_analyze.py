from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file path 
DATA_PATH = Path(r"C:\Users\ed024981\Desktop\DS502project\airline_route_profitability.xlsx")

# Output folder 
OUT_DIR = Path(r"C:\Users\ed024981\Desktop\DS502project\outputs\eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH.exists(), OUT_DIR

# Read Excel into a pandas DataFrame
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Show dimensions of df
print("Shape:", df.shape)

# Preview first rows
display(df.head(5))
