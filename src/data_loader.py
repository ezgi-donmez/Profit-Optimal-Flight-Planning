"""
data_loader.py
--------------
Loads and structures all parameters needed for the Flight Planning MILP
from the processed CSV files in data/processed/.

Returns a single `data` dict containing sets and parameters as described
in the mathematical model 
"""

import pandas as pd
import os

# __file__ is not available in Jupyter notebooks
try:
    # Running as a script: go up one level from src/ to project root
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Running in a Jupyter notebook: cwd is already the project root
    _PROJECT_ROOT = os.path.abspath(os.getcwd())

DATA_DIR = os.path.join(_PROJECT_ROOT, 'data', 'processed')


def load_data(
    n_routes=None,
    n_months=None,
    n_aircraft=None,
    route_subset=None,
    month_subset=None,
    aircraft_subset=None,
    K_ct_value=1,
    alpha=1.0,
):
    """
    Load MILP parameters from processed CSV files.

    Parameters
    n_routes   : int,keep top-N routes by total historical flights
    n_months   : int,keep first N months (1-based)
    n_aircraft : int,keep first N aircraft types
    route_subset   : list or None  explicit route names to include
    month_subset   : list or None  explicit month numbers to include
    aircraft_subset: list or None  explicit aircraft types to include
    K_ct_value : int, minimum routes per category per period (default 1)
    alpha      : float, fleet-hour capacity multiplier (default 1.0)

    Returns
    data : dict with keys:
        routes, periods, aircraft, categories, route_category, R_c,
        pi, h, f, M, H_bar, H, L, K, combos
    """
    
    # Load raw files
    params   = pd.read_csv(os.path.join(DATA_DIR, 'params_rta.csv'))
    H_bar_df = pd.read_csv(os.path.join(DATA_DIR, 'H_bar_at.csv'))
    L_df     = pd.read_csv(os.path.join(DATA_DIR, 'L_rt.csv'))
    cat_df   = pd.read_csv(os.path.join(DATA_DIR, 'route_category.csv'))

    # Determine sets (apply filters)
    all_routes   = params['Route'].unique().tolist()
    all_months   = sorted(params['Month'].unique().tolist())
    all_aircraft = params['Aircraft_Type'].unique().tolist()

    # Filter by explicit subsets
    if route_subset is not None:
        all_routes = [r for r in route_subset if r in all_routes]
    if month_subset is not None:
        all_months = [t for t in month_subset if t in all_months]
    if aircraft_subset is not None:
        all_aircraft = [a for a in aircraft_subset if a in all_aircraft]

    # Filter by size limits
    if n_routes is not None:
        # Select top-N routes by total historical flight count
        route_totals = (
            params.groupby('Route')['flight_count'].sum()
            .sort_values(ascending=False)
        )
        all_routes = [r for r in route_totals.index if r in all_routes][:n_routes]

    if n_months is not None:
        all_months = all_months[:n_months]

    if n_aircraft is not None:
        all_aircraft = all_aircraft[:n_aircraft]

    # Route categories
    cat_map   = cat_df.set_index('Route')['Route_Category'].to_dict()
    categories = sorted(cat_df['Route_Category'].unique().tolist())

    route_category = {r: cat_map[r] for r in all_routes if r in cat_map}
    R_c = {c: [r for r in all_routes if route_category.get(r) == c]
           for c in categories}

    # Filter params to selected sets
    mask = (
        params['Route'].isin(all_routes) &
        params['Month'].isin(all_months) &
        params['Aircraft_Type'].isin(all_aircraft)
    )
    p = params[mask].copy()

    # pi_rta  – expected profit per flight (USD)
    # h_rta   – expected aircraft-hours per flight
    # f_rta   – average fuel cost per flight (USD)
    # M_rta   – max historical flights (upper bound)
    combos = set(zip(p['Route'], p['Month'], p['Aircraft_Type']))

    pi  = {}
    h   = {}
    f   = {}
    M   = {}

    for _, row in p.iterrows():
        key = (row['Route'], row['Month'], row['Aircraft_Type'])
        pi[key] = float(row['pi_rta'])
        h[key]  = float(row['h_rta'])
        f[key]  = float(row['f_rta'])
        M[key]  = int(row['M_rta'])

    # H_bar_at  – historical total aircraft-hours
    # H_at      – available aircraft-hours = alpha * H_bar_at
    hbar_mask = (
        H_bar_df['Aircraft_Type'].isin(all_aircraft) &
        H_bar_df['Month'].isin(all_months)
    )
    hb = H_bar_df[hbar_mask].copy()

    H_bar = {}
    H     = {}
    for _, row in hb.iterrows():
        key        = (row['Aircraft_Type'], row['Month'])
        H_bar[key] = float(row['H_bar_at'])
        H[key]     = alpha * float(row['H_bar_at'])

    # L_rt  – minimum flights on route r in period t if served
    l_mask = (
        L_df['Route'].isin(all_routes) &
        L_df['Month'].isin(all_months)
    )
    l = L_df[l_mask].copy()

    L = {}
    for _, row in l.iterrows():
        L[(row['Route'], row['Month'])] = int(row['L_rt'])

    # Fill missing (r,t) with 1 as a safe default
    for r in all_routes:
        for t in all_months:
            if (r, t) not in L:
                L[(r, t)] = 1

    # K_ct  – minimum routes per category per period
    K = {}
    for c in categories:
        for t in all_months:
            # Only enforce if category has routes in the filtered set
            max_possible = len(R_c[c])
            K[(c, t)] = min(K_ct_value, max_possible)

    # Pack and return
    data = {
        'routes':         all_routes,
        'periods':        all_months,
        'aircraft':       all_aircraft,
        'categories':     categories,
        'route_category': route_category,
        'R_c':            R_c,
        'pi':             pi,
        'h':              h,
        'f':              f,
        'M':              M,
        'H_bar':          H_bar,
        'H':              H,
        'L':              L,
        'K':              K,
        'combos':         combos,
    }
    return data


def print_data_summary(data):
    print("=" * 55)
    print("DATA SUMMARY")
    print("=" * 55)
    print(f"  Routes   : {len(data['routes'])}")
    print(f"  Periods  : {len(data['periods'])} months {data['periods']}")
    print(f"  Aircraft : {len(data['aircraft'])}")
    print(f"  (r,t,a) combos : {len(data['combos'])}")
    print()
    print("  Category breakdown:")
    for c, routes in data['R_c'].items():
        print(f"    {c}: {len(routes)} routes")
    print("=" * 55)
