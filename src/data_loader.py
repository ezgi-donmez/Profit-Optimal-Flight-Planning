"""
data_loader.py
"""


import os
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd


REQUIRED_CSVS = [
    "params_rta.csv",
    "H_bar_at.csv",
    "L_rt.csv",
    "route_category.csv",
]


def _project_root() -> str:
    """Return the project root assuming this file is in project_root/src."""
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.path.abspath(os.getcwd())


def _candidate_data_dirs() -> List[str]:
    """Candidate locations for the processed CSV files."""
    root = _project_root()
    cwd = os.path.abspath(os.getcwd())
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else cwd

    candidates = [
        os.environ.get("FLIGHT_DATA_DIR", ""),
        os.path.join(root, "data", "processed"),
        os.path.join(root, "data"),
        root,
        os.path.join(cwd, "data", "processed"),
        os.path.join(cwd, "data"),
        cwd,
        os.path.join(here, "..", "data", "processed"),
        os.path.join(here, "..", "data"),
        here,
    ]

    # Normalize and remove duplicates / empty strings
    cleaned = []
    for c in candidates:
        if not c:
            continue
        c = os.path.abspath(c)
        if c not in cleaned:
            cleaned.append(c)
    return cleaned


def _find_data_dir() -> str:
    """Find a directory that contains all required processed CSVs."""
    for d in _candidate_data_dirs():
        if all(os.path.exists(os.path.join(d, fn)) for fn in REQUIRED_CSVS):
            return d

    msg = ["Could not find the processed CSV files."]
    msg.append("I searched these folders:")
    for d in _candidate_data_dirs():
        msg.append(f"  - {d}")
    msg.append("Expected files:")
    for fn in REQUIRED_CSVS:
        msg.append(f"  - {fn}")
    msg.append("")
    msg.append("Fix options:")
    msg.append("  1) Put the CSV files in project_root/data/processed/")
    msg.append("  2) Or set FLIGHT_DATA_DIR to the folder containing the CSV files.")
    raise FileNotFoundError("\n".join(msg))


DATA_DIR = _find_data_dir()


def _guess_seats(aircraft_name: str) -> int:
    """Reasonable seat-capacity placeholders by aircraft type name."""
    name = str(aircraft_name).lower()
    if "a380" in name:
        return 500
    if "777" in name:
        return 360
    if "a350" in name:
        return 325
    if "787" in name:
        return 290
    if "a320" in name:
        return 180
    if "737" in name:
        return 170
    return 220


def _estimate_fleet_size(max_monthly_hours: float, aircraft_name: str) -> int:
    """
    Estimate physical aircraft count from aggregate historical monthly hours.

    Assumption:
    A physical aircraft can reasonably contribute around 120 flight-hours/month
    in this strategic planning model. Long-haul aircraft may have similar or
    slightly higher monthly utilization, but using 120 keeps the estimate simple.

    This is a placeholder because the provided CSV files do not contain fleet
    counts. The model remains consistent because H_unit is derived from H_bar
    and FleetSize, preserving the original aggregate capacity scale.
    """
    if max_monthly_hours <= 0:
        return 1
    return max(1, int(math.ceil(max_monthly_hours / 120.0)))


# math is imported here to avoid hiding it in the function body
import math


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(round(float(value)))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_data(
    n_routes: Optional[int] = None,
    n_months: Optional[int] = None,
    n_aircraft: Optional[int] = None,
    route_subset: Optional[List[str]] = None,
    month_subset: Optional[List[int]] = None,
    aircraft_subset: Optional[List[str]] = None,
    K_ct_value: int = 1,
    alpha: float = 1.0,
    oversupply_factor: float = 1.20,
    assumed_load_factor: float = 0.85,
) -> Dict[str, Any]:

    # ------------------------------------------------------------------
    # Load CSV files
    # ------------------------------------------------------------------
    params = pd.read_csv(os.path.join(DATA_DIR, "params_rta.csv"))
    H_bar_df = pd.read_csv(os.path.join(DATA_DIR, "H_bar_at.csv"))
    L_df = pd.read_csv(os.path.join(DATA_DIR, "L_rt.csv"))
    cat_df = pd.read_csv(os.path.join(DATA_DIR, "route_category.csv"))

    # Basic column validation
    required_params_cols = {
        "Route", "Month", "Aircraft_Type", "flight_count",
        "pi_rta", "h_rta", "f_rta", "M_rta",
    }
    missing_cols = sorted(required_params_cols - set(params.columns))
    if missing_cols:
        raise ValueError(f"params_rta.csv is missing columns: {missing_cols}")

    # ------------------------------------------------------------------
    # Determine selected sets
    # ------------------------------------------------------------------
    all_routes = params["Route"].dropna().unique().tolist()
    all_months = sorted(params["Month"].dropna().astype(int).unique().tolist())
    all_aircraft = params["Aircraft_Type"].dropna().unique().tolist()

    if route_subset is not None:
        all_routes = [r for r in route_subset if r in all_routes]

    if month_subset is not None:
        month_subset = [int(t) for t in month_subset]
        all_months = [t for t in month_subset if t in all_months]

    if aircraft_subset is not None:
        all_aircraft = [a for a in aircraft_subset if a in all_aircraft]

    if n_routes is not None:
        route_totals = (
            params.groupby("Route")["flight_count"]
            .sum()
            .sort_values(ascending=False)
        )
        all_routes = [r for r in route_totals.index if r in all_routes][: int(n_routes)]

    if n_months is not None:
        all_months = all_months[: int(n_months)]

    if n_aircraft is not None:
        all_aircraft = all_aircraft[: int(n_aircraft)]

    if not all_routes:
        raise ValueError("No routes selected. Check n_routes or route_subset.")
    if not all_months:
        raise ValueError("No months selected. Check n_months or month_subset.")
    if not all_aircraft:
        raise ValueError("No aircraft selected. Check n_aircraft or aircraft_subset.")

    # ------------------------------------------------------------------
    # Route categories
    # ------------------------------------------------------------------
    cat_map = cat_df.set_index("Route")["Route_Category"].to_dict()
    route_category = {r: cat_map.get(r, "Uncategorized") for r in all_routes}

    categories = sorted(set(route_category.values()))
    R_c = {
        c: [r for r in all_routes if route_category.get(r) == c]
        for c in categories
    }

    # ------------------------------------------------------------------
    # Filter route-period-aircraft rows
    # ------------------------------------------------------------------
    mask = (
        params["Route"].isin(all_routes)
        & params["Month"].astype(int).isin(all_months)
        & params["Aircraft_Type"].isin(all_aircraft)
    )
    p = params[mask].copy()
    p["Month"] = p["Month"].astype(int)

    if p.empty:
        raise ValueError("No route-period-aircraft rows after filtering.")

    # Valid combinations from existing data
    combos = sorted(set(zip(p["Route"], p["Month"], p["Aircraft_Type"])))

    pi: Dict[Tuple[str, int, str], float] = {}
    h: Dict[Tuple[str, int, str], float] = {}
    f: Dict[Tuple[str, int, str], float] = {}
    M: Dict[Tuple[str, int, str], int] = {}
    nfoc: Dict[Tuple[str, int, str], float] = {}

    for _, row in p.iterrows():
        key = (row["Route"], int(row["Month"]), row["Aircraft_Type"])
        pi[key] = _safe_float(row["pi_rta"])
        h[key] = max(0.0, _safe_float(row["h_rta"]))
        f[key] = max(0.0, _safe_float(row["f_rta"]))
        M[key] = max(0, _safe_int(row["M_rta"]))
        # If c_rta exists, use it as a non-fuel/operating-cost placeholder.
        # Otherwise keep optional NFOC inactive unless B_nfoc is passed.
        nfoc[key] = max(0.0, _safe_float(row.get("c_rta", 0.0)))

    # ------------------------------------------------------------------
    # Historical available aircraft hours
    # ------------------------------------------------------------------
    H_bar: Dict[Tuple[str, int], float] = {}
    H: Dict[Tuple[str, int], float] = {}

    hb = H_bar_df[
        H_bar_df["Aircraft_Type"].isin(all_aircraft)
        & H_bar_df["Month"].astype(int).isin(all_months)
    ].copy()
    hb["Month"] = hb["Month"].astype(int)

    for _, row in hb.iterrows():
        key = (row["Aircraft_Type"], int(row["Month"]))
        H_bar[key] = max(0.0, _safe_float(row["H_bar_at"]))
        H[key] = alpha * H_bar[key]

    # Fill missing aircraft-month capacity with 0.0 to avoid KeyError
    for a in all_aircraft:
        for t in all_months:
            H_bar.setdefault((a, t), 0.0)
            H.setdefault((a, t), alpha * H_bar[(a, t)])

    # ------------------------------------------------------------------
    # Minimum route service L[r,t]
    # ------------------------------------------------------------------
    L: Dict[Tuple[str, int], int] = {}

    ldf = L_df[
        L_df["Route"].isin(all_routes)
        & L_df["Month"].astype(int).isin(all_months)
    ].copy()
    ldf["Month"] = ldf["Month"].astype(int)

    for _, row in ldf.iterrows():
        L[(row["Route"], int(row["Month"]))] = max(0, _safe_int(row["L_rt"], 1))

    for r in all_routes:
        for t in all_months:
            L.setdefault((r, t), 1)

    # ------------------------------------------------------------------
    # Category coverage K[c,t]
    # ------------------------------------------------------------------
    K: Dict[Tuple[str, int], int] = {}
    for c in categories:
        for t in all_months:
            K[(c, t)] = min(max(0, int(K_ct_value)), len(R_c.get(c, [])))

    # ------------------------------------------------------------------
    # New final-model parameters
    # ------------------------------------------------------------------

    # Route-aircraft compatibility.
    # Assumption: If a route-aircraft pair exists in the provided params file,
    # it is compatible. Pairs not present are treated as incompatible.
    compatible_pairs = {(r, a) for (r, _, a) in combos}
    compat = {
        (r, a): 1 if (r, a) in compatible_pairs else 0
        for r in all_routes
        for a in all_aircraft
    }

    # Seat capacity placeholder from aircraft type name.
    Seats = {a: _guess_seats(a) for a in all_aircraft}

    # FleetSize and H_unit.
    # H_bar is aggregate aircraft-type capacity. We estimate FleetSize and set
    # H_unit = H_bar / FleetSize so max capacity preserves old scale:
    # H_unit[a,t] * FleetSize[a] ~= H_bar[a,t].
    FleetSize = {}
    for a in all_aircraft:
        max_h = max(H_bar.get((a, t), 0.0) for t in all_months)
        FleetSize[a] = _estimate_fleet_size(max_h, a)

    H_unit = {}
    for a in all_aircraft:
        fs = max(1, FleetSize[a])
        for t in all_months:
            H_unit[(a, t)] = H_bar.get((a, t), 0.0) / fs

    # Route-period maximum frequency.
    # Assumption: maximum total route frequency equals historical sum of M over
    # compatible aircraft types. This is safe and prevents over-scheduling.
    U_route = {}
    for r in all_routes:
        for t in all_months:
            total_m = sum(M.get((r, t, a), 0) for a in all_aircraft)
            U_route[(r, t)] = max(L.get((r, t), 0), total_m)

    # Hub slot capacity.
    # Assumption: hub capacity equals total historical maximum departures in
    # that month. This is safe and usually non-binding unless other constraints
    # try to increase total flights beyond historical levels.
    HubSlot = {}
    for t in all_months:
        HubSlot[t] = sum(U_route[(r, t)] for r in all_routes)

    # Demand estimate.
    # Assumption: passenger demand is estimated from historical maximum offered
    # seats and an assumed load factor. This avoids scheduling too many seats.
    Demand = {}
    for r in all_routes:
        for t in all_months:
            max_offered_seats = sum(
                Seats[a] * M.get((r, t, a), 0)
                for a in all_aircraft
            )
            Demand[(r, t)] = max(1, int(math.ceil(assumed_load_factor * max_offered_seats)))

    # Oversupply factor eta[r,t].
    eta = {
        (r, t): float(oversupply_factor)
        for r in all_routes
        for t in all_months
    }

    # Startup cost and recurring route fixed cost.
    # Assumption: not in the processed CSVs, so use conservative placeholders.
    # You can tune these later for scenario analysis.
    SC = {r: 0.0 for r in all_routes}
    RC = {(r, t): 0.0 for r in all_routes for t in all_months}

    # Aircraft fixed cost per deployed physical aircraft.
    # Assumption: placeholder 0 by default. main_v2.py changes this in the
    # aircraft fixed cost scenario.
    FC = {a: 0.0 for a in all_aircraft}

    # Initial route status before the horizon.
    # Assumption: routes are initially inactive, so active routes in period 1
    # are counted as opened. Since SC is 0 by default, this does not distort
    # the baseline objective unless you set SC > 0.
    Y0 = {r: 0 for r in all_routes}

    data = {
        # Sets
        "routes": all_routes,
        "periods": all_months,
        "aircraft": all_aircraft,
        "categories": categories,
        "route_category": route_category,
        "R_c": R_c,
        "combos": combos,

        # Original parameters
        "pi": pi,
        "h": h,
        "f": f,
        "M": M,
        "H_bar": H_bar,
        "H": H,
        "L": L,
        "K": K,
        "nfoc": nfoc,

        # Final-model parameters
        "FleetSize": FleetSize,
        "H_unit": H_unit,
        "SC": SC,
        "RC": RC,
        "FC": FC,
        "Demand": Demand,
        "Seats": Seats,
        "oversupply_factor": oversupply_factor,
        "eta": eta,
        "compat": compat,
        "HubSlot": HubSlot,
        "U_route": U_route,
        "Y0": Y0,

        # Useful metadata for reporting/debugging
        "DATA_DIR": DATA_DIR,
        "assumptions": {
            "FleetSize": "Estimated from H_bar_at using about 120 flight-hours per physical aircraft per month.",
            "H_unit": "H_bar_at divided by estimated FleetSize, preserving original aggregate capacity scale.",
            "Demand": "Estimated from historical maximum offered seats times assumed_load_factor.",
            "Seats": "Estimated from aircraft type name.",
            "compat": "Aircraft-route pair is compatible if it exists in params_rta.csv.",
            "HubSlot": "Set to sum of route maximum frequencies per period.",
            "U_route": "Set to sum of M_rta over compatible aircraft for each route-period.",
            "SC_RC_FC": "Default fixed costs are 0 unless changed by scenario analysis.",
        },
    }

    return data


def print_data_summary(data: Dict[str, Any]) -> None:
    """Print a compact summary of loaded data."""
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"  Data folder : {data.get('DATA_DIR', DATA_DIR)}")
    print(f"  Routes      : {len(data['routes'])}")
    print(f"  Periods     : {len(data['periods'])} months {data['periods']}")
    print(f"  Aircraft    : {len(data['aircraft'])}")
    print(f"  Valid combos: {len(data['combos'])}")
    print()
    print("  Category breakdown:")
    for c, routes in data["R_c"].items():
        print(f"    {c}: {len(routes)} routes")

    print()
    print("  Aircraft assumptions:")
    for a in data["aircraft"]:
        print(
            f"    {a:<20} FleetSize={data['FleetSize'][a]:>2} "
            f"Seats={data['Seats'][a]:>3}"
        )
    print("=" * 60)
