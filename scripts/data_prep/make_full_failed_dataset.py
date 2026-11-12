"""Full-timeline AR datasets with the resistance 'failed protection' feature.

failed_<class> = irs_decay_<class> * (1 - mort_<class>/100)   (resistance-weighted decay)
failed_total   = sum over the four insecticide classes.

Base = spray_monthly_ar_champion_cov.csv (full 2013-01..2026-01; has irs_decay_<class>
and the champion IRS timing features). Resistance from sector_month_resistance.csv.

Writes two CSVs:
  spray_ar_full_failed.csv  -- climate + relative_humidity + failed_total   (mirror of the
                               lastspray winner, on the full timeline)
  spray_ar_full_rich.csv    -- the above + champion IRS timing features + riskmap statics
                               + spatial (centroids, neighbour-mean climate)
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

# Directory holding the source CSVs and receiving the outputs (default: current directory).
DATA_DIR = Path(os.environ.get("AR_DATA_DIR", "."))

BASE = DATA_DIR / "spray_monthly_ar_champion_cov.csv"
RES = DATA_DIR / "sector_month_resistance.csv"
RISK = DATA_DIR / "sector_month_riskmap_features.csv"
SPAT = DATA_DIR / "sector_spatial.csv"
NBRS = DATA_DIR / "sector_neighbours.json"

CLASS_MORT = {  # insecticide class -> resistance mortality column
    "carbamate": "mort_bendiocarb",
    "pyrethroid": "mort_deltamethrin",
    "organophosphate": "mort_pirimiphos_methyl",
    "clothianidin": "mort_clothianidin",
}
CLIMATE = ["rainfall", "mean_temperature", "relative_humidity"]
CHAMP_IRS = [
    "irs_level",
    "irs_since",
    "irs_cumulative",
    "irs_decay2",
    "irs_decay8",
    "irs_recent3",
    "irs_recent6",
    "irs_recent12",
    "irs_rounds12",
]
RISK_COLS = ["sig_temp", "focal_habitat", "focal_builtup", "env3d_risk_oof"]
KEEP_BASE = [
    "time_period",
    "location",
    "disease_cases",
    "rainfall",
    "mean_temperature",
    "population",
    "relative_humidity",
]


def num(v):
    """Parse v as a float, returning None for NaN or unparseable values."""
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def main():
    """Build and write the failed-only and rich full-timeline datasets."""
    with open(BASE) as f:
        rows = list(csv.DictReader(f))

    # resistance: (loc, time) -> {class: sus_fraction}
    res = {}
    with open(RES) as f:
        for r in csv.DictReader(f):
            res[(r["location_id"], r["time"])] = {cls: num(r.get(col)) for cls, col in CLASS_MORT.items()}

    # failed_total per row
    for r in rows:
        key = (r["location"], r["time_period"])
        rr = res.get(key, {})
        tot = 0.0
        for cls in CLASS_MORT:
            decay = num(r.get(f"irs_decay_{cls}")) or 0.0
            mort = rr.get(cls)
            resist = 1.0 - (mort / 100.0) if mort is not None else 0.0
            tot += decay * resist
        r["failed_total"] = f"{tot:.6f}"

    # --- riskmap (static per location, fill NaN with col mean) ---
    risk = {}
    csum = {c: 0.0 for c in RISK_COLS}
    cn = {c: 0 for c in RISK_COLS}
    with open(RISK) as f:
        for r in csv.DictReader(f):
            risk[(r["location_id"], r["time"])] = {c: r[c] for c in RISK_COLS}
            for c in RISK_COLS:
                x = num(r[c])
                if x is not None:
                    csum[c] += x
                    cn[c] += 1
    rmean = {c: (csum[c] / cn[c] if cn[c] else 0.0) for c in RISK_COLS}

    # --- spatial ---
    cent = {}
    with open(SPAT) as f:
        for r in csv.DictReader(f):
            cent[r["location"]] = (r["centroid_lon"], r["centroid_lat"])
    nbrs = json.load(open(NBRS))
    clim = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for c in CLIMATE:
            x = num(r.get(c))
            if x is not None:
                clim[r["time_period"]][r["location"]][c] = x

    def nbr_mean(t, loc, col):
        vals = [clim[t][nb][col] for nb in nbrs.get(loc, []) if col in clim[t].get(nb, {})]
        if vals:
            return sum(vals) / len(vals)
        return clim[t].get(loc, {}).get(col, 0.0)

    # --- write failed-only dataset ---
    cols_failed = KEEP_BASE + ["failed_total"]
    with open(DATA_DIR / "spray_ar_full_failed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_failed, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- write rich dataset ---
    spat_cols = ["centroid_lon", "centroid_lat"] + [f"nbr_{c}" for c in CLIMATE]
    cols_rich = KEEP_BASE + ["failed_total"] + CHAMP_IRS + RISK_COLS + spat_cols
    with open(DATA_DIR / "spray_ar_full_rich.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_rich, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            loc, t = r["location"], r["time_period"]
            r["centroid_lon"], r["centroid_lat"] = cent.get(loc, ("0", "0"))
            for c in CLIMATE:
                r[f"nbr_{c}"] = f"{nbr_mean(t, loc, c):.6f}"
            rk = risk.get((loc, t), {})
            for c in RISK_COLS:
                x = num(rk.get(c))
                r[c] = f"{x:.6f}" if x is not None else f"{rmean[c]:.6f}"
            # ensure champion IRS numeric (fill NaN 0)
            for c in CHAMP_IRS:
                if num(r.get(c)) is None:
                    r[c] = "0"
            w.writerow(r)
    print("wrote spray_ar_full_failed.csv:", len(rows), "rows, cols:", cols_failed[3:])
    print("wrote spray_ar_full_rich.csv  :", len(rows), "rows, cols:", cols_rich[3:])


if __name__ == "__main__":
    main()
