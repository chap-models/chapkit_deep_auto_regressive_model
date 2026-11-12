# Data preparation for the `full_rich` configuration

These scripts build the feature-rich training CSV used by [`configs/monthly_full_rich.yaml`](../../configs/monthly_full_rich.yaml),
the configuration that beats the MSTL champion on the Rwanda disease-case dataset on every
evaluation window (full dataset: log-CRPS 0.3184 vs champion 0.3254; active-spray window:
0.3330 vs 0.3596).

The scripts are reproduction tooling for the reference Rwanda dataset. They read the source
files from, and write their outputs to, the directory in `AR_DATA_DIR` (default: the current
directory):

```bash
AR_DATA_DIR=/path/to/data uv run python scripts/data_prep/make_spatial_features.py
AR_DATA_DIR=/path/to/data uv run python scripts/data_prep/make_full_failed_dataset.py
```

## Pipeline

1. **`make_spatial_features.py`** — from the ADM5 boundary geojson
   (`geoBoundaries-RWA-ADM5.geojson`, real polygons; all 406 sectors match by id) compute
   per-sector polygon centroids and an edge-adjacency neighbour graph (two sectors are
   neighbours if their boundaries share an edge — detected as ≥2 shared vertices, no shapely
   dependency). Writes `sector_spatial.csv` (centroid_lon/lat, n_neighbours) and
   `sector_neighbours.json`.

2. **`make_full_failed_dataset.py`** — merge onto the base AR CSV
   (`spray_monthly_ar_champion_cov.csv`, which already carries the champion IRS timing
   features and `irs_decay_<class>`):
   - **resistance**: `failed_total = Σ_class irs_decay_class × (1 − mort_class/100)` —
     resistance-weighted spray decay, from `sector_month_resistance.csv`.
   - **riskmap statics**: `sig_temp, focal_habitat, focal_builtup, env3d_risk_oof` from
     `sector_month_riskmap_features.csv` (NaN filled with the column mean over valid
     locations; `env3d_risk_full` deliberately dropped — it is the in-sample fit and leaks).
   - **spatial**: centroids + neighbour-mean climate (rainfall / mean_temperature /
     relative_humidity), exogenous so they do not leak the target over the horizon.

   Writes `spray_ar_full_rich.csv` (the `full_rich` training set, 22 covariates) and
   `spray_ar_full_failed.csv` (climate + `failed_total` only, for ablation).

## Required source files

| file | provides |
|---|---|
| `geoBoundaries-RWA-ADM5.geojson` | sector polygons (centroids, neighbour graph) |
| `spray_monthly_ar_champion_cov.csv` | base climate + champion IRS timing features + `irs_decay_<class>` |
| `sector_month_resistance.csv` | per-class insecticide-resistance mortality (`mort_<class>`) |
| `sector_month_riskmap_features.csv` | static riskmap covariates |

## Notes

- The geojson centroid mapping `sector → district` distributed as
  `spray_sector_to_district.geojson` is **not** usable — its centroids are all `(0,0)`. Use
  the real `geoBoundaries-RWA-ADM5.geojson` boundaries instead.
- On the full dataset the champion IRS timing features are the backbone; `failed_total`
  alone (without them) underperforms because it is zero ~78 % of the timeline.
- The 22-covariate set only wins **with regularization** (`dropout_rate: 0.3`,
  `input_dropout_rate: 0.25`, early stop `n_iter: 300`); unregularized it overfits.
