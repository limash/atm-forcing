# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that downloads NORA3 atmospheric reanalysis data from the met.no THREDDS
OPeNDAP server and regrids it into forcing files for ocean models (ROMS-style).
Covers the Oslofjord region by default.

Install is hybrid (see `README.md`): conda-forge supplies the ESMF stack
(`esmf`/`esmpy`/`xesmf`, none on PyPI), then `uv pip install -e .` (run inside
the activated conda env) installs the project + PyPI deps into that same env.
`uv pip install` auto-targets the active env via `CONDA_PREFIX` — no env vars —
and, unlike `uv sync`, doesn't remove the conda packages.

## Commands

```bash
# Download + regrid to a rectilinear lat/lon grid (the main entry point)
python app/nora3.py -o /where/to/save [--start-year 2010 --end-year 2020]

# Same, but regrid onto a curvilinear ROMS grid (reads FILE_PATH_GRID, rotates winds)
python app/nora3.py -o /where/to/save --use-roms

# Download raw NORA3 (no regridding/variable selection)
python app/nora3_download.py -o /where/to/save --start-year 2020 --end-year 2025

# Merge --use-roms's per-variable daily files into one file per variable per month
python app/nora3_merge.py -o /where/to/save [--year 2020]

# Lint
ruff check .
```

There is no test suite. `notebooks/nora3.ipynb` is exploratory.

## Architecture

- `atm_forcing/stuff.py` — all the real logic (re-exported via `atm_forcing/__init__.py`).
  The `app/` scripts are thin CLI wrappers around it.
- `generate_catalog_urls(start_year, end_year)` yields one THREDDS catalog URL per
  6-hour analysis cycle (hours 0, 6, 12, 18). Each day = 4 cycles; the app scripts
  accumulate 4 cycles' worth of data and, at the day boundary, combine them and
  dedup the overlapping time index before writing. Existing output files are
  skipped, so runs are resumable.
- Within each catalog only datasets whose key contains `_fp` are used.

### Two regridding paths (mutually exclusive)

- `get_ds` → `regrid`: bilinear interpolation onto a rectilinear lat/lon grid
  (`LAT_NEW`/`LON_NEW` constants in `app/nora3.py`). Writes one merged `YYYYMMDD.nc`
  per day containing all variables.
- `get_ds_roms` → `regrid_curvilinear`: bilinear interpolation onto a curvilinear
  ROMS grid read from `FILE_PATH_GRID` (`lat_rho`/`lon_rho`). Additionally rotates
  the regridded winds into grid-relative `x_wind_10m`/`y_wind_10m` using the grid's
  `angle` field. `FILE_PATH_GRID` defaults to a hardcoded Sigma2/NIRD project path,
  overridable via `--file-path-grid`. Writes one `<roms_name>_YYYYMMDD.nc` per
  variable per day (8 files/day), since `get_ds_roms` returns a `{roms_name: da}`
  dict instead of a merged dataset.

### Non-obvious data handling

- `CF_ROMS` in `stuff.py` is the canonical variable table:
  `(NORA3 name, height dim to squeeze, ROMS name, ROMS time coord)`. `nora3.py`
  selects its source variables from the first column of this tuple. The 4th column
  (`ROMS_TIME_DIMS`) names each ROMS variable's own time dimension (`wind_time`,
  `swrad_time`, ...), used only by the `get_ds_roms` path.
- Accumulated fluxes (shortwave, longwave, precipitation — flagged "accumulated"
  in `CF_ROMS`) are de-accumulated in `deaccumulate()` with `.diff(dim="time") / 3600`
  to get per-second rates, then relabeled to the interval midpoint (`midpoint_time()`)
  and moved onto their own time dimension — `time_acc` for `get_ds`, the variable's
  `ROMS_TIME_DIMS` entry for `get_ds_roms` — since `diff` drops the first timestep and
  midpoint labels never coincide with the instantaneous variables' `time` labels.
  Merging both onto one shared `time` dimension silently produces all-NaN
  instantaneous variables (confirmed by direct reproduction), which is why they're
  kept on separate dimensions instead.
- Air temperature is converted Kelvin → Celsius (`-= 273.15`).
- Wind handling: `get_winds` derives east/north components from NORA3's
  `x_wind_10m`/`y_wind_10m` by computing the grid rotation `angle` from the 2D
  lat/lon coords (`lonlat_to_angle`) and de-rotating (`rotate_u_v`).
- `get_ds` output is always written zlib-compressed at `complevel=5`. `get_ds_roms` output
  (the per-variable ROMS files) is written uncompressed, intentionally.

## Conventions

- ruff with line-length 120; rule set in `pyproject.toml` (E, F, UP, B, SIM, I).
- The package name is `atm-forcing` but the import name is `atm_forcing`.
