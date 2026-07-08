# NORA3 data fetching and processing for ocean models.

## Installation

The regridding stack (ESMF and its Python bindings `esmpy`/`xesmf`) is **not**
available on PyPI, so it must come from conda-forge. Everything else is managed
by [uv](https://docs.astral.sh/uv/): conda provides the ESMF binaries, then uv
installs the project and its PyPI dependencies into the same environment.

1. Install [miniforge/conda](https://conda-forge.org/download/) and
   [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone and enter the repo:
   ```bash
   git clone https://github.com/limash/atm-forcing.git
   cd atm-forcing
   ```
3. Create a conda environment providing the ESMF regridding stack:
   ```bash
   conda create -n atm-forcing -c conda-forge python=3.12 esmf esmpy xesmf
   conda activate atm-forcing
   ```
4. With that env active, install the project and the remaining (PyPI)
   dependencies with uv:
   ```bash
   uv pip install -e .
   ```
   `uv pip install` automatically targets the activated conda env (via
   `CONDA_PREFIX`) — no environment variables to set — and, unlike `uv sync`,
   leaves the conda-provided `esmf`/`esmpy`/`xesmf` untouched. (Use a named env
   as above, not the conda `base` env, which uv treats as a system environment.)

> A single tool cannot install both the conda and PyPI pieces, because `esmpy`
> is conda-only and uv has no conda support. If you'd prefer one tool for
> everything, use [pixi](https://pixi.sh) instead of conda + uv.

## Usage

With the `atm-forcing` env active, run one of the scripts below. All of them download
6-hourly NORA3 analysis cycles from the met.no THREDDS OPeNDAP server; already-downloaded
daily files are skipped, so runs are resumable after interruptions or failures.
**Note:** downloading a full year range can take several days.

### Download + regrid to a lat–lon grid (main entry point)
```bash
python app/nora3.py -o /where/to/save [--start-year 2009] [--end-year 2023]
```
Bilinearly regrids onto a rectilinear lat–lon grid covering the Oslofjord region
(`LAT_NEW`/`LON_NEW` in `app/nora3.py` — edit these to change the domain), writing
one merged `<output>/daily/YYYYMMDD.nc` per day.

### Download + regrid onto a ROMS grid
```bash
python app/nora3.py -o /where/to/save --use-roms --file-path-grid /path/to/grid.nc
```
Bilinearly regrids onto a curvilinear ROMS grid (`lat_rho`/`lon_rho`) and rotates winds
into grid-relative `x_wind_10m`/`y_wind_10m` using the grid's `angle` field. Since each
ROMS variable has its own time dimension, this writes one file per variable per day:
`<output>/daily/<roms_name>_YYYYMMDD.nc` (8 files/day, uncompressed).

### Merge daily ROMS files into monthly files
```bash
python app/nora3_merge.py -o /where/to/save [--year 2020]
```
For `--use-roms` output only: merges the per-variable daily files in `<output>/daily/`
into one file per variable per month in `<output>/monthly/`, skipping months that already
have an output file and warning about duplicate or missing timestamps. Omit `--year` to
process all available years.

### Download raw NORA3 (no regridding)
```bash
python app/nora3_download.py -o /where/to/save [--start-year 2020 --end-year 2025]
```
Downloads the full NORA3 dataset with no variable selection or regridding, as one merged
`<output>/YYYYMMDD.nc` per day.

All three scripts accept `--debug` to disable lazy (dask) loading, which can help when
diagnosing errors. `-o` defaults to a Sigma2/NIRD project path in `nora3.py` and
`nora3_merge.py` — pass your own path for local use. `olivia_nora3_roms.sbatch` is an
example Slurm batch script for running `nora3.py --use-roms` on a cluster.
