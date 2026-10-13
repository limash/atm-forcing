#!/usr/bin/env python3
"""
Process CARRA2 (CDS reanalysis-pan-carra) atmospheric forcing data and save locally as monthly regridded NetCDF files.

Needs CDS credentials in ~/.cdsapirc and the dataset licence accepted on the CDS website.
"""

import argparse
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr
from nora3 import LAT_NEW, LON_NEW, TIME_UNITS, _get_source_code_url, _to_netcdf_atomic

from atm_forcing import (
    CARRA_DATASET,
    CF_ROMS,
    ROMS_TIME_DIMS,
    carra_requests,
    get_ds_carra,
    get_ds_roms_carra,
    open_carra_grib,
)

RETRIES = 4
BACKOFF_SECONDS = 30
AREA_PAD_DEGREES = 0.5  # source data margin around the target grid for the interpolation


def _retrieve_with_retry(request, target):
    client = cdsapi.Client()  # one per thread
    for attempt in range(RETRIES):
        try:
            client.retrieve(CARRA_DATASET, request, str(target))
            return
        except Exception as e:  # cdsapi raises plain Exception, as well as requests errors
            if attempt == RETRIES - 1:
                raise
            wait = BACKOFF_SECONDS * 2**attempt
            print(f"CDS retrieve failed ({e}), retrying in {wait}s...")
            time.sleep(wait)


def _next_month(month):
    return (month.replace(day=28) + timedelta(days=4)).replace(day=1)


def _months(start, end):
    """First days of the months from start to end (inclusive)."""
    month = start
    while month <= end:
        yield month
        month = _next_month(month)


def process_carra2(
    output_dir: Path,
    use_roms: bool = False,
    start: date = None,
    end: date = None,
    file_path_grid: Path = None,
):
    output_dir = output_dir / "monthly"
    output_dir.mkdir(parents=True, exist_ok=True)
    roms_names = [x[2] for x in CF_ROMS]
    ds_grid = xr.open_dataset(file_path_grid) if use_roms else None
    command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    source_code_url = _get_source_code_url()

    lat, lon = (ds_grid.lat_rho.values, ds_grid.lon_rho.values) if use_roms else (LAT_NEW, LON_NEW)
    area = [
        float(np.max(lat)) + AREA_PAD_DEGREES,
        float(np.min(lon)) - AREA_PAD_DEGREES,
        float(np.min(lat)) - AREA_PAD_DEGREES,
        float(np.max(lon)) + AREA_PAD_DEGREES,
    ]

    regridder = None

    # One request set per month: the CDS queue takes minutes per request, so daily requests would be far too slow.
    for month in _months(start, end):
        yyyymm = month.strftime("%Y%m")
        if use_roms:
            file_paths = {name: output_dir / f"{name}_{yyyymm}.nc" for name in roms_names}
            if all(p.exists() for p in file_paths.values()):
                continue
        else:
            file_path = output_dir / f"{yyyymm}.nc"
            if file_path.exists():
                continue

        days = [month + timedelta(days=i) for i in range((_next_month(month) - month).days)]
        # The full pan-Arctic GRIB is only needed to crop it: it lives in a temporary directory that is deleted as
        # soon as the cropped data is loaded.
        with tempfile.TemporaryDirectory(dir=output_dir, prefix=f"carra2_{yyyymm}_") as tmp:
            tmp = Path(tmp)
            # The requests queue at CDS independently, so submit them together: waiting for one after the other
            # adds up their queue times.
            requests = dict(zip(("analysis", "forecast"), carra_requests(days, area)))
            print(f"Downloading {', '.join(requests)} for {yyyymm}")
            with ThreadPoolExecutor(max_workers=len(requests)) as pool:
                futures = [
                    pool.submit(_retrieve_with_retry, req, tmp / f"{name}.grib") for name, req in requests.items()
                ]
                for future in futures:
                    future.result()
            ds_an = open_carra_grib(tmp / "analysis.grib", area)
            ds_fc = open_carra_grib(tmp / "forecast.grib", area)

        attrs = {"source_code": source_code_url, "command": command}
        if use_roms:
            regridder, variables = get_ds_roms_carra(regridder, ds_an, ds_fc, ds_grid)
            for name, da in variables.items():
                time_dim = ROMS_TIME_DIMS[name]
                ds_out = da.to_dataset()
                ds_out.attrs.update(attrs)
                # see nora3.py: must be set on the variable's encoding, not via to_netcdf's encoding=
                ds_out[name].encoding["coordinates"] = None
                print(f"Processing, saving {file_paths[name]}")
                _to_netcdf_atomic(
                    ds_out,
                    file_paths[name],
                    encoding={
                        name: {"zlib": True, "complevel": 5},
                        time_dim: {"units": TIME_UNITS, "dtype": "float64"},
                    },
                )
        else:
            regridder, ds_out = get_ds_carra(regridder, ds_an, ds_fc, LAT_NEW, LON_NEW)
            ds_out.attrs.update(attrs)
            print(f"Processing, saving {file_path}")
            _to_netcdf_atomic(
                ds_out, file_path, encoding={var: {"zlib": True, "complevel": 5} for var in ds_out.data_vars}
            )


def _year_month(text):
    return datetime.strptime(text, "%Y-%m").date()


def main():
    parser = argparse.ArgumentParser(description="Download and regrid CARRA2 atmospheric data into monthly files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/cluster/projects/nn9490k/CARRA2"),
        help="Base output directory; monthly files are written to <output>/monthly",
    )
    parser.add_argument(
        "--use-roms",
        action="store_true",
        help="Regrid onto the ROMS grid (one file per variable per month) instead of a lat/lon grid",
    )
    parser.add_argument("--start", type=_year_month, default=date(2009, 1, 1), help="First month, YYYY-MM")
    parser.add_argument("--end", type=_year_month, default=date(2023, 12, 1), help="Last month, YYYY-MM (inclusive)")
    parser.add_argument(
        "--file-path-grid",
        type=Path,
        default=Path("/cluster/projects/nn9490k/ROHO800/Grid/ROHO800_grid_v2.nc"),
        help="Path to the ROMS grid file (used with --use-roms)",
    )
    args = parser.parse_args()

    process_carra2(
        args.output,
        use_roms=args.use_roms,
        start=args.start,
        end=args.end,
        file_path_grid=args.file_path_grid,
    )


if __name__ == "__main__":
    main()
