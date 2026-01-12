#!/usr/bin/env python3
"""
Process NORA3 atmospheric forcing data and save locally as daily regridded NetCDF files.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from siphon.catalog import TDSCatalog

from atm_forcing import CF_ROMS, generate_catalog_urls, get_ds, get_ds_roms  # noqa: F401

LAT_NEW = np.arange(58.9, 60, 0.02)
LON_NEW = np.arange(10.1, 11.1, 0.02)
FILE_PATH_GRID = Path.home() / "dump_fram_nn9297k" / "ROHO800_grid_fix5.nc"


def process_nora3(output_dir: Path, use_roms: bool = False, start_year: int = None, end_year: int = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = [x[0] for x in CF_ROMS]
    ds_grid = xr.open_dataset(FILE_PATH_GRID) if use_roms else None

    regridder = None
    dss = []
    timestamps = []

    for date_and_time, catalog_url in generate_catalog_urls(start_year, end_year):
        timestamp = date_and_time.strftime("%Y%m%d")
        file_path = output_dir / f"{timestamp}.nc"

        if file_path.exists():
            continue

        cat = TDSCatalog(catalog_url)
        urls = [v.access_urls["opendap"] for k, v in cat.datasets.items() if "_fp" in k]

        ds = xr.open_mfdataset(
            urls,
            combine="by_coords",
            compat="no_conflicts",
            data_vars="all",
        )
        ds = ds[parameters]

        if use_roms:
            regridder, ds = get_ds_roms(regridder, ds, ds_grid)
        else:
            regridder, ds = get_ds(regridder, ds, LAT_NEW, LON_NEW)

        dss.append(ds)
        timestamps.append(timestamp)

        # There should be 4 files per day
        if len(dss) > 3:
            assert len(set(timestamps)) <= 1
            ds_out = xr.combine_by_coords(
                dss, coords=["time"], join="outer", combine_attrs="override", compat="no_conflicts"
            )
            ds_out = ds_out.sel(time=~ds_out.get_index("time").duplicated())

            print(f"Downloading, processing, saving {file_path}")
            ds_out.to_netcdf(file_path, encoding={var: {"zlib": True, "complevel": 5} for var in ds.data_vars})

            dss = []
            timestamps = []


def main():
    parser = argparse.ArgumentParser(description="Process and regrid NORA3 atmospheric data.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path.home() / "NORA3",
        help="Output directory for NetCDF files",
    )
    parser.add_argument(
        "--use-roms",
        action="store_true",
        help="Use get_ds_roms for regridding instead of get_ds",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Start year for data processing",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="End year for data processing",
    )
    args = parser.parse_args()

    process_nora3(args.output, use_roms=args.use_roms, start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
