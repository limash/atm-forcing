#!/usr/bin/env python3
"""
Download NORA3 atmospheric forcing data and save locally as daily NetCDF files.
"""

import argparse
from pathlib import Path

import xarray as xr
from siphon.catalog import TDSCatalog

from atm_forcing import generate_catalog_urls


def download_nora3(output_dir: Path, start_year: int = None, end_year: int = None):
    output_dir.mkdir(parents=True, exist_ok=True)
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

        dss.append(ds)
        timestamps.append(timestamp)

        # There should be 4 files per day
        if len(dss) > 3:
            assert len(set(timestamps)) <= 1
            ds_out = xr.combine_by_coords(
                dss, coords=["time"], join="outer", combine_attrs="override", compat="no_conflicts"
            )
            ds_out = ds_out.sel(time=~ds_out.get_index("time").duplicated())

            print(f"Downloading, saving {file_path}")
            ds_out.to_netcdf(file_path, encoding={var: {"zlib": True, "complevel": 5} for var in ds.data_vars})

            dss = []
            timestamps = []


def main():
    parser = argparse.ArgumentParser(description="Download NORA3 atmospheric data and save as daily NetCDF files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path.home() / "NORA3_all_data",
        help="Output directory for NetCDF files",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Start year for data download",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for data download",
    )
    args = parser.parse_args()

    download_nora3(args.output, start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
