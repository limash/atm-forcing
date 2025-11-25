#!/usr/bin/env python3
"""
Process NORA3 atmospheric forcing data and save locally as daily regridded NetCDF files.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from siphon.catalog import TDSCatalog

from atm_forcing import CF_ROMS, generate_catalog_urls, get_ds

LAT_NEW = np.arange(58.9, 60, 0.02)
LON_NEW = np.arange(10.1, 11.1, 0.02)


def process_nora3(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = [x[0] for x in CF_ROMS]

    regridder = None
    dss = []
    timestamps = []

    for date_and_time, catalog_url in generate_catalog_urls():
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

        regridder, ds = get_ds(regridder, ds, LAT_NEW, LON_NEW)

        dss.append(ds)
        timestamps.append(timestamp)

        # There should be 4 files per day
        if len(dss) > 3:
            assert len(set(timestamps)) <= 1
            ds_out = xr.combine_by_coords(dss, coords=["time"], join="outer")

            print(f"Downloading, processing, saving {file_path}")
            ds_out.to_netcdf(file_path)

            dss = []
            timestamps = []


def main():
    parser = argparse.ArgumentParser(description="Process and regrid NORA3 atmospheric data.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path.home() / "FjordSim_data" / "NORA3",
        help="Output directory for NetCDF files",
    )
    args = parser.parse_args()

    process_nora3(args.output)


if __name__ == "__main__":
    main()
