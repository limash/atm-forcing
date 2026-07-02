#!/usr/bin/env python3
"""
Process NORA3 atmospheric forcing data and save locally as daily regridded NetCDF files.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from siphon.catalog import TDSCatalog

from atm_forcing import CF_ROMS, ROMS_TIME_DIMS, generate_catalog_urls, get_ds, get_ds_roms  # noqa: F401

LAT_NEW = np.arange(58.9, 60, 0.02)
LON_NEW = np.arange(10.1, 11.1, 0.02)
REPO_URL = "https://github.com/limash/atm-forcing"


def _get_source_code_url():
    try:
        commit = (
            subprocess.check_output(
                ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return f"{REPO_URL}/commit/{commit}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return REPO_URL


def process_nora3(
    output_dir: Path,
    use_roms: bool = False,
    start_year: int = None,
    end_year: int = None,
    file_path_grid: Path = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = [x[0] for x in CF_ROMS]
    roms_names = [x[2] for x in CF_ROMS]
    ds_grid = xr.open_dataset(file_path_grid) if use_roms else None
    command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    source_code_url = _get_source_code_url()

    regridder = None
    dss = []
    variable_cycles = {name: [] for name in roms_names}
    timestamps = []

    for date_and_time, catalog_url in generate_catalog_urls(start_year, end_year):
        timestamp = date_and_time.strftime("%Y%m%d")

        if use_roms:
            file_paths = {name: output_dir / f"{name}_{timestamp}.nc" for name in roms_names}
            if all(p.exists() for p in file_paths.values()):
                continue
        else:
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
            regridder, cycle = get_ds_roms(regridder, ds, ds_grid)
            for name, da in cycle.items():
                variable_cycles[name].append(da)
        else:
            regridder, ds = get_ds(regridder, ds, LAT_NEW, LON_NEW)
            dss.append(ds)

        timestamps.append(timestamp)

        # There should be 4 files per day
        if len(timestamps) > 3:
            assert len(set(timestamps)) <= 1

            if use_roms:
                for name in roms_names:
                    time_dim = ROMS_TIME_DIMS[name]
                    ds_out = xr.combine_by_coords(
                        variable_cycles[name],
                        coords=[time_dim],
                        join="outer",
                        combine_attrs="override",
                        compat="no_conflicts",
                    )
                    ds_out = ds_out.sel({time_dim: ~ds_out.get_index(time_dim).duplicated()})
                    ds_out.attrs["source_code"] = source_code_url
                    ds_out.attrs["command"] = command

                    print(f"Downloading, processing, saving {file_paths[name]}")
                    ds_out.to_netcdf(file_paths[name])

                variable_cycles = {name: [] for name in roms_names}
            else:
                # "time" (instantaneous) and "time_acc" (deaccumulated) are independent
                # dimensions, so each needs its own combine_by_coords call.
                time_dss = [d[[v for v in d.data_vars if "time" in d[v].dims]] for d in dss]
                acc_dss = [d[[v for v in d.data_vars if "time_acc" in d[v].dims]] for d in dss]
                time_out = xr.combine_by_coords(
                    time_dss, coords=["time"], join="outer", combine_attrs="override", compat="no_conflicts"
                )
                acc_out = xr.combine_by_coords(
                    acc_dss, coords=["time_acc"], join="outer", combine_attrs="override", compat="no_conflicts"
                )
                time_out = time_out.sel(time=~time_out.get_index("time").duplicated())
                acc_out = acc_out.sel(time_acc=~acc_out.get_index("time_acc").duplicated())
                ds_out = xr.merge([time_out, acc_out], combine_attrs="override")
                ds_out.attrs["source_code"] = source_code_url
                ds_out.attrs["command"] = command

                print(f"Downloading, processing, saving {file_path}")
                ds_out.to_netcdf(file_path, encoding={var: {"zlib": True, "complevel": 5} for var in ds_out.data_vars})

                dss = []

            timestamps = []


def main():
    parser = argparse.ArgumentParser(description="Process and regrid NORA3 atmospheric data.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/cluster/projects/nn9490k/NORA3/daily"),
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
        default=2009,
        help="Start year for data processing",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="End year for data processing",
    )
    parser.add_argument(
        "--file-path-grid",
        type=Path,
        default=Path("/cluster/projects/nn9490k/ROHO800/Grid/ROHO800_grid_v2.nc"),
        help="Path to the ROMS grid file (used with --use-roms)",
    )
    args = parser.parse_args()

    process_nora3(
        args.output,
        use_roms=args.use_roms,
        start_year=args.start_year,
        end_year=args.end_year,
        file_path_grid=args.file_path_grid,
    )


if __name__ == "__main__":
    main()
