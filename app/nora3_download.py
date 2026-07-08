#!/usr/bin/env python3
"""
Download NORA3 atmospheric forcing data and save locally as daily NetCDF files.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import xarray as xr
from requests.exceptions import HTTPError
from siphon.catalog import TDSCatalog

from atm_forcing import generate_catalog_urls

REPO_URL = "https://github.com/limash/atm-forcing"
CATALOG_RETRIES = 4
CATALOG_BACKOFF_SECONDS = 5


def _open_catalog_with_retry(catalog_url):
    for attempt in range(CATALOG_RETRIES):
        try:
            return TDSCatalog(catalog_url)
        except HTTPError:
            if attempt == CATALOG_RETRIES - 1:
                raise
            wait = CATALOG_BACKOFF_SECONDS * 2**attempt
            print(f"Catalog fetch failed for {catalog_url}, retrying in {wait}s...")
            time.sleep(wait)


def _open_mfdataset_with_retry(urls, **kwargs):
    for attempt in range(CATALOG_RETRIES):
        try:
            return xr.open_mfdataset(urls, **kwargs)
        except OSError:
            if attempt == CATALOG_RETRIES - 1:
                raise
            wait = CATALOG_BACKOFF_SECONDS * 2**attempt
            print(f"Dataset open failed for {urls}, retrying in {wait}s...")
            time.sleep(wait)


def _to_netcdf_atomic(ds, path, **kwargs):
    tmp_path = path.with_name(path.name + ".tmp")
    ds.to_netcdf(tmp_path, **kwargs)
    tmp_path.rename(path)


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
        return f"{REPO_URL}/tree/{commit}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return REPO_URL


def download_nora3(output_dir: Path, start_year: int = None, end_year: int = None, debug: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    dss = []
    timestamps = []
    command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    source_code_url = _get_source_code_url()

    for date_and_time, catalog_url in generate_catalog_urls(start_year, end_year):
        timestamp = date_and_time.strftime("%Y%m%d")
        file_path = output_dir / f"{timestamp}.nc"

        if file_path.exists():
            continue

        cat = _open_catalog_with_retry(catalog_url)
        urls = [v.access_urls["opendap"] for k, v in cat.datasets.items() if "_fp" in k]

        ds = _open_mfdataset_with_retry(
            urls,
            combine="by_coords",
            compat="no_conflicts",
            data_vars="all",
        )
        if debug:
            ds = ds.load()

        dss.append(ds)
        timestamps.append(timestamp)

        # There should be 4 files per day
        if len(dss) > 3:
            assert len(set(timestamps)) <= 1
            ds_out = xr.combine_by_coords(
                dss, coords=["time"], join="outer", combine_attrs="override", compat="no_conflicts"
            )
            ds_out = ds_out.sel(time=~ds_out.get_index("time").duplicated())
            ds_out.attrs["source_code"] = source_code_url
            ds_out.attrs["command"] = command

            print(f"Downloading, saving {file_path}")
            _to_netcdf_atomic(ds_out, file_path, encoding={var: {"zlib": True, "complevel": 5} for var in ds.data_vars})

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable lazy (dask) computation and load each cycle's data eagerly",
    )
    args = parser.parse_args()

    download_nora3(args.output, start_year=args.start_year, end_year=args.end_year, debug=args.debug)


if __name__ == "__main__":
    main()
