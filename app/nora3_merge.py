import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import dask
import numpy as np
import xarray as xr

from atm_forcing import ACCUMULATED_ROMS_NAMES, CF_ROMS, ROMS_TIME_DIMS

ROMS_NAMES = [x[2] for x in CF_ROMS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge daily NORA3 ROMS forcing files (from --use-roms) into monthly files, one per variable."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/cluster/projects/nn9490k/NORA3"),
        help="Base output directory; daily files are read from <output>/daily, monthly files written to <output>/monthly.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to process. If omitted, all available years are processed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(len(ROMS_NAMES), os.cpu_count() or 1),
        help="Number of variables to merge in parallel, each in its own process (default: %(default)s).",
    )
    return parser.parse_args()


def _dedup_and_check_gaps(ds: xr.Dataset, time_dim: str, label: str) -> xr.Dataset:
    index = ds.get_index(time_dim)
    duplicated = index.duplicated(keep="last")
    if duplicated.any():
        print(f"{label}: removing {duplicated.sum()} duplicate {time_dim} entries.")
        ds = ds.sel({time_dim: ~duplicated})
        index = ds.get_index(time_dim)

    diffs = np.diff(index.values)
    if len(diffs) == 0:
        return ds
    step = Counter(diffs).most_common(1)[0][0]
    for i in np.where(diffs != step)[0]:
        print(f"{label}: gap at {index[i]} -> {index[i + 1]} ({diffs[i]} instead of {step}).")

    return ds


def process_variable(name: str, daily_dir: Path, monthly_dir: Path, year: int | None) -> None:
    dask.config.set(scheduler="synchronous")

    time_dim = ROMS_TIME_DIMS[name]
    year_glob = f"{year}[0-9][0-9]" if year else "[0-9][0-9][0-9][0-9][0-9][0-9]"
    pattern = f"{name}_{year_glob}[0-9][0-9].nc"
    files = sorted(daily_dir.glob(pattern))

    months = defaultdict(list)
    for f in files:
        date_str = f.stem.rsplit("_", 1)[-1]
        months[date_str[:6]].append(f)

    for yyyymm, month_files in sorted(months.items()):
        filename = f"{name}_{yyyymm}.nc"
        if (monthly_dir / filename).exists():
            print(f"{filename} already exists, skipping.")
            continue

        ds = xr.open_mfdataset(
            month_files,
            combine="by_coords",
            join="outer",
            combine_attrs="override",
            compat="no_conflicts",
            chunks={},
        )
        ds = _dedup_and_check_gaps(ds, time_dim, f"{name} {yyyymm}")
        if name not in ACCUMULATED_ROMS_NAMES:
            ds = ds.isel({time_dim: slice(None, -1)})

        ds.to_netcdf(
            monthly_dir / filename,
            encoding={name: {"zlib": True, "complevel": 5}},
        )
        print(f"{filename} saved.")


def main() -> None:
    args = parse_args()
    daily_dir = args.output / "daily"
    monthly_dir = args.output / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(process_variable, ROMS_NAMES, repeat(daily_dir), repeat(monthly_dir), repeat(args.year)))


if __name__ == "__main__":
    main()
