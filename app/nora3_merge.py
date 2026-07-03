import argparse
from collections import Counter, defaultdict
from pathlib import Path

import dask
import numpy as np
import xarray as xr

from atm_forcing import CF_ROMS, ROMS_TIME_DIMS

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
    return parser.parse_args()


def _dedup_and_check_gaps(ds: xr.Dataset, time_dim: str, label: str) -> xr.Dataset:
    index = ds.get_index(time_dim)
    duplicated = index.duplicated()
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


def main() -> None:
    dask.config.set(scheduler="threads", num_workers=4)

    args = parse_args()
    daily_dir = args.output / "daily"
    monthly_dir = args.output / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    for name in ROMS_NAMES:
        time_dim = ROMS_TIME_DIMS[name]
        files = sorted(daily_dir.glob(f"{name}_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].nc"))

        months = defaultdict(list)
        for f in files:
            date_str = f.stem.rsplit("_", 1)[-1]
            months[date_str[:6]].append(f)

        for yyyymm, month_files in sorted(months.items()):
            ds = xr.open_mfdataset(
                month_files,
                combine="by_coords",
                join="outer",
                combine_attrs="override",
                compat="no_conflicts",
                parallel=True,
                chunks={},
            )
            ds = _dedup_and_check_gaps(ds, time_dim, f"{name} {yyyymm}")

            filename = f"{name}_{yyyymm}.nc"
            ds.to_netcdf(
                monthly_dir / filename,
                encoding={name: {"zlib": True, "complevel": 5}},
            )
            print(f"{filename} saved.")


if __name__ == "__main__":
    main()
