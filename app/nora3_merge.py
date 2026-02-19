import argparse
from pathlib import Path

import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge NORA3 files and export each variable to its own netCDF file.")
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=Path.home() / "NORA3",
        help="Path to folder containing daily NORA3 .nc files (default: ~/NORA3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = str(args.input_folder / "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].nc")
    ds = xr.open_mfdataset(pattern, combine="by_coords")

    for var_name, da in ds.data_vars.items():
        da = da.rename(var_name)
        filename = f"{var_name}.nc"
        da.to_netcdf(
            Path.home() / "FjordSim_data" / "NORA3" / filename,
            encoding={var_name: {"zlib": True, "complevel": 5}},
        )
        print(f"{filename} saved.")


if __name__ == "__main__":
    main()