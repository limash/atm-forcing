from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

CF_ROMS = (
    ("x_wind_10m", "height4", "Uwind", "wind_time"),
    ("y_wind_10m", "height4", "Vwind", "wind_time"),
    ("integral_of_surface_net_downward_shortwave_flux_wrt_time", "height0", "swrad", "swrad_time"),  # accumulated
    ("specific_humidity_2m", "height1", "Qair", "qair_time"),
    ("air_temperature_2m", "height1", "Tair", "Tair_time"),  # Kelvin -> to Celsius
    ("precipitation_amount_acc", "height0", "rain", "rain_time"),  # accumulated
    ("air_pressure_at_sea_level", "height_above_msl", "Pair", "pair_time"),
    (
        "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time",
        "height0",
        "lwrad_down",
        "lwrad_time",
    ),  # accumulated; units - 1 watt = 1 joule per second.
    # ("cloud_area_fraction", "height3", "cloud", "cloud_time"),
)

ROMS_TIME_DIMS = {roms_name: roms_time_name for _, _, roms_name, roms_time_name in CF_ROMS}


def lonlat_to_angle(lon, lat):
    # this returns angle from east to the current x
    diff_lon = np.diff(lon, axis=1)
    diff_lon = np.hstack([diff_lon, diff_lon[:, -1:]])
    diff_lat = np.diff(lat, axis=1)
    diff_lat = np.hstack([diff_lat, diff_lat[:, -1:]])
    assert np.all(np.abs(diff_lon) < 180)
    diff_lon *= np.cos(np.deg2rad(lat))
    return np.arctan2(diff_lat, diff_lon)


def rotate_u_v(angle, u_east, v_north):
    # rotate in the direction of angle
    cos_alpha = np.cos(angle)
    sin_alpha = np.sin(angle)
    u_x = u_east * cos_alpha + v_north * sin_alpha
    v_y = v_north * cos_alpha - u_east * sin_alpha
    return u_x, v_y


def get_u_v_from_coords(ds):
    angle = lonlat_to_angle(ds.longitude.values, ds.latitude.values)
    # rotate in the opposite direction
    u, v = rotate_u_v(-angle, ds.x_wind_10m, ds.y_wind_10m)
    return u, v


def wind_direction_transform(da):
    # blows from to blows to
    da += 180
    da = xr.where(da >= 360, da - 360, da)
    # clockwise to unticlockwise
    da = -1 * da + 360
    # rotate so east is pos x and north is pos y
    da += 90
    da = xr.where(da >= 360, da - 360, da)
    return (np.pi / 180) * da


def get_u_v_from_direction(ds):
    da_wd = wind_direction_transform(ds.wind_direction.copy(deep=True))
    u = ds.wind_speed * np.cos(da_wd)
    v = ds.wind_speed * np.sin(da_wd)
    return u, v


def regrid(regridder, da, lat, lon):
    if regridder is None:
        target_grid = xr.Dataset({"lat": (["lat"], lat), "lon": (["lon"], lon)})
        source_grid = xr.Dataset(
            {
                "lat": (("y", "x"), da.latitude.data),
                "lon": (("y", "x"), da.longitude.data),
            }
        )
        regridder = xe.Regridder(source_grid, target_grid, method="bilinear", unmapped_to_nan=True)
    return regridder, regridder(da)


def regrid_curvilinear(regridder, da, target_lat, target_lon):
    """
    Regrid a DataArray onto a curvilinear target grid.

    Parameters
    ----------
    regridder : xe.Regridder or None
        Existing regridder (for reuse), or None to create a new one.
    da : xarray.DataArray
        Source data with curvilinear coordinates.
        Must have 2D latitude/longitude coordinates.
    target_lon : xarray.DataArray
        2D longitude of target grid (e.g., lon_rho).
    target_lat : xarray.DataArray
        2D latitude of target grid (e.g., lat_rho).

    Returns
    -------
    regridder : xe.Regridder
    da_out : xarray.DataArray
        Regridded data.
    """

    if regridder is None:
        # --- Source grid (from data) ---
        source_grid = xr.Dataset(
            {
                "lon": (da.longitude.dims, da.longitude.data),
                "lat": (da.latitude.dims, da.latitude.data),
            }
        )

        # --- Target grid (curvilinear) ---
        target_grid = xr.Dataset(
            {
                "lon": (target_lon.dims, target_lon.data),
                "lat": (target_lat.dims, target_lat.data),
            }
        )

        regridder = xe.Regridder(
            source_grid,
            target_grid,
            method="bilinear",
            unmapped_to_nan=True,
            reuse_weights=False,
        )

    da_out = regridder(da)
    return regridder, da_out


def get_winds(ds):
    u, v = get_u_v_from_coords(ds)
    da_u = xr.DataArray(data=u, coords=ds.x_wind_10m.coords, dims=ds.x_wind_10m.dims, name="u")
    da_v = xr.DataArray(data=v, coords=ds.x_wind_10m.coords, dims=ds.x_wind_10m.dims, name="v")
    da_u_wind_10m = da_u.isel(height4=0)
    da_v_wind_10m = da_v.isel(height4=0)
    return da_u_wind_10m, da_v_wind_10m


def midpoint_time(time):
    # the interval midpoint, for relabeling a diffed variable's end-of-interval timestamps.
    return time.values[:-1] + (time.values[1:] - time.values[:-1]) / 2


def deaccumulate(da_acc, time_dim="time_acc"):
    # relabeled to the interval midpoint and moved off the shared "time" dim so it
    # doesn't get force-aligned with instantaneous variables still on endpoint labels.
    da = da_acc.diff(dim="time") / (60 * 60)
    da = da.assign_coords(time=midpoint_time(da_acc.time))
    return da.rename({"time": time_dim})


def get_ds(regridder, ds, lat, lon):
    da_u_wind_10m, da_v_wind_10m = get_winds(ds)
    regridder, da_u_wind_10m = regrid(regridder, da_u_wind_10m, lat, lon)
    regridder, da_v_wind_10m = regrid(regridder, da_v_wind_10m, lat, lon)
    da_u_wind_10m.attrs["units"] = "m/s"
    da_v_wind_10m.attrs["units"] = "m/s"

    da_swrad_acc = ds["integral_of_surface_net_downward_shortwave_flux_wrt_time"].isel(height0=0)
    regridder, da_swrad_acc = regrid(regridder, da_swrad_acc, lat, lon)
    da_swrad = deaccumulate(da_swrad_acc)
    da_swrad.attrs["units"] = "W/m^2"

    da_specific_humidity = ds["specific_humidity_2m"].isel(height1=0)
    regridder, da_specific_humidity = regrid(regridder, da_specific_humidity, lat, lon)
    da_specific_humidity.attrs["units"] = "kg/kg"

    da_air_temperature = ds["air_temperature_2m"].isel(height1=0)
    regridder, da_air_temperature = regrid(regridder, da_air_temperature, lat, lon)
    da_air_temperature -= 273.15
    da_air_temperature.attrs["units"] = "degC"

    da_precipitation_acc = ds["precipitation_amount_acc"].isel(height0=0)
    regridder, da_precipitation_acc = regrid(regridder, da_precipitation_acc, lat, lon)
    da_precipitation = deaccumulate(da_precipitation_acc)
    da_precipitation.attrs["units"] = "kg/m^2/s"

    da_air_pressure = ds["air_pressure_at_sea_level"].isel(height_above_msl=0)
    regridder, da_air_pressure = regrid(regridder, da_air_pressure, lat, lon)
    da_air_pressure.attrs["units"] = "Pa"

    da_lwrad_acc = ds["integral_of_surface_downwelling_longwave_flux_in_air_wrt_time"].isel(height0=0)
    regridder, da_lwrad_acc = regrid(regridder, da_lwrad_acc, lat, lon)
    da_lwrad = deaccumulate(da_lwrad_acc)
    da_lwrad.attrs["units"] = "W/m^2"

    # da_cloud_area_fraction = ds["cloud_area_fraction"].isel(height3=0)
    # regridder, da_cloud_area_fraction = regrid(regridder, da_cloud_area_fraction)

    ds_out = xr.Dataset(
        {
            "u_wind_10m": da_u_wind_10m,
            "v_wind_10m": da_v_wind_10m,
            "swrad": da_swrad,
            "specific_humidity_2m": da_specific_humidity,
            "air_temperature_2m": da_air_temperature,
            "precipitation": da_precipitation,
            "air_pressure_at_sea_level": da_air_pressure,
            "lwrad": da_lwrad,
            # "cloud_area_fraction": da_cloud_area_fraction,
        }
    )
    return regridder, ds_out


def get_ds_roms(regridder, ds, ds_grid):
    """Regrid onto the ROMS grid and return {roms_name: da}, one DataArray per
    variable, each on its own ROMS_TIME_DIMS-named time dimension."""
    lat, lon = ds_grid.lat_rho, ds_grid.lon_rho
    da_u_wind_10m, da_v_wind_10m = get_winds(ds)
    regridder, da_u_wind_10m = regrid_curvilinear(regridder, da_u_wind_10m, lat, lon)
    regridder, da_v_wind_10m = regrid_curvilinear(regridder, da_v_wind_10m, lat, lon)
    # assuming that angle is from east to x (but it is called 'is between x and east')
    da_x_wind_10m, da_y_wind_10m = rotate_u_v(ds_grid.angle.values, da_u_wind_10m, da_v_wind_10m)
    da_x_wind_10m = da_x_wind_10m.rename({"time": ROMS_TIME_DIMS["Uwind"]})
    da_x_wind_10m.attrs["units"] = "m/s"
    da_y_wind_10m = da_y_wind_10m.rename({"time": ROMS_TIME_DIMS["Vwind"]})
    da_y_wind_10m.attrs["units"] = "m/s"

    da_swrad_acc = ds["integral_of_surface_net_downward_shortwave_flux_wrt_time"].isel(height0=0)
    regridder, da_swrad_acc = regrid_curvilinear(regridder, da_swrad_acc, lat, lon)
    da_swrad = deaccumulate(da_swrad_acc, time_dim=ROMS_TIME_DIMS["swrad"])
    da_swrad.attrs["units"] = "W/m^2"

    da_specific_humidity = ds["specific_humidity_2m"].isel(height1=0)
    regridder, da_specific_humidity = regrid_curvilinear(regridder, da_specific_humidity, lat, lon)
    # ROMS bulk_flux.F reads Qair as RH and treats values < 2.0 as fractional relative
    # humidity; kg/kg specific humidity is always < 2.0 and would be misread. Convert to
    # g/kg (typically 1-20) so ROMS's autodetection takes the specific-humidity branch.
    da_specific_humidity = da_specific_humidity * 1000.0
    da_specific_humidity = da_specific_humidity.rename({"time": ROMS_TIME_DIMS["Qair"]})
    da_specific_humidity.attrs["units"] = "g/kg"

    da_air_temperature = ds["air_temperature_2m"].isel(height1=0)
    regridder, da_air_temperature = regrid_curvilinear(regridder, da_air_temperature, lat, lon)
    da_air_temperature -= 273.15
    da_air_temperature = da_air_temperature.rename({"time": ROMS_TIME_DIMS["Tair"]})
    da_air_temperature.attrs["units"] = "degC"

    da_precipitation_acc = ds["precipitation_amount_acc"].isel(height0=0)
    regridder, da_precipitation_acc = regrid_curvilinear(regridder, da_precipitation_acc, lat, lon)
    da_precipitation = deaccumulate(da_precipitation_acc, time_dim=ROMS_TIME_DIMS["rain"])
    da_precipitation.attrs["units"] = "kg/m^2/s"

    da_air_pressure = ds["air_pressure_at_sea_level"].isel(height_above_msl=0)
    regridder, da_air_pressure = regrid_curvilinear(regridder, da_air_pressure, lat, lon)
    # ROMS expects Pair in millibar; NORA3 provides Pa.
    da_air_pressure = da_air_pressure / 100.0
    da_air_pressure = da_air_pressure.rename({"time": ROMS_TIME_DIMS["Pair"]})
    da_air_pressure.attrs["units"] = "millibar"

    da_lwrad_acc = ds["integral_of_surface_downwelling_longwave_flux_in_air_wrt_time"].isel(height0=0)
    regridder, da_lwrad_acc = regrid_curvilinear(regridder, da_lwrad_acc, lat, lon)
    da_lwrad = deaccumulate(da_lwrad_acc, time_dim=ROMS_TIME_DIMS["lwrad_down"])
    da_lwrad.attrs["units"] = "W/m^2"

    variables = {
        "Uwind": da_x_wind_10m,
        "Vwind": da_y_wind_10m,
        "swrad": da_swrad,
        "Qair": da_specific_humidity,
        "Tair": da_air_temperature,
        "rain": da_precipitation,
        "Pair": da_air_pressure,
        "lwrad_down": da_lwrad,
    }
    return regridder, {name: da.rename(name) for name, da in variables.items()}


def generate_catalog_urls(start_year, end_year):
    hours = 0, 6, 12, 18
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    datetime(year, month, day)  # validate date
                except ValueError:
                    continue
                for hour in hours:
                    yield (
                        datetime(year, month, day, hour),
                        f"https://thredds.met.no/thredds/catalog/nora3/{year}/{month:02d}/{day:02d}/{hour:02d}/catalog.xml",
                    )


def reshape_to_full_year(ds, start="2020-01-01 00:00:00", end="2021-01-01 00:00:00", dim="time"):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    tmin = pd.to_datetime(ds[dim].min().data)
    dt = pd.to_datetime(ds[dim][1].data) - pd.to_datetime(ds[dim][0].data)
    n_before = int((tmin - start) / dt)
    new_coord = start + dt * np.arange(n_before)
    first = ds.isel(time=0)
    pad = xr.concat([first] * n_before, dim=dim)
    pad = pad.assign_coords({dim: new_coord})
    ds = xr.concat([pad, ds], dim=dim)
    return ds.sel(time=slice(None, end))
