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


def get_winds(regridder, ds, lat, lon):
    u, v = get_u_v_from_coords(ds)
    da_u = xr.DataArray(data=u, coords=ds.x_wind_10m.coords, dims=ds.x_wind_10m.dims, name="u")
    da_v = xr.DataArray(data=v, coords=ds.x_wind_10m.coords, dims=ds.x_wind_10m.dims, name="v")

    da_x_wind_10m = da_u.isel(height4=0)
    regridder, da_x_wind_10m = regrid(regridder, da_x_wind_10m, lat, lon)

    da_y_wind_10m = da_v.isel(height4=0)
    regridder, da_y_wind_10m = regrid(regridder, da_y_wind_10m, lat, lon)

    return regridder, da_x_wind_10m, da_y_wind_10m


def get_ds(regridder, ds, lat, lon):
    regridder, da_x_wind_10m, da_y_wind_10m = get_winds(regridder, ds, lat, lon)

    da_swrad_acc = ds["integral_of_surface_net_downward_shortwave_flux_wrt_time"].isel(height0=0)
    regridder, da_swrad_acc = regrid(regridder, da_swrad_acc, lat, lon)
    da_swrad = da_swrad_acc.diff(dim="time") / (60 * 60)

    da_specific_humidity = ds["specific_humidity_2m"].isel(height1=0)
    regridder, da_specific_humidity = regrid(regridder, da_specific_humidity, lat, lon)

    da_air_temperature = ds["air_temperature_2m"].isel(height1=0)
    regridder, da_air_temperature = regrid(regridder, da_air_temperature, lat, lon)
    da_air_temperature -= 273.15

    da_precipitation_acc = ds["precipitation_amount_acc"].isel(height0=0)
    regridder, da_precipitation_acc = regrid(regridder, da_precipitation_acc, lat, lon)
    da_precipitation = da_precipitation_acc.diff(dim="time") / (60 * 60)

    da_air_pressure = ds["air_pressure_at_sea_level"].isel(height_above_msl=0)
    regridder, da_air_pressure = regrid(regridder, da_air_pressure, lat, lon)

    da_lwrad_acc = ds["integral_of_surface_downwelling_longwave_flux_in_air_wrt_time"].isel(height0=0)
    regridder, da_lwrad_acc = regrid(regridder, da_lwrad_acc, lat, lon)
    da_lwrad = da_lwrad_acc.diff(dim="time") / (60 * 60)

    # da_cloud_area_fraction = ds["cloud_area_fraction"].isel(height3=0)
    # regridder, da_cloud_area_fraction = regrid(regridder, da_cloud_area_fraction)

    ds_out = xr.Dataset(
        {
            "x_wind_10m": da_x_wind_10m,
            "y_wind_10m": da_y_wind_10m,
            "swrad": da_swrad,
            "specific_humidity_2m": da_specific_humidity,
            "air_temperature_2m": da_air_temperature,
            "precipitation": da_precipitation,
            "air_pressure_at_sea_level": da_air_pressure,
            "lwrad": da_lwrad,
            # "cloud_area_fraction": da_cloud_area_fraction,
        }
    )
    time_aligned = da_swrad.time
    return regridder, ds_out.reindex(time=time_aligned).reset_coords(drop=True)


def generate_catalog_urls(start_year=2010, end_year=2020):
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
