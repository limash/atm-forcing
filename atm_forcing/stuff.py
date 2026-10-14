from datetime import datetime

import cfgrib
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
ACCUMULATED_ROMS_NAMES = {"swrad", "rain", "lwrad_down"}


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
    da_u_wind_10m.attrs["long_name"] = "Eastward component of 10 metre wind"
    da_v_wind_10m.attrs["units"] = "m/s"
    da_v_wind_10m.attrs["long_name"] = "Northward component of 10 metre wind"

    da_swrad_acc = ds["integral_of_surface_net_downward_shortwave_flux_wrt_time"].isel(height0=0)
    regridder, da_swrad_acc = regrid(regridder, da_swrad_acc, lat, lon)
    da_swrad = deaccumulate(da_swrad_acc)
    da_swrad.attrs["units"] = "W/m^2"
    da_swrad.attrs["long_name"] = "Net downward surface SW radiation"

    da_specific_humidity = ds["specific_humidity_2m"].isel(height1=0)
    regridder, da_specific_humidity = regrid(regridder, da_specific_humidity, lat, lon)
    da_specific_humidity.attrs["units"] = "kg/kg"
    da_specific_humidity.attrs["long_name"] = "Screen level specific humidity"

    da_air_temperature = ds["air_temperature_2m"].isel(height1=0)
    regridder, da_air_temperature = regrid(regridder, da_air_temperature, lat, lon)
    da_air_temperature -= 273.15
    da_air_temperature.attrs["units"] = "degC"
    da_air_temperature.attrs["long_name"] = "Screen level temperature (T2M)"

    da_precipitation_acc = ds["precipitation_amount_acc"].isel(height0=0)
    regridder, da_precipitation_acc = regrid(regridder, da_precipitation_acc, lat, lon)
    da_precipitation = deaccumulate(da_precipitation_acc)
    da_precipitation.attrs["units"] = "kg/m^2/s"
    da_precipitation.attrs["long_name"] = "Precipitation flux"

    da_air_pressure = ds["air_pressure_at_sea_level"].isel(height_above_msl=0)
    regridder, da_air_pressure = regrid(regridder, da_air_pressure, lat, lon)
    da_air_pressure.attrs["units"] = "Pa"
    da_air_pressure.attrs["long_name"] = "Mean Sea Level Pressure (MSLP)"

    da_lwrad_acc = ds["integral_of_surface_downwelling_longwave_flux_in_air_wrt_time"].isel(height0=0)
    regridder, da_lwrad_acc = regrid(regridder, da_lwrad_acc, lat, lon)
    da_lwrad = deaccumulate(da_lwrad_acc)
    da_lwrad.attrs["units"] = "W/m^2"
    da_lwrad.attrs["long_name"] = "Surface LW downwelling radiation"

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
    da_x_wind_10m.attrs["long_name"] = "X-component of 10 metre wind"
    da_y_wind_10m = da_y_wind_10m.rename({"time": ROMS_TIME_DIMS["Vwind"]})
    da_y_wind_10m.attrs["units"] = "m/s"
    da_y_wind_10m.attrs["long_name"] = "Y-component of 10 metre wind"

    da_swrad_acc = ds["integral_of_surface_net_downward_shortwave_flux_wrt_time"].isel(height0=0)
    regridder, da_swrad_acc = regrid_curvilinear(regridder, da_swrad_acc, lat, lon)
    da_swrad = deaccumulate(da_swrad_acc, time_dim=ROMS_TIME_DIMS["swrad"])
    da_swrad.attrs["units"] = "W/m^2"
    da_swrad.attrs["long_name"] = "Net downward surface SW radiation"

    da_specific_humidity = ds["specific_humidity_2m"].isel(height1=0)
    regridder, da_specific_humidity = regrid_curvilinear(regridder, da_specific_humidity, lat, lon)
    da_specific_humidity = da_specific_humidity.rename({"time": ROMS_TIME_DIMS["Qair"]})
    da_specific_humidity.attrs["units"] = "kg/kg"
    da_specific_humidity.attrs["long_name"] = "Screen level specific humidity"

    da_air_temperature = ds["air_temperature_2m"].isel(height1=0)
    regridder, da_air_temperature = regrid_curvilinear(regridder, da_air_temperature, lat, lon)
    da_air_temperature -= 273.15
    da_air_temperature = da_air_temperature.rename({"time": ROMS_TIME_DIMS["Tair"]})
    da_air_temperature.attrs["units"] = "degC"
    da_air_temperature.attrs["long_name"] = "Screen level temperature (T2M)"

    da_precipitation_acc = ds["precipitation_amount_acc"].isel(height0=0)
    regridder, da_precipitation_acc = regrid_curvilinear(regridder, da_precipitation_acc, lat, lon)
    da_precipitation = deaccumulate(da_precipitation_acc, time_dim=ROMS_TIME_DIMS["rain"])
    da_precipitation.attrs["units"] = "kg/m^2/s"
    da_precipitation.attrs["long_name"] = "Precipitation flux"

    da_air_pressure = ds["air_pressure_at_sea_level"].isel(height_above_msl=0)
    regridder, da_air_pressure = regrid_curvilinear(regridder, da_air_pressure, lat, lon)
    da_air_pressure = da_air_pressure.rename({"time": ROMS_TIME_DIMS["Pair"]})
    da_air_pressure.attrs["units"] = "Pa"
    da_air_pressure.attrs["long_name"] = "Mean Sea Level Pressure (MSLP)"

    da_lwrad_acc = ds["integral_of_surface_downwelling_longwave_flux_in_air_wrt_time"].isel(height0=0)
    regridder, da_lwrad_acc = regrid_curvilinear(regridder, da_lwrad_acc, lat, lon)
    da_lwrad = deaccumulate(da_lwrad_acc, time_dim=ROMS_TIME_DIMS["lwrad_down"])
    da_lwrad.attrs["units"] = "W/m^2"
    da_lwrad.attrs["long_name"] = "Surface LW downwelling radiation"

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


CARRA_DATASET = "reanalysis-pan-carra"
CARRA_ANALYSIS_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_specific_humidity",
    "mean_sea_level_pressure",
]
# accumulated since the forecast start (J/m^2 for radiation, kg/m^2 for precipitation)
CARRA_FORECAST_VARIABLES = [
    "surface_net_solar_radiation",
    "thermal_surface_radiation_downwards",
    "total_precipitation",
]
# forecasts start at 00 and 12 UTC; leads 1..12 of both runs tile the whole day hourly
CARRA_FORECAST_LEADTIMES = [str(h) for h in range(1, 13)]


def carra_requests(days, area):
    """CDS requests (analysis, forecast) for consecutive `days` of one month. `area` is [north, west, south, east]."""

    def day(ds):
        return {"year": f"{ds[0].year}", "month": f"{ds[0].month:02d}", "day": [f"{d.day:02d}" for d in ds]}

    common = {"level_type": "single_levels", "data_format": "grib", "area": area}
    analysis = {
        **common,
        "product_type": "analysis",
        "variable": CARRA_ANALYSIS_VARIABLES,
        "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
        **day(days),
    }
    forecast = {
        **common,
        "product_type": "forecast",
        "variable": CARRA_FORECAST_VARIABLES,
        "time": ["00:00", "12:00"],
        "leadtime_hour": CARRA_FORECAST_LEADTIMES,
        **day(days),
    }
    return analysis, forecast


def open_carra_grib(path, area):
    """Open a CARRA2 GRIB file and crop it to `area` = [north, west, south, east].

    The CDS "area" request key does not subset this grid (the full 2869 x 2869 pan-Arctic field comes back, with
    longitudes spanning 0..360), so the crop happens here. Longitudes are returned in -180..180.
    """
    # cfgrib splits a file with mixed level types (10 m vs 2 m vs surface) into several datasets
    dss = cfgrib.open_datasets(str(path))
    north, west, south, east = area
    lat = dss[0].latitude.values
    lon = (dss[0].longitude.values + 180) % 360 - 180
    ys, xs = np.nonzero((lat <= north) & (lat >= south) & (lon >= west) & (lon <= east))
    window = {"y": slice(ys.min(), ys.max() + 1), "x": slice(xs.min(), xs.max() + 1)}
    dss = [d.isel(window).drop_vars(["heightAboveGround", "surface", "meanSea"], errors="ignore") for d in dss]
    dss = [d.assign_coords(longitude=(d.longitude + 180) % 360 - 180) for d in dss]
    # cfgrib builds the full-grid array of everything it is asked for before cropping (about 25 GB for a month of
    # forecasts), so load one time index at a time: each GRIB field is still decoded once. The loaded crop is a view
    # of that full-grid array, hence the copy (otherwise every chunk keeps its full-grid array alive).
    dss = [_load_per_time(d) for d in dss]
    return xr.merge(dss, compat="override", join="exact")


def _load_per_time(ds):
    if "time" not in ds.dims:
        return ds.load()
    chunks = [ds.isel(time=i).load().copy(deep=True) for i in range(ds.sizes["time"])]
    return xr.concat(chunks, dim="time", coords=["valid_time"], compat="override")


def _carra_var(ds, short_name):
    for da in ds.data_vars.values():
        if da.attrs.get("GRIB_shortName") == short_name:
            return da
    have = [d.attrs.get("GRIB_shortName") for d in ds.data_vars.values()]
    raise KeyError(f"No variable with GRIB shortName {short_name!r}, have {have}")


def _carra_instantaneous(ds, short_name):
    # analysis: one valid time per record
    da = _carra_var(ds, short_name)
    if "time" not in da.dims:
        da = da.expand_dims("time")
    times = np.atleast_1d(da.valid_time.values)
    return da.drop_vars(["valid_time", "step"], errors="ignore").assign_coords(time=times).sortby("time")


def deaccumulate_forecast(da_acc, time_dim="time_acc"):
    """Per-second rate from a (time, step, ...) accumulation since each forecast start.

    The accumulation is zero at step 0, which is not in the data, so it is prepended; each hourly interval is then
    labelled with its midpoint on `time_dim` (see deaccumulate() for why not the shared "time").
    """
    da_acc = da_acc.drop_vars("valid_time", errors="ignore")
    if "time" not in da_acc.dims:
        da_acc = da_acc.expand_dims("time")
    pieces = []
    for init in da_acc.time.values:
        da_i = da_acc.sel(time=init)
        zero = xr.zeros_like(da_i.isel(step=0)).assign_coords(step=np.timedelta64(0, "ns"))
        full = xr.concat([zero, da_i], dim="step")
        rate = full.diff(dim="step") / (60 * 60)
        valid = init + full.step.values
        rate = rate.assign_coords(step=valid[:-1] + (valid[1:] - valid[:-1]) / 2).rename({"step": time_dim})
        pieces.append(rate.drop_vars("time"))
    return xr.concat(pieces, dim=time_dim).sortby(time_dim).transpose(time_dim, ...)  # time first, as ROMS expects


def _carra_fields(regridder, ds_an, ds_fc, regrid_fn):
    """Regrid and convert the CARRA2 variables. regrid_fn(regridder, da) -> (regridder, da).

    Returns {name: da} with ROMS names, except winds, which are "u"/"v" (east/north). Instantaneous variables
    are on "time", the de-accumulated ones on "time_acc".
    """
    u, v = (_carra_instantaneous(ds_an, name) for name in ("10u", "10v"))
    # CARRA2 winds follow the polar stereographic grid: de-rotate to east/north on the source grid
    angle = lonlat_to_angle(u.longitude.values, u.latitude.values)
    u_east, v_north = rotate_u_v(-angle, u, v)

    fields = {}
    for name, da in (
        ("u", u_east),
        ("v", v_north),
        ("Qair", _carra_instantaneous(ds_an, "2sh")),
        ("Tair", _carra_instantaneous(ds_an, "2t")),
        ("Pair", _carra_instantaneous(ds_an, "msl")),
    ):
        regridder, fields[name] = regrid_fn(regridder, da)
    for name, short_name in (("swrad", "ssr"), ("lwrad_down", "strd"), ("rain", "tp")):
        regridder, da = regrid_fn(regridder, _carra_var(ds_fc, short_name))
        fields[name] = deaccumulate_forecast(da)

    fields["u"].attrs = {"units": "m/s", "long_name": "Eastward component of 10 metre wind"}
    fields["v"].attrs = {"units": "m/s", "long_name": "Northward component of 10 metre wind"}
    fields["swrad"].attrs = {"units": "W/m^2", "long_name": "Net downward surface SW radiation"}
    fields["Qair"].attrs = {"units": "kg/kg", "long_name": "Screen level specific humidity"}
    fields["Tair"] = fields["Tair"] - 273.15
    fields["Tair"].attrs = {"units": "degC", "long_name": "Screen level temperature (T2M)"}
    fields["rain"].attrs = {"units": "kg/m^2/s", "long_name": "Precipitation flux"}
    fields["Pair"].attrs = {"units": "Pa", "long_name": "Mean Sea Level Pressure (MSLP)"}
    fields["lwrad_down"].attrs = {"units": "W/m^2", "long_name": "Surface LW downwelling radiation"}
    return regridder, fields


def get_ds_carra(regridder, ds_an, ds_fc, lat, lon):
    """CARRA2 counterpart of get_ds: rectilinear lat/lon grid, one merged dataset."""
    regridder, f = _carra_fields(regridder, ds_an, ds_fc, lambda r, da: regrid(r, da, lat, lon))
    ds_out = xr.Dataset(
        {
            "u_wind_10m": f["u"],
            "v_wind_10m": f["v"],
            "swrad": f["swrad"],
            "specific_humidity_2m": f["Qair"],
            "air_temperature_2m": f["Tair"],
            "precipitation": f["rain"],
            "air_pressure_at_sea_level": f["Pair"],
            "lwrad": f["lwrad_down"],
        }
    )
    return regridder, ds_out


def get_ds_roms_carra(regridder, ds_an, ds_fc, ds_grid):
    """CARRA2 counterpart of get_ds_roms: {roms_name: da}, each on its own ROMS_TIME_DIMS time dimension."""
    lat, lon = ds_grid.lat_rho, ds_grid.lon_rho
    regridder, f = _carra_fields(regridder, ds_an, ds_fc, lambda r, da: regrid_curvilinear(r, da, lat, lon))
    f["Uwind"], f["Vwind"] = rotate_u_v(ds_grid.angle.values, f.pop("u"), f.pop("v"))
    f["Uwind"].attrs = {"units": "m/s", "long_name": "X-component of 10 metre wind"}
    f["Vwind"].attrs = {"units": "m/s", "long_name": "Y-component of 10 metre wind"}
    variables = {}
    for name, roms_time in ROMS_TIME_DIMS.items():
        da = f[name]
        old_time = "time_acc" if name in ACCUMULATED_ROMS_NAMES else "time"
        variables[name] = da.rename({old_time: roms_time}).rename(name)
    return regridder, variables


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
