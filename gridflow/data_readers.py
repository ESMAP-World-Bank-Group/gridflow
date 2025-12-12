import os
from pathlib import Path

import requests
import yaml

# Various data readers
import geopandas as gpd
import pandas as pd
import fiona

# Raster packages
import rasterio
from rasterio.windows import from_bounds
import rioxarray

# Gridflow imports
from gridflow import utils

# Reader functions for openinframaps data
def read_line_data(path, region, minkm=0):
    """Load power-line geometries that intersect the region and filter short segments."""
    # Get CRS of line data
    with fiona.open(path, layer="power_line") as src:
        crs = src.crs
    # Get bounding box of region
    minx, miny, maxx, maxy = utils.get_bb(pd.concat([region.countries, region.neighbors],
                                                    ignore_index=True), crs=crs)
    # Get lines that intersect bounding box
    linepd = gpd.read_file(path, layer="power_line",
                           bbox=(minx, miny, maxx, maxy))
    # Clean and prepare line data
    # Filter short lines
    linepd["km"] = linepd.to_crs(epsg=32633).geometry.length / 1000
    linepd = linepd[linepd["km"] > minkm]
    # Clean some columns
    cols = ["circuits", "cables"]
    linepd[cols] = linepd[cols].apply(lambda col: pd.to_numeric(col, errors='coerce').fillna(1))
    # We will use the max voltage of the line as the operating voltage
    linepd = linepd.rename(columns={"max_voltage" : "voltage"})
    linepd["voltage"] = linepd["voltage"].astype(float).fillna(0)
    return linepd.reset_index(drop=True)


# Reader functions for country borders
def read_borders(path, countries=None):
    """Load country borders for the ISO3 codes of interest or return all."""
    gdf = gpd.read_file(path)
    if countries is None:
        return gdf
    filtered = gdf[gdf["ISO_A3"].isin(countries)]
    return filtered


# Determine neighbors of countries given neighbor dataset
def get_neighbors(countries):
    return []


# Convert country to zone
def ctry_to_zone_format(countries):
    """Converts the dataframe containing countries to the format of zone data.
    This is used when adding neighboring countries into network estimation.
    """
    if len(countries) == 0:
        empty = pd.DataFrame(columns=["country", "zone_label", "geometry"])
        empty.index.name = "zone"
        return empty

    ctry_zone = countries[["ISO_A3", "geometry"]]
    ctry_zone = ctry_zone.assign(country=ctry_zone["ISO_A3"],
                                 zone_label=ctry_zone["ISO_A3"])
    ctry_zone = ctry_zone.set_index("ISO_A3")
    ctry_zone.index.name = "zone"
    return ctry_zone


# Functions to read a subset of a global raster
def get_country_raster(country, raster_path):
    """Clip a global raster down to the country bounding box and return it."""
    # Get raster crs
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    minx, miny, maxx, maxy = utils.get_bb(country, crs=raster_crs)
    ras = rioxarray.open_rasterio(raster_path, chunks=True)

    ras = ras.rio.clip_box(minx=minx, miny=miny,
                           maxx=maxx, maxy=maxy)
    ras = ras.rio.clip(country.geometry, country.crs)
    return ras

# Data queries to renewablesninja

def get_zonal_re(zone, type="pv", 
                 start_date="2024-01-01", end_date="2024-12-31", n=1):
    """Get hourly time series of renewable potential by zone."""
    pts = utils.get_random_points(zone, n=n)
    all_re_series = []
    for idx in range(len(pts)):
        pt = pts.iloc[[idx]]
        re = get_reninja_data(pt, start_date, end_date, type=type)
        all_re_series.append(re)
    return pd.concat(all_re_series, axis=1).mean(axis=1)

def get_reninja_data(location, start_date, end_date, api_key=None, type="pv", verify_ssl=True, ca_bundle=None):
    """Query renewables time series for a given point location from renewablesninja.

    Parameters
    ----------
    location : GeoDataFrame row
        Point geometry to query.
    start_date, end_date : str
        Date range (YYYY-MM-DD).
    api_key : str, optional
        Renewables.ninja API key. If None, pulled from config.yaml.
    type : {"pv", "wind"}
        What dataset to query.
    verify_ssl : bool
        Verify HTTPS certificates (recommended; defaults to True).
    ca_bundle : str, optional
        Path to a certificate bundle. If not provided, tries to use certifi.
    """
    # A whole bunch of constants for the query - doubtful users will want to change.
    dataset = "merra2"
    capacity = 1
    system_loss = 0.1
    height = 100
    tracking = 0
    tilt = 35
    azim = 180
    turbine = "Gamesa+G114+2000"
    local_time = "true"

    # Get the API key from config if not passed
    if api_key is None:
        api_key = get_config_data("renewables_ninja", "api_key")
        if api_key == "your_api_key":
            raise ValueError("Missing reninja API key in config.yaml")

    base_url = "https://www.renewables.ninja/api/data/"
    header = {"Authorization": f"Token {api_key}"}

    lat, lon = location.geometry.x.iloc[0], location.geometry.y.iloc[0]
    if type=="pv":
        query = (f"{base_url}pv?lat={lat}&lon={lon}&date_from={start_date}&date_to={end_date}"
                 f"&dataset={dataset}&capacity={capacity}&system_loss={system_loss}&tracking={tracking}"
                 f"&tilt={tilt}&azim={azim}&local_time={local_time}&format=json")
    elif type=="wind":
        query = (f"{base_url}wind?lat={lat}&lon={lon}&date_from={start_date}&date_to={end_date}&dataset={dataset}"
                 f"&capacity={capacity}&height={height}&turbine={turbine}&local_time={local_time}&format=json")
        
    verify_target = False if not verify_ssl else True
    if verify_ssl:
        if ca_bundle:
            verify_target = ca_bundle
        else:
            try:
                import certifi
                verify_target = certifi.where()
            except ImportError:
                verify_target = True  # fall back to system certs

    response = requests.get(query, headers=header, verify=verify_target)

    if response.status_code != 200:
        raise Exception(f"Error querying data at ({lat}, {lon}): {response.status_code}")
    
    data = response.json()["data"]
    # Convert to dataframe - local time as index, electricity as value
    df = pd.DataFrame.from_dict(data, orient="index")
    df["local_time"] = pd.to_datetime(df["local_time"])
    df.set_index("local_time", inplace=True)
    return df

# Read from config file
def get_config_data(category, key):
    """Read a specific key from the YAML config file."""
    path = "config.yaml"
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data[category][key]


def get_global_datasets_path(fallback="data/global_datasets"):
    """Return the global datasets folder configured in config.yaml."""
    try:
        return get_config_data("input_data", "global_datasets")
    except (FileNotFoundError, KeyError, TypeError):
        return fallback


def get_global_dataset_file_path(name, default_relative, *, root=None):
    """Return a dataset file path, preferring config overrides."""
    try:
        files = get_config_data("input_data", "files")
    except (FileNotFoundError, KeyError, TypeError):
        files = None

    if isinstance(files, dict) and name in files:
        return files[name]

    if root is None:
        root = get_global_datasets_path()
    return os.path.join(root, default_relative)


def load_background_map(name, default_path=None):
    """Load a background map defined in config.yaml (if available)."""
    try:
        backgrounds = get_config_data("input_data", "backgrounds")
    except (FileNotFoundError, KeyError, TypeError):
        backgrounds = None

    path = None
    if isinstance(backgrounds, dict):
        path = backgrounds.get(name)
    if not path:
        path = default_path

    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        return gpd.read_file(file_path)
    except Exception:
        return None
