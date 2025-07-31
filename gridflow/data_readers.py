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
    # Get CRS of line data
    with fiona.open(path, layer="power_line") as src:
        crs = src.crs
    # Get bounding box of region
    minx, miny, maxx, maxy = utils.get_bb(region.countries, crs=crs)
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
def read_borders(path, countries):
    with fiona.open(path) as src:
        filtered_features = [
            feature for feature in src
            if feature["properties"]["ISO_A3"] in countries
        ]
    # Convert to GeoDataFrame
    rez = gpd.GeoDataFrame.from_features(filtered_features, crs="EPSG:4326")
    return rez


# Functions to read a subset of a global raster
def get_country_raster(country, raster_path):
    # Get a region of a large raster corresponding to a country bounding box.
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
                 start_date="2024-01-01", end_date="2024-12-31", n=2):
    """Get hourly time series of renewable potential by zone."""
    pts = utils.get_random_points(zone, n=n)
    all_re_series = []
    for idx in range(len(pts)):
        pt = pts.iloc[[idx]]
        re = get_reninja_data(pt, start_date, end_date, type=type)
        all_re_series.append(re)
    return pd.concat(all_re_series, axis=1).mean(axis=1)

def get_reninja_data(location, start_date, end_date, api_key=None, type="pv"):
    """Query renewables time series for a given point location from renewablesninja"""
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
        
    response = requests.get(query, headers=header, verify=False)

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
    path = "config.yaml"
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data[category][key]