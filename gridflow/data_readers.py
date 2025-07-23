# Various data readers
import geopandas as gpd
import pandas as pd
import fiona

# Raster packages
import rasterio
from rasterio.windows import from_bounds
import rioxarray

# Reader functions for openinframaps data
def read_line_data(path):
    # Load and prepare line data
    linepd = gpd.read_file(path, layer="power_line")
    linepd[["circuits", "cables"]] = linepd[["circuits", "cables"]].astype(float).fillna(1)
    # We will use the max voltage of the line as the operating voltage
    linepd = linepd.rename(columns={"max_voltage" : "voltage"})
    linepd["voltage"] = linepd["voltage"].astype(float).fillna(0)
    return linepd


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

    minx, miny, maxx, maxy = get_country_bb(country, raster_crs)
    ras = rioxarray.open_rasterio(raster_path, chunks=True)
    ras = ras.rio.clip_box(minx=minx, miny=miny,
                           maxx=maxx, maxy=maxy)
    ras = ras.rio.clip(country.geometry, country.crs)
    return ras
    

def get_country_bb(countries, raster_crs, buffer_degrees=0.1):
    #Get the bounding box for a country or set of countries.
    
    countries = countries.to_crs(raster_crs)
    minx, miny, maxx, maxy = countries.total_bounds

    minx -= buffer_degrees
    miny -= buffer_degrees
    maxx += buffer_degrees
    maxy += buffer_degrees

    return minx, miny, maxx, maxy