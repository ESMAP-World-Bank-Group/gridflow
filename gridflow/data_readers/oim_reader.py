# Read and prepare openinframaps data
import geopandas as gpd
import pandas as pd

def read_line_data(path):
    # Load and prepare line data
    linepd = gpd.read_file(path, layer="power_line")
    linepd[["circuits", "cables"]] = linepd[["circuits", "cables"]].astype(float).fillna(1)
    # We will use the max voltage of the line as the operating voltage
    linepd = linepd.rename(columns={"max_voltage" : "voltage"})
    linepd["voltage"] = linepd["voltage"].astype(float).fillna(0)
    return linepd
    