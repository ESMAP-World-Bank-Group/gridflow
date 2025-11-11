"""Convenience functions for testing.

A set of functionalities to test the package, including functions to
generate mock data.
"""

import pandas as pd
import numpy as np
import os
import calendar

from gridflow.utils import *
from gridflow.model import *

cc = country_code_map()

def synth_region(countries, zones_per_country=2):
    global_data_path = "data/global_datasets"
    sregion = model.region(countries, global_data_path=global_data_path)
    # Create zones
    zdf = pd.DataFrame(columns=["geometry", "country"])
    zdf["geometry"] = None
    zdf["country"] = countries * zones_per_country
    zdf = zdf.sort_values("country")

    sregion.zones = zdf
    return sregion

def synth_data(countries, path="data/test/epm_inputs_raw"):
    n = len(countries)
    countries = [cc.iso3_to_name(country) for country in countries]
    years = range(2024, 2051)
    t = len(years)

    """Create load csv files."""
    load_path = path + "/load"
    os.makedirs(load_path, exist_ok=True)

    # pDemandForecast.csv
    pvals = np.random.uniform(low=100, high=500, size=[n, t])
    peak = pd.DataFrame(data=pvals, columns=years)
    peak["type"] = "Peak"
    peak["zone"] = countries

    energy = pd.DataFrame(data=pvals*(8760/1000)*0.4, columns=years)
    energy["type"] = "Energy"
    energy["zone"] = countries

    df = pd.concat([peak, energy], ignore_index=True).sort_values(by="zone")\
           .set_index("zone")
    df.to_csv(load_path + "/pDemandForecast.csv")

    # pDemandProfile.csv
    df_list = []
    for country in countries: 
        mat = _get_annual_matrix()
        mat["zone"] = country
        df_list.append(mat)
    df = pd.concat(df_list, ignore_index=True).set_index("zone")
    df.to_csv(load_path + "/pDemandProfile.csv")


def _get_annual_matrix(leap=False):
    if leap:
        ndays = 366
        y = 2000 # Use some leap year to get monthly counts from calendar.
    else:
        ndays = 365
        y = 2001 # Some non leap year

    df = pd.DataFrame(data=np.random.uniform(low=0, high=1, size=[ndays, 24]),
                      columns=[f"t{h}" for h in range(1, 25)])
    # Month and day start empty
    df[["q", "d"]] = None, None
    start_idx = 0
    for m in range(1, 13):
        days_in_month = calendar.monthrange(y, m)[1]
        month = f"m{m}"
        days = [f"d{i+1}" for i in range(days_in_month)]
        df.loc[start_idx:start_idx+days_in_month-1, "q"] = month
        df.loc[start_idx:start_idx+days_in_month-1, "d"] = days
        start_idx += days_in_month
    return df


