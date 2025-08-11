"""Convenience functions for testing.

A set of functionalities to test the package, including functions to
generate mock data.
"""

import pandas as pd
import numpy as np
import os

from gridflow.utils import *

cc = country_code_map()

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
    return df



