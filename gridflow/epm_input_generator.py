"""EPM Input Data Generation Module.

This module contains functions to process input csv's into
a set of ouputs that are ready for EPM and reflect the flow model 
structure created by gridflow. The intention is for the input
and (necessarily) output files to match the structure of EPM inputs as
closely as possible. In some cases, additional data will need to be added to a file
to support gridflow processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os

from gridflow.utils import *

cc = country_code_map()

def generate_epm_inputs(region, input_base_dir, output_base_dir, verbose=False):
    """Master function to process all EPM input files.
    
    Parameters
    ----------
    region : gridflow.model.region
        Region object containing the zone definitions used for expansion
    input_base_dir : str or Path
        Base directory containing subdirectories with input files
    output_base_dir : str or Path
        Base directory where processed files will be saved
    verbose : bool
        If True, print detailed processing information
        
    Returns
    -------
    None
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)

    toprocess = {
        # Hourly demand profile is assumed identical for every zone carved out of a country
        "pDemandProfile": {
            "path": "load/pDemandProfile.csv",
            "func": "zone_replicate",
            "args": []
        },
        # Annual demand forecast must be split across zones according to the selected scaling metric
        "pDemandForecast": {
            "path": "load/pDemandForecast.csv",
            "func": "zone_distribute",
            "args": [["zone", "type", "country"]]
        },
        # Raw chronological demand traces are duplicated so each zone inherits its country's series
        "pDemandData": {
            "path": "load/pDemandData.csv",
            "func": "zone_replicate",
            "args": []
        },
        # Technology-specific availability assumptions are copied to every zone within the same country
        "pAvailabilityDefault": {
            "path": "supply/pAvailabilityDefault.csv",
            "func": "zone_replicate",
            "args": []
        },
        # Default generator parameter table is cloned for each zone to preserve consistent tech data
        "pGenDataInputDefault": {
            "path": "supply/pGenDataInputDefault.csv",
            "func": "zone_replicate",
            "args": []
        },
        # Capital expenditure trajectories stay uniform inside a country, so replicate per zone
        "pCapexTrajectoriesDefault": {
            "path": "supply/pCapexTrajectoriesDefault.csv",
            "func": "zone_replicate",
            "args": []
        },
        # Fuel price assumptions are defined at the country level and copied to all constituent zones
        "pFuelPrice": {
            "path": "supply/pFuelPrice.csv",
            "func": "zone_replicate",
            "args": []
        },
        # VRE profiles are replicated so each zone created within a country shares the same profile
        "pVREProfile": {
            "path": "supply/pVREProfile.csv",
            "func": "zone_replicate",
            "args": []
        },
        # Fuel use caps are imposed per country, so duplicate them to every derived zone
        "pMaxFuellimit": {
            "path": "constraint/pMaxFuellimit.csv",
            "func": "zone_replicate",
            "args": []
        }
    }

    processed_rel_paths = set()

    for a, b in toprocess.items():
        if verbose:
            print(f"Processing {a}")
        rel_path = Path(b["path"])
        processed_rel_paths.add(rel_path.as_posix())
        in_path = input_base_dir / rel_path
        out_path = output_base_dir / rel_path
        globals()[b["func"]](region, str(in_path), str(out_path), *b["args"], verbose=verbose)

    _copy_remaining_inputs(input_base_dir, output_base_dir, processed_rel_paths)
    return None

def noop(region, input_path, output_path, verbose=False):
    """Do nothing."""
    shutil.copyfile(input_path, output_path)

def zone_replicate(region, input_path, output_path, verbose=False):
    """Replicate input data for each zone.

    Takes a CSV file with a 'zone' column and replicates the data
    for each zone defined in the model.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Check if subregions have been created
    if region.zones.empty:
        raise ValueError("The region contains no zones.")
    
    # Get zone identifiers
    zone_ids = region.zones.index.tolist()
    n_zones = len(zone_ids)
    
    # Read the original CSV
    df_original = pd.read_csv(input_path)
 

    if 'zone' in df_original.columns:
        # Replicate data for each subregion
        df_final = pd.concat([
            df_original.assign(zone=zone_id) 
            for zone_id in zone_ids
        ], ignore_index=True)
        
        if verbose:
            print(f"original shape: {df_original.shape}, final shape: {df_final.shape}, {n_zones} zones.")
    else:
        raise ValueError("The input data is not zonal.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path)


def zone_distribute(region, input_path, output_path, exclude_cols=[], 
	                scaleby="population", verbose=False):
    """Proportionally distribute input data to each zone. 

    Takes a CSV file with a 'zone' column and distributes the specified
    time series quantities to each zone proportionally.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Check if subregions have been created
    if region.zones.empty:
        raise ValueError("The region contains no zones.")
    
    # Get subregion identifiers
    zone_ids = region.zones.index.tolist()
    n_zones = len(zone_ids)
    # Get the scalars for each subregion
    scale = region.zones[scaleby] / np.sum(region.zones[scaleby])
    
    # Read the original CSV
    df_original = pd.read_csv(input_path)
 

    if 'zone' in df_original.columns:
        # Replicate and scale data for each subregion
        df_final = []
        for z in zone_ids:
            country = cc.iso3_to_name(region.zones.loc[z, "country"])
            df = df_original[df_original["country"]==country].assign(zone=z)
            df[df.columns.difference(exclude_cols)] *= scale.loc[z]
            df_final.append(df)
        df_final = pd.concat(df_final, ignore_index=True)
        if verbose:
            print(f"original shape: {df_original.shape}, final shape: {df_final.shape}, {n_zones} zones.")
    else:
        raise ValueError("The input data is not zonal.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path)


def _copy_remaining_inputs(input_base_dir, output_base_dir, handled_rel_paths):
    """Copy inputs that do not require zonal processing and guard against omissions."""
    for src in input_base_dir.rglob("*.csv"):
        rel_path = src.relative_to(input_base_dir).as_posix()
        if rel_path in handled_rel_paths:
            continue

        try:
            columns = pd.read_csv(src, nrows=0).columns
        except pd.errors.EmptyDataError:
            columns = []
        lower_cols = {str(col).strip().lower() for col in columns}
        if {"zone", "z"} & lower_cols:
            raise ValueError(
                f"File '{rel_path}' contains zonal data but is not explicitly processed."
            )

        destination = output_base_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
