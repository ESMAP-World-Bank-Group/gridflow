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

def generate_epm_inputs(
    region,
    input_base_dir,
    output_base_dir,
    demand_scaleby="population",
    verbose=False,
):
    """Master function to process all EPM input files.

    Parameters
    ----------
    region : gridflow.model.region
        Region object containing the zone definitions used for expansion
    input_base_dir : str or Path
        Base directory containing subdirectories with input files
    output_base_dir : str or Path
        Base directory where processed files will be saved
    demand_scaleby : str, optional
        Column in ``region.zone_stats`` used to distribute country-level demand
        forecasts across zones (e.g., "population"). Must be loaded via
        ``zone_stats_to_load`` when creating the region.
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
            "func": zone_replicate,
            "args": []
        },
        # Annual demand forecast must be split across zones according to the selected scaling metric
        "pDemandForecast": {
            "path": "load/pDemandForecast.csv",
            "func": zone_distribute,
            "args": [["zone", "type", "country"], demand_scaleby]
        },
        # Raw chronological demand traces are duplicated so each zone inherits its country's series
        "pDemandData": {
            "path": "load/pDemandData.csv",
            "func": zone_replicate,
            "args": []
        },
        # Technology-specific availability assumptions are copied to every zone within the same country
        "pAvailabilityDefault": {
            "path": "supply/pAvailabilityDefault.csv",
            "func": zone_replicate,
            "args": []
        },
        # Default generator parameter table is cloned for each zone to preserve consistent tech data
        "pGenDataInputDefault": {
            "path": "supply/pGenDataInputDefault.csv",
            "func": zone_replicate,
            "args": []
        },
        # Capital expenditure trajectories stay uniform inside a country, so replicate per zone
        "pCapexTrajectoriesDefault": {
            "path": "supply/pCapexTrajectoriesDefault.csv",
            "func": zone_replicate,
            "args": []
        },
        # Fuel price assumptions are defined at the country level and copied to all constituent zones
        "pFuelPrice": {
            "path": "supply/pFuelPrice.csv",
            "func": zone_replicate,
            "args": []
        },
        # VRE profiles are replicated so each zone created within a country shares the same profile
        "pVREProfile": {
            "path": "supply/pVREProfile.csv",
            "func": zone_replicate,
            "args": []
        },
        # Fuel use caps are imposed per country, so duplicate them to every derived zone
        "pMaxFuellimit": {
            "path": "constraint/pMaxFuellimit.csv",
            "func": zone_replicate,
            "args": []
        }
    }

    processed_rel_paths = set()

    for param_name, param in toprocess.items():
        if verbose:
            print(f"Processing {param_name}")
        rel_path = Path(param["path"])
        processed_rel_paths.add(rel_path.as_posix())
        in_path = input_base_dir / rel_path
        out_path = output_base_dir / rel_path
        func = param["func"]
        func(region, str(in_path), str(out_path), *param["args"], verbose=verbose)

    _copy_remaining_inputs(
        input_base_dir, output_base_dir, processed_rel_paths, verbose=verbose
    )
    return None

def noop(region, input_path, output_path, verbose=False):
    """Do nothing."""
    shutil.copyfile(input_path, output_path)

def zone_replicate(region, input_path, output_path,
                   originally_countries=False, verbose=True):
    """Replicate input data for each zone.

    Takes a CSV file with a 'zone' column and replicates the data
    for each zone defined in the model.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    df_epm_original = pd.read_csv(input_path)
    
    # Check if zones have been created
    if region.zones.empty:
        raise ValueError("The region contains no zones.")
    
    if originally_countries:
        # Inputs are country specific, with country
        # specified in the "zone" column
        countries = df_epm_original.zone.unique()
        # Add a "country" column to the original dataframe
        df_epm_original["country"] = df_epm_original["zone"]

        # Iterate through countries, get rows for country, and replicate
        # rows by number of zones for the country
        new_rows = []
        for country in countries:
            # gridflow zone ids for this country
            zone_ids = region.zones[region.zones.country == country].index.tolist()
            # rows for this country
            country_rows = df_epm_original[df_epm_original.zone == country]
            # replicate rows by number of country zones
            new_rows.extend([
                country_rows.assign(zone=zone_id)
                for zone_id in zone_ids])

        df_final = pd.concat(new_rows, ignore_index=True)
    else:
        # Get zone identifiers
        zone_ids = region.zones.index.tolist()
        n_zones = len(zone_ids)
        
        # Read the original CSV
        if verbose:
            print(f"[zone_replicate] Reading input file: {input_path}")
            
        df_epm_original = pd.read_csv(input_path)
    
        if 'zone' in df_epm_original.columns:
            # Replicate data for each zone
            df_final = pd.concat([
                df_epm_original.assign(zone=zone_id) 
                for zone_id in zone_ids
            ], ignore_index=True)
            
            if verbose:
                print(f"original shape: {df_epm_original.shape}, final shape: {df_final.shape}, {n_zones} zones.")
        else:
            raise ValueError("The input data is not zonal.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path)


def zone_distribute(region, input_path, output_path, exclude_cols=None,
                    scaleby="population", verbose=True):
    """Proportionally distribute input data to each zone. 

    Takes a CSV file with a 'zone' column and distributes the specified
    time series quantities to each zone proportionally.
    """

    if exclude_cols is None:
        exclude_cols = []

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Check if subregions have been created
    if region.zone_stats.empty:
        raise ValueError("The region contains no zones.")

    if scaleby not in region.zone_stats.columns:
        loaded_stats = list(region.zone_stats.columns)
        configured_stats = list(getattr(region, "zone_stat_specs", {}).keys())
        raise ValueError(
            "Scaling metric '{0}' not found in region.zone_stats. "
            "This metric is used to distribute country-level values across zones. "
            "Ensure it is loaded by adding '{0}' to 'zone_stats_to_load' when "
            "creating the region and running region.set_zone_data before calling "
            "generate_epm_inputs. Loaded stats: {1}. Configured stats: {2}.".format(
                scaleby, loaded_stats, configured_stats
            )
        )
    
    # Ensure zone identifiers are consistent with region.zones ordering
    zone_ids = list(region.zones.index)
    missing_stats = set(zone_ids) - set(region.zone_stats.index)
    if missing_stats:
        raise ValueError(
            f"Zone stats are missing for zone ids {sorted(missing_stats)}. "
            "Run region.set_zone_data after creating zones."
        )

    n_zones = len(zone_ids)
    if n_zones == 0:
        raise ValueError("No zones found on the region object.")

    denom = np.sum(region.zone_stats.loc[zone_ids, scaleby])
    if denom <= 0:
        raise ValueError(
            f"Scaling metric '{scaleby}' sums to zero or is empty; cannot distribute demand."
        )
    scale = region.zone_stats.loc[zone_ids, scaleby] / denom
    
    # Read the original CSV
    if verbose:
        print(f"[zone_distribute] Reading input file: {input_path}")
    
    df_epm_original = pd.read_csv(input_path)

    if "zone" not in df_epm_original.columns:
        raise ValueError(
            "The input data must include a 'zone' column; matching by country is not supported."
        )

    df_final = []
    for z in zone_ids:
        df = df_epm_original[df_epm_original["zone"].astype(str) == str(z)].copy()
        df["zone"] = z
        if df.empty:
            raise ValueError(
                f"No rows found in input for zone {z}; ensure 'zone' column labels align with region.zones."
            )

        # Restrict scaling to numeric columns not explicitly excluded
        excluded = set(exclude_cols) | {"zone"}
        numeric_cols = [
            c for c in df.columns.difference(excluded)
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        df[numeric_cols] = df[numeric_cols].mul(scale.loc[z])
        df_final.append(df)
    df_final = pd.concat(df_final, ignore_index=True)
    if verbose:
        print(f"original shape: {df_epm_original.shape}, final shape: {df_final.shape}, {n_zones} zones.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path)


def _copy_remaining_inputs(
    input_base_dir, output_base_dir, handled_rel_paths, verbose=False
):
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
            message = (
                f"File '{rel_path}' contains zonal data but is not explicitly processed."
            )
            if verbose:
                print(f"[EPM inputs] Unhandled zonal input detected: {message}")
            raise ValueError(message)

        destination = output_base_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
