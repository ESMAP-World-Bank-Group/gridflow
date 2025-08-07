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

def generate_epm_inputs(region, input_base_dir, output_base_dir, verbose=False):
    """Master function to process all EPM input files.
    
    Parameters
    ----------
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
    toprocess = {
    # Top-level files
        "config" : {
            "path" : "config.csv",
            "func" : "noop",
            "args" : []
        },
    # Load data
    	"pDemandProfile" : {
    		"path" : "load/pDemandProfile.csv",
    		"func" : "subregion_replicate",
    		"args" : []
    	},
    	"pDemandForecast" : {
    	 	"path" : "load/pDemandForecast.csv",
    	 	"func" : "subregion_distribute",
    	 	"args" : [["zone", "type"]]
    	},
    # Supply data
        "pGenDataExcelDefault" : {
            "path" : "supply/pGenDataExcelDefault.csv",
            "func" : "subregion_replicate",
            "args" : []
        }
    }

    for a, b in toprocess.items():
        if verbose:
            print(f"Processing {a}")
        in_path = input_base_dir + "/" + b["path"]
        out_path = output_base_dir + "/" + b["path"]
        globals()[b["func"]](region, in_path, out_path, *b["args"], verbose=verbose)
    return None

def noop(region, input_path, output_path, verbose=False):
    """Do nothing."""
    shutil.copyfile(input_path, output_path)

def zone_replicate(region, input_path, output_path, verbose=False):
    """Replicate input data for each zone.

    Takes a CSV file with a 'zone' column and replicates the data
    for each zone defined in the model.
    """
    
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
    
    df_final.to_csv(output_path)


def zone_distribute(region, input_path, output_path, exclude_cols=[], 
	                scaleby="population", verbose=False):
    """Proportionally distribute input data to each zone. 

    Takes a CSV file with a 'zone' column and distributes the specified
    time series quantities to each zone proportionally.
    """

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
            df = df_original.assign(zone=z)
            df[df.columns.difference(exclude_cols)] *= scale.loc[z]
            df_final.append(df)
        df_final = pd.concat(df_final, ignore_index=True)
        if verbose:
            print(f"original shape: {df_original.shape}, final shape: {df_final.shape}, {n_zones} zones.")
    else:
        raise ValueError("The input data is not zonal.")
    
    df_final.to_csv(output_path)

