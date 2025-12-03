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
        },
        "pTransferLimit": {
            "path": "trade/pTransferLimit.csv",
            "func": trade_transfer_limit,
            "args": []
        }
    }

    processed_rel_paths = set()

    for param_name, param in toprocess.items():
        verbose_log("EPM_PROCESS", f"Processing {param_name}", verbose)
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

def zone_replicate(region, input_path, output_path, verbose=True):
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
    verbose_log("EPM_ZONE_REPLICATE", f"Reading input file: {input_path}", verbose)
        
    df_epm_original = pd.read_csv(input_path)
 
    if "zone" not in df_epm_original.columns:
        raise ValueError("The input data is not zonal.")

    df_epm_original["zone_str"] = df_epm_original["zone"].astype(str)
    country_set = set(region.zones["country"].astype(str))
    static_rows = df_epm_original[~df_epm_original["zone_str"].isin(country_set)].copy()

    replicated_rows = []
    for zone_id in zone_ids:
        zone_country = str(region.zones.at[zone_id, "country"])
        mask = df_epm_original["zone_str"] == zone_country
        if not mask.any():
            verbose_log(
                "EPM_ZONE_REPLICATE",
                f"No source rows for country '{zone_country}'; skipping zone {zone_id}.",
                verbose,
            )
            continue
        segment = df_epm_original[mask].copy()
        segment["zone"] = zone_id
        replicated_rows.append(segment)

    if replicated_rows:
        df_final = pd.concat([static_rows] + replicated_rows, ignore_index=True)
    else:
        df_final = static_rows.copy()
    df_final.drop(columns=["zone_str"], inplace=True, errors="ignore")
    verbose_log(
        "EPM_ZONE_REPLICATE",
        f"Original shape: {df_epm_original.shape}, final shape: {df_final.shape}, "
        f"{len(replicated_rows)} zones replicated out of {n_zones}.",
        verbose,
    )
    
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

    verbose_log(
        "EPM_ZONE_DISTRIBUTE",
        f"Distributing using '{scaleby}' (sum={denom:.6f}) across {n_zones} zones.",
        verbose,
    )
    
    # Read the original CSV
    verbose_log("EPM_ZONE_DISTRIBUTE", f"Reading input file: {input_path}", verbose)
    
    df_epm_original = pd.read_csv(input_path)

    if "zone" not in df_epm_original.columns:
        raise ValueError(
            "The input data must include a 'zone' column; matching by country is not supported."
        )

    if "country" not in region.zones.columns:
        raise ValueError("Region zones must include a 'country' column to map legacy inputs.")
    df_final = []
    for z in zone_ids:
        zone_country = region.zones.at[z, "country"]
        mask = df_epm_original["zone"].astype(str) == str(zone_country)
        if not mask.any():
            raise ValueError(
                f"No rows found in input for legacy zone '{zone_country}' while mapping to new zone {z}."
            )
        df = df_epm_original[mask].copy()
        df["zone"] = z

        # Restrict scaling to numeric columns not explicitly excluded
        excluded = set(exclude_cols) | {"zone"}
        numeric_cols = [
            c for c in df.columns.difference(excluded)
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        df[numeric_cols] = df[numeric_cols].mul(scale.loc[z])
        verbose_log(
            "EPM_ZONE_DISTRIBUTE",
            f"  - Zone {z} (country {zone_country}): '{scaleby}' value "
            f"{region.zone_stats.loc[z, scaleby]:.6f}, scale factor {scale.loc[z]:.6f}.",
            verbose,
        )
        df_final.append(df)
    df_final = pd.concat(df_final, ignore_index=True)
    static_rows = df_epm_original[~df_epm_original["zone"].astype(str).isin(region.zones["country"].astype(str))].copy()
    if not static_rows.empty:
        df_final = pd.concat([static_rows, df_final], ignore_index=True)
    verbose_log(
        "EPM_ZONE_DISTRIBUTE",
        f"Original shape: {df_epm_original.shape}, final shape: {df_final.shape}, {n_zones} zones.",
        verbose,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path)


def trade_transfer_limit(region, input_path, output_path, verbose=True):
    """Generate trade/pTransferLimit entries from the region grid flow model."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if region.grid.flow is None:
        raise ValueError("Flow model missing; run region.create_network() before computing transfer limits.")

    template = pd.read_csv(input_path, dtype=str)
    quarters = list(dict.fromkeys(template["q"].dropna().astype(str).tolist()))
    base_years = [col for col in template.columns if col not in {"From", "To", "q"}]
    if not base_years:
        raise ValueError("pTransferLimit template lacks year columns.")

    flow = region.grid.flow.fillna(0)
    rows = []
    for frm in flow.index:
        for to in flow.columns:
            if frm == to:
                continue
            cap = flow.at[frm, to]
            if not np.isfinite(cap) or cap <= 0:
                continue
            for q in quarters:
                row = {"From": frm, "To": to, "q": q}
                row.update({year: cap for year in base_years})
                rows.append(row)

    if not rows:
        verbose_log("EPM_TRANSFER_LIMIT", "No inter-zone capacity was found; leaving pTransferLimit empty.", verbose)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=template.columns).to_csv(output_path, index=False)
        return

    df_final = pd.DataFrame(rows, columns=["From", "To", "q"] + base_years)
    verbose_log(
        "EPM_TRANSFER_LIMIT",
        f"Created {len(df_final)} transfer limit rows across {len(quarters)} quarter(s) from {len(flow.index)} zones.",
        verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)


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
                verbose_log("EPM_PROCESS", f"[EPM inputs] Unhandled zonal input detected: {message}", verbose)
                raise ValueError(message)

        destination = output_base_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, destination)
