"""
Command-line entry point mirroring the GridflowProcess notebook.

Runs the core pipeline:
1) build a region and create zones,
2) compute zonal statistics + renewables profiles,
3) create the transmission network flow representation,
4) generate zonalized EPM inputs.
"""

import argparse
from pathlib import Path

from gridflow import model, epm_input_generator


def _comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def run_pipeline(
    *,
    data_path,
    countries,
    n_zones,
    zone_stats_to_load,
    method_zoning,
    epm_input_raw,
    epm_output_dir,
    verbose=False,
):
    """Execute the end-to-end gridflow pipeline."""
    region = model.region(
        countries,
        data_path,
        zone_stats_to_load=zone_stats_to_load,
    )

    region.create_zones(n=n_zones, method=method_zoning)
    region.set_zone_data(verbose=verbose)
    region.create_network()

    epm_input_generator.generate_epm_inputs(
        region,
        epm_input_raw,
        epm_output_dir,
        verbose=verbose,
    )
    return region


def parse_args():
    parser = argparse.ArgumentParser(description="Run the gridflow pipeline.")
    parser.add_argument(
        "--data-path",
        default="data/global_datasets",
        help="Path containing borders, rasters, and grid.gpkg.",
    )
    parser.add_argument(
        "--countries",
        default="LUX",
        type=_comma_list,
        help="Comma-separated ISO3 codes (default: LUX).",
    )
    parser.add_argument(
        "--n-zones",
        type=int,
        default=5,
        help="Number of zones per country.",
    )
    parser.add_argument(
        "--zone-stats",
        default="population",
        type=_comma_list,
        help="Comma-separated list of zone statistics to load (default: population).",
    )
    parser.add_argument(
        "--method-zoning",
        choices=["pv", "wind"],
        default="pv",
        help="Segmentation method for zone creation.",
    )
    parser.add_argument(
        "--epm-input-raw",
        default="data/epm_inputs_raw",
        help="Path to raw EPM CSV templates.",
    )
    parser.add_argument(
        "--epm-output-dir",
        default="data/epm_inputs",
        help="Destination for zonalized EPM CSVs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    region = run_pipeline(
        data_path=args.data_path,
        countries=args.countries,
        n_zones=args.n_zones,
        zone_stats_to_load=args.zone_stats,
        method_zoning=args.method_zoning,
        epm_input_raw=args.epm_input_raw,
        epm_output_dir=args.epm_output_dir,
        verbose=args.verbose,
    )
    print(f"Created {len(region.zones)} zones across {len(args.countries)} countries.")
    print(f"Zonal EPM inputs written to {Path(args.epm_output_dir).resolve()}")


if __name__ == "__main__":
    main()
