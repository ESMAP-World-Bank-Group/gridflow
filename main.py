"""
Command-line entry point mirroring the GridflowProcess notebook.

Runs the core pipeline:
1) build a region and create zones,
2) compute zonal statistics + renewables profiles,
3) create the transmission network flow representation,
4) generate zonalized EPM inputs.
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from gridflow import model, epm_input_generator, visuals
from gridflow.data_readers import (
    get_global_dataset_file_path,
    get_global_datasets_path,
    read_borders,
    read_line_data,
)
from gridflow.utils import verbose_log


def _comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _save_figure(fig, destination):
    """Persist the supplied figure while ensuring its directory exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _export_zone_segmentation(region, plot_root):
    """Store the segmented zone map if zones exist."""
    if region.zones.empty:
        return
    fig, _ = visuals.zone_segmentation_map(region)
    _save_figure(fig, plot_root / "create_zones_zones.pdf")


def _export_zone_stats(region, plot_root):
    """Write choropleth plots for each loaded zone statistic."""
    if region.zones.empty or region.zone_stats.empty:
        return
    for stat_column in region.zone_stats.columns:
        fig, _ = visuals.zone_stat_choropleth(region, stat_column)
        safe_name = stat_column.replace(" ", "_")
        _save_figure(fig, plot_root / f"set_zone_data_{safe_name}.pdf")


def _export_network_plots(region, plot_root):
    """Save network-level visualizations, ignoring failures."""
    try:
        fig, _ = visuals.country_viz(region, title="Network visualization", show_arrows=True, show_flow_values=True)
        _save_figure(fig, plot_root / "create_network_map.pdf")
    except Exception:
        pass
    try:
        fig, _ = visuals.flow_field_heatmap(region, title="Network flow heatmap")
        _save_figure(fig, plot_root / "create_network_flow_heatmap.pdf")
    except Exception:
        pass


def _export_neighbor_map(region, lines, plot_dir, global_data_path, verbose=True):
    """Save a map showing the country context and the neighbor transmission lines."""
    if lines.empty or not plot_dir:
        return
    plot_root = Path(plot_dir)
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    countries = region.countries.to_crs(epsg=3857)
    lines_proj = lines.to_crs(epsg=3857)
    border_path = get_global_dataset_file_path(
        "borders", "borders/WB_GAD_ADM0_complete.shp", root=global_data_path
    )
    all_countries = read_borders(border_path)
    verbose_log(
        "NEIGHBOR_MAP",
        f"Loaded {len(all_countries)} country polygons from {border_path}.",
        verbose,
    )
    all_proj = all_countries.to_crs(countries.crs)
    line_union = lines_proj.geometry.union_all()
    iso_col = next(
        (col for col in ("ISO_A3", "ADM0_A3", "iso_a3", "adm0_a3") if col in all_proj.columns),
        None,
    )
    neighbors = None
    if iso_col:
        region_isos = set(region.countries["ISO_A3"].tolist())
        touch = all_proj[all_proj.geometry.intersects(line_union)]
        neighbors = touch[~touch[iso_col].isin(region_isos)]
        if verbose and neighbors is not None and not neighbors.empty:
            neighbor_list = ", ".join(sorted(set(neighbors[iso_col].tolist())))
            verbose_log(
                "NEIGHBOR_MAP",
                f"Neighbors detected via borders file: {neighbor_list}.",
                verbose,
            )
        if not neighbors.empty:
            neighbors.plot(
                ax=ax,
                facecolor="navajowhite",
                edgecolor="orange",
                linewidth=1.2,
                alpha=0.5,
                zorder=0,
            )
    countries.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=1.2)
    lines_proj.plot(ax=ax, color="tab:blue", linewidth=1.2, alpha=0.7)
    bounds = countries.total_bounds
    if not np.isnan(bounds).any():
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.2
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    for idx, row in countries.iterrows():
        centroid = row.geometry.centroid
        ax.text(
            centroid.x,
            centroid.y,
            row.get("NAME", row.get("ADMIN", row.get("ISO_A3", ""))),
            fontsize=10,
            fontweight="bold",
            color="black",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
            ha="center",
            va="center",
        )
    if neighbors is not None and not neighbors.empty:
        for idx, row in neighbors.iterrows():
            intersection = row.geometry.intersection(line_union)
            if intersection.is_empty:
                location = row.geometry.centroid
            else:
                location = intersection.representative_point()
            label = row.get("NAME", row.get("ADMIN", row.get("ISO_A3", "")))
            ax.text(
                location.x,
                location.y,
                label,
                fontsize=9,
                color="dimgray",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
                ha="center",
                va="center",
            )
    ax.set_title("Neighbor transmission lines")
    ax.set_axis_off()
    fig.tight_layout()
    _save_figure(fig, plot_root / "neighbor_lines_map.pdf")


def summarize_country_exports(country_iso, neighbor_caps, *, verbose=False):
    """Log export-style neighbors for a single country given neighbor capacities."""
    iso = country_iso.strip().upper()
    if not iso:
        return {}
    exports = []
    for pair, cap in neighbor_caps.items():
        if iso in pair:
            neighbor = pair[1] if pair[0] == iso else pair[0]
            exports.append((neighbor, cap))

    if not exports:
        verbose_log("COUNTRY_EXPORTS", f"No neighboring exports discovered for {iso}.", verbose)
        return {}

    total = sum(cap for _, cap in exports)
    verbose_log(
        "COUNTRY_EXPORTS",
        f"{iso} has {len(exports)} neighbor(s) with {total:.1f} MW total capacity.",
        verbose,
    )
    for neighbor, cap in sorted(exports):
        verbose_log(
            "COUNTRY_EXPORTS",
            f"  {iso} ↔ {neighbor} : {cap:.1f} MW available.",
            verbose,
        )
    return dict(exports)


def run_zoning_for_epm(
    *,
    data_path,
    countries,
    n_zones,
    zone_stats_to_load,
    method_zoning,
    epm_input_raw,
    epm_output_dir,
    demand_scaleby="population",
    generate_plots=True,
    plot_dir="output",
    verbose=False,
):
    """Execute the zone segmentation + EPM pipeline."""
    region = model.region(
        countries,
        data_path,
        zone_stats_to_load=zone_stats_to_load,
    )

    region.create_zones(n=n_zones, method=method_zoning, verbose=verbose)
    if generate_plots:
        _export_zone_segmentation(region, Path(plot_dir))
    region.set_zone_data(verbose=verbose)
    if generate_plots:
        _export_zone_stats(region, Path(plot_dir))
    region.create_network()
    if generate_plots:
        _export_network_plots(region, Path(plot_dir))

    epm_input_generator.generate_epm_inputs(
        region,
        epm_input_raw,
        epm_output_dir,
        demand_scaleby=demand_scaleby,
        verbose=verbose,
    )
    return region


def _plot_neighbor_capacity(neighbor_caps, plot_dir, *, verbose=False):
    """Draw a bar chart summarizing neighbor interconnect capacity."""
    if not neighbor_caps or not plot_dir:
        return
    plot_root = Path(plot_dir)
    plot_root.mkdir(parents=True, exist_ok=True)
    pairs = sorted(neighbor_caps.items())
    labels = [f"{a}-{b}" for (a, b), _ in pairs]
    values = [cap for _, cap in pairs]
    fig, ax = plt.subplots(figsize=(max(6, len(values) * 0.4), 4))
    y = list(range(len(values)))
    ax.barh(y, values, color="tab:blue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Capacity (MW)")
    ax.set_title("Neighbor Interconnect Capacities")
    fig.tight_layout()
    plot_path = plot_root / "neighbor_capacities.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    verbose_log("NEIGHBOR_CAPACITY", f"Saved neighbor capacity plot to {plot_path}", verbose)


def run_neighbors(
    data_path,
    countries,
    plot_dir="output",
    verbose=False,
):
    """Build country-based network and summarize neighbor capacities."""
    region = model.region(countries, data_path)
    lines = read_line_data(region.grid.path, region, minkm=0)
    lines = lines.to_crs(region.countries.crs)
    _export_neighbor_map(region, lines, plot_dir, data_path)
    lines["capacity"] = region.grid._get_line_capacity(lines)
    neighbor_caps = _compute_country_neighbor_caps(region, lines, verbose=verbose)
    region.grid.lines = lines
    _plot_neighbor_capacity(neighbor_caps, plot_dir, verbose=verbose)
    return region, neighbor_caps


def _compute_country_neighbor_caps(region, lines, *, verbose=False):
    """Aggregate transmission capacity per country pair from the line GeoDataFrame."""
    neighbor_caps = defaultdict(float)
    detail_messages = []
    for line_id, line in lines.iterrows():
        touched = region.countries[region.countries.intersects(line.geometry)]
        isos = sorted({iso for iso in touched["ISO_A3"] if iso})
        if len(isos) < 2:
            continue
        cap_val = line.capacity
        try:
            capacity = float(cap_val) if np.isfinite(cap_val) else 0.0
        except (TypeError, ValueError):
            capacity = 0.0
        for pair in combinations(isos, 2):
            neighbor_caps[pair] += capacity
            if verbose:
                detail_messages.append(
                    f"Line {line_id} ({capacity:.1f} MW) links {pair[0]} and {pair[1]}."
                )
    if verbose:
        verbose_log(
            "NEIGHBOR_CAPACITY",
            f"Computed {len(neighbor_caps)} neighbor pair(s) from {len(lines)} lines.",
            verbose,
        )
        for message in detail_messages:
            verbose_log("NEIGHBOR_CAPACITY", message, verbose)
        for pair, cap in sorted(neighbor_caps.items()):
            verbose_log(
                "NEIGHBOR_CAPACITY",
                f"  {pair[0]} <-> {pair[1]} : {cap:.1f} MW total capacity",
                verbose,
            )
    return neighbor_caps




def parse_args():
    parser = argparse.ArgumentParser(description="Run the gridflow pipeline.")
    parser.add_argument(
        "--data-path",
        default=get_global_datasets_path(),
        help="Path containing borders, rasters, and grid.gpkg (default from config.yaml).",
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
        default=4,
        help="Number of zones per country.",
    )
    parser.add_argument(
        "--zone-stats",
        default=["population", "gdp"],
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
        "--mode",
        choices=["zoning", "neighbors"],
        default="zoning",
        help="Choose workflow: zoning/EPM pipeline or neighbor capacity summary.",
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
        "--demand-scaleby",
        default="population",
        help="Zone statistic column used to distribute country-level demand (default: population).",
    )
    parser.add_argument(
        "--plot-dir",
        default="output",
        help="Directory where generated maps for each stage are stored.",
    )
    parser.add_argument(
        "--generate-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), save maps after each pipeline stage.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True, # keep it
        help="Print detailed progress.",
    )
    parser.add_argument(
        "--export-country",
        help="ISO3 code of a country for which to print export neighbors after the run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    neighbor_caps = {}
    try:
        if args.mode == "neighbors":
            region, neighbor_caps = run_neighbors(
                data_path=args.data_path,
                countries=args.countries,
                plot_dir=args.plot_dir,
                verbose=args.verbose,
            )
        else:
            region = run_zoning_for_epm(
                data_path=args.data_path,
                countries=args.countries,
                n_zones=args.n_zones,
                zone_stats_to_load=args.zone_stats,
                method_zoning=args.method_zoning,
                epm_input_raw=args.epm_input_raw,
                epm_output_dir=args.epm_output_dir,
                demand_scaleby=args.demand_scaleby,
                generate_plots=args.generate_plots,
                plot_dir=args.plot_dir,
                verbose=args.verbose,
            )
    except FileNotFoundError as err:
        if args.verbose and err.filename:
            verbose_log("RUN_ZONING_FOR_EPM", f"Missing file: {Path(err.filename).resolve()}", args.verbose)
        raise
    if args.export_country:
        summarize_country_exports(args.export_country, neighbor_caps, verbose=args.verbose)
    print(f"Created {len(region.zones)} zones across {len(args.countries)} countries.")
    if args.mode == "zoning":
        print(f"Zonal EPM inputs written to {Path(args.epm_output_dir).resolve()}")


if __name__ == "__main__":
    main()
