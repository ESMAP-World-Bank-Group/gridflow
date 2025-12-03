import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from gridflow.data_readers import load_background_map


_DEFAULT_COUNTRIES_BACKGROUND = "data/maps/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
_DEFAULT_POPULATED_BACKGROUND = "data/maps/ne_110m_populated_places/ne_110m_populated_places.shp"


def _prepare_background_layer(name, default_path):
    """Load a background layer if the shapefile is available."""
    gdf = load_background_map(name, default_path=default_path)
    if gdf is None or gdf.empty:
        return None
    return gdf


def _plot_background_layer(ax, gdf, *, target_crs=None, zorder=0, **kwargs):
    """Plot a pre-projected GeoDataFrame if available."""
    if gdf is None or gdf.empty:
        return
    plot_gdf = gdf.to_crs(target_crs) if target_crs is not None else gdf
    plot_gdf.plot(ax=ax, zorder=zorder, **kwargs)


def _fit_axes_to_bounds(ax, bounds, *, buffer_factor=0.05, min_span=1e4):
    """Clamp axes to given bounds with a safety margin to avoid global zoom."""
    if bounds is None:
        return
    minx, miny, maxx, maxy = bounds
    if np.isnan([minx, miny, maxx, maxy]).any():
        return

    margin_x = max(maxx - minx, min_span) * buffer_factor
    margin_y = max(maxy - miny, min_span) * buffer_factor
    ax.set_xlim(minx - margin_x, maxx + margin_x)
    ax.set_ylim(miny - margin_y, maxy + margin_y)


def _fit_axes_to_zones(ax, zones_gdf, **kwargs):
    """Zoom the axis to the extent of the supplied zones GeoDataFrame."""
    if zones_gdf is None or zones_gdf.empty:
        return
    _fit_axes_to_bounds(ax, zones_gdf.total_bounds, **kwargs)


_COUNTRY_BACKGROUND = _prepare_background_layer("countries", _DEFAULT_COUNTRIES_BACKGROUND)
_POPULATED_BACKGROUND = _prepare_background_layer("populated_places", _DEFAULT_POPULATED_BACKGROUND)


def flow_field_heatmap(
    reg,
    figsize=(7, 5),
    percentile=99,
    cmap="viridis",
    flow_units="MW",
    title="Flow field for the generated network",
    show_grid=False,
):
    """Plot a heatmap of grid flow capacities with labeled axes."""
    if reg.grid.flow is None:
        raise ValueError("Grid flow data not found. Run region.create_network() first.")

    flow = reg.grid.flow.values.astype(float)
    flow = np.where(np.isfinite(flow), flow, np.nan)

    # Clip extreme values so outliers do not flatten the scale.
    vmax = np.nanpercentile(flow, percentile) if np.isfinite(np.nanmax(flow)) else None
    vmax = vmax if vmax and vmax > 0 else None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        flow,
        cmap=cmap,
        origin="upper",
        vmin=0,
        vmax=vmax
    )

    zones = reg.grid.flow.index
    ax.set_xticks(np.arange(len(zones)))
    ax.set_yticks(np.arange(len(zones)))
    ax.set_xticklabels(zones)
    ax.set_yticklabels(zones)
    ax.set_xlabel("To zone")
    ax.set_ylabel("From zone")
    ax.set_title(title)

    ax.grid(show_grid)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Flow ({flow_units})")
    fig.tight_layout()
    return fig, ax


def country_viz(
    reg,
    figsize=(15, 10),
    flow_percentile=95,
    max_linewidth=8,
    min_capacity=0,
    flow_color="darkolivegreen",
    zone_label_size=12,
    flow_units="MW",
    show_arrows=False,
    arrow_head_scale=10,
    alpha=0.9,
    title=None,
    show_grid=False,
    show_flow_values=False,
    show_background=False,
    flow_value_scale=1.0,
    flow_value_fmt="{:.0f}",
):
    """Visualize grid zones and flow capacities with configurable styling."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if show_background:
        _plot_background_layer(
            ax,
            _COUNTRY_BACKGROUND,
            target_crs=zones_p.crs,
            facecolor="whitesmoke",
            edgecolor="gray",
            linewidth=0.5,
            alpha=0.75,
        )
        _plot_background_layer(
            ax,
            _POPULATED_BACKGROUND,
            target_crs=zones_p.crs,
            color="firebrick",
            markersize=3,
            alpha=0.4,
        )

    zones_p = reg.zones.to_crs(epsg=3857)
    lines_p = reg.grid.lines.to_crs(epsg=3857)
    zones_p["centers"] = zones_p.geometry.centroid

    zones_p.plot(edgecolor="black", color="lightgrey", linewidth=2, ax=ax)
    lines_p.plot(color="mediumseagreen", linewidth=1, ax=ax, alpha=0.6)

    for idx, z in zones_p.iterrows():
        x, y = z.centers.x, z.centers.y
        ax.text(
            x,
            y,
            idx,
            fontsize=zone_label_size,
            ha="center",
            va="center",
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
        )

    flow_values = reg.grid.flow.values
    finite_flow = flow_values[np.isfinite(flow_values)]
    if finite_flow.size == 0:
        maxmw = 0
    else:
        # Percentile scaling reduces dominance of outliers.
        maxmw = np.nanpercentile(finite_flow, flow_percentile)
    maxmw = maxmw if maxmw > 0 else 1  # avoid divide-by-zero

    ax.grid(show_grid)
    _fit_axes_to_zones(ax, zones_p)
    ax.set_axis_off()

    zidx = zones_p.index
    for i in zidx:
        for j in zidx:
            capmw = reg.grid.flow.loc[i, j]
            if not np.isfinite(capmw) or capmw <= min_capacity:
                continue
            ci = zones_p.centers.loc[i]
            cj = zones_p.centers.loc[j]
            lw = min((capmw / maxmw) * max_linewidth, max_linewidth)
            if show_arrows and i != j:
                arrow = FancyArrowPatch(
                    (ci.x, ci.y),
                    (cj.x, cj.y),
                    arrowstyle="->",
                    mutation_scale=arrow_head_scale,
                    color=flow_color,
                    linewidth=lw,
                    alpha=alpha,
                    shrinkA=5,
                    shrinkB=5,
                    zorder=3,
                )
                ax.add_patch(arrow)
            else:
                ax.plot(
                    [ci.x, cj.x],
                    [ci.y, cj.y],
                    linewidth=lw,
                    color=flow_color,
                    alpha=alpha,
                )

            if show_flow_values and i != j:
                mx, my = (ci.x + cj.x) / 2, (ci.y + cj.y) / 2
                display_val = flow_value_scale * capmw
                label = f"{flow_value_fmt.format(display_val)} {flow_units}".strip()
                ax.text(
                    mx,
                    my,
                    label,
                    fontsize=max(zone_label_size - 2, 8),
                    ha="center",
                    va="center",
                    color=flow_color,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
                )

    if title:
        ax.set_title(title, fontsize=14)

    legend_line = Line2D(
        [0], [0], color=flow_color, lw=max_linewidth, label=f"Flow ({flow_units})"
    )
    ax.legend(handles=[legend_line], loc="lower left", frameon=False)
    return fig, ax


def zone_segmentation_map(
    reg,
    figsize=(10, 6),
    facecolor="lightgrey",
    edgecolor="black",
    linewidth=1,
    title="Zone segmentation",
    show_background=False,
):
    """Render the zone outlines for the current region."""
    fig, ax = plt.subplots(figsize=figsize)
    zones_proj = reg.zones.to_crs(epsg=3857)
    if show_background:
        _plot_background_layer(
            ax,
            _COUNTRY_BACKGROUND,
            target_crs=zones_proj.crs,
            facecolor="whitesmoke",
            edgecolor="gray",
            linewidth=0.4,
            alpha=0.6,
        )
    zones_proj.plot(ax=ax, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)
    ax.set_title(title)
    _fit_axes_to_zones(ax, zones_proj)
    ax.set_axis_off()
    return fig, ax


def zone_stat_choropleth(
    reg,
    stat_column,
    figsize=(10, 6),
    cmap="OrRd",
    linewidth=0.8,
    edgecolor="white",
    title=None,
    show_background=False,
):
    """Render a choropleth of a specific zone statistic."""
    zones_with_stats = reg.zones.join(reg.zone_stats[[stat_column]])
    zones_proj = zones_with_stats.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=figsize)
    if show_background:
        _plot_background_layer(
            ax,
            _COUNTRY_BACKGROUND,
            target_crs=zones_proj.crs,
            facecolor="whitesmoke",
            edgecolor="gray",
            linewidth=0.4,
            alpha=0.6,
        )
    zones_proj.plot(
        column=stat_column,
        cmap=cmap,
        legend=True,
        linewidth=linewidth,
        edgecolor=edgecolor,
        ax=ax,
    )
    ax.set_title(title or f"{stat_column} by zone")
    _fit_axes_to_zones(ax, zones_proj)
    ax.set_axis_off()
    return fig, ax
