import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D


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
    flow_value_scale=1.0,
    flow_value_fmt="{:.0f}",
):
    """Visualize grid zones and flow capacities with configurable styling."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

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
