import numpy as np
import matplotlib.pyplot as plt

def country_viz(reg):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    zones_p = reg.zones.to_crs(epsg=3857)
    lines_p = reg.grid.lines.to_crs(epsg=3857)
    zones_p["centers"] = zones_p.geometry.centroid

    zones_p.plot(edgecolor="black", color="lightblue", ax=ax)
    lines_p.plot(color="pink", ax=ax)

    for idx, z in zones_p.iterrows():
        x, y = z.centers.x, z.centers.y
        ax.text(x, y, idx, fontsize=22)

    # Flow model overlay
    # Get the maximum capacity in the flow model
    maxmw = np.max(np.max(reg.grid.flow))
    maxlw = 5

    zidx = zones_p.index
    for i in zidx:
        for j in zidx:
            ci = zones_p.centers.loc[i]
            cj = zones_p.centers.loc[j]
            capmw = reg.grid.flow.loc[i, j]
            plt.plot([ci.x, cj.x], [ci.y, cj.y], 
                     linewidth=(capmw/maxmw)*maxlw)