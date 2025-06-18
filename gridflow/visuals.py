import numpy as np
import matplotlib.pyplot as plt

def country_viz(terr):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    regions_p = terr.regions.to_crs(epsg=3857)
    lines_p = terr.grid.lines.to_crs(epsg=3857)
    regions_p["centers"] = regions_p.geometry.centroid

    regions_p.plot(edgecolor="black", color="lightblue", ax=ax)
    lines_p.plot(color="pink", ax=ax)

    for idx, r in regions_p.iterrows():
        x, y = r.centers.x, r.centers.y
        ax.text(x, y, idx, fontsize=22)

    # Flow model overlay
    # Get the maximum capacity in the flow model
    maxmw = np.max(np.max(terr.grid.flow))
    maxlw = 5

    ridx = regions_p.index
    for i in ridx:
        for j in ridx:
            ci = regions_p.centers.loc[i]
            cj = regions_p.centers.loc[j]
            capmw = terr.grid.flow.loc[i, j]
            plt.plot([ci.x, cj.x], [ci.y, cj.y], 
                     linewidth=(capmw/maxmw)*maxlw)