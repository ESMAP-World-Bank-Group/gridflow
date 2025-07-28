import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

def get_random_points(gpdf, n=20, maxattempts=10):
    "Obtains n randomly sampled points within a geometry."
    # Convert the geometry to a projected one for even sampling
    gpdf_flat = gpdf.to_crs(epsg=32633)
    # Get union of all polygons
    boundary = gpdf_flat.unary_union
    minx, miny, maxx, maxy = get_bb(gpdf_flat)

    n_found, attempts = 0, 0
    all_pts = []
    while (n_found < n) and (attempts < maxattempts):
        # Generate n random points within the bounding box
        x = np.random.uniform(minx, maxx, n)
        y = np.random.uniform(miny, maxy, n)
        points = gpd.GeoDataFrame(geometry = [Point(i, j) for i, j in zip(x, y)],
                                  crs = gpdf_flat.crs)
        # Filter down to points within the polygons
        points = points[points.geometry.within(boundary)]

        n_new = len(points)
        remain = n - n_found
        if n_new >= remain:
            points = points.head(remain)
            n_new = remain
        all_pts.append(points)
        n_found += n_new
        attempts += 1
    # Combine samples and return
    all_pts = gpd.GeoDataFrame(pd.concat(all_pts, ignore_index=True))
    return all_pts.to_crs(gpdf.crs)
    

def get_bb(gpdf, crs=None, buffer_degrees=0.1):
    "Get the bounding box for a set of polygons in a geopandas df."
    if crs is not None:
        gpdf = gpdf.to_crs(crs)
    minx, miny, maxx, maxy = gpdf.total_bounds

    minx -= buffer_degrees
    miny -= buffer_degrees
    maxx += buffer_degrees
    maxy += buffer_degrees

    return minx, miny, maxx, maxy