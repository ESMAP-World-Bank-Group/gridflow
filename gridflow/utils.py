import errno
import os
import math
from collections import defaultdict
from pathlib import Path

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


def verbose_log(step, message, verbose):
    """Print the step name and message when verbose is requested."""
    if verbose:
        print(f"[{step}] {message}")


def ensure_file_exists(path, *, verbose=False):
    """Raise if the supplied file path is missing, optionally logging the resolved path."""
    file_path = Path(path)
    if not file_path.exists():
        resolved = file_path.resolve()
        verbose_log("DATA_LOAD", f"Missing file: {resolved}", verbose)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(resolved))
    return file_path

class country_code_map:
    def __init__(self, code_path="data/global_datasets/country_names.csv"):
        path = ensure_file_exists(code_path)
        self.codes = pd.read_csv(path)

    def iso3_to_name(self, iso3):
        return self.codes[self.codes["alpha-3"]==iso3]["name"].iloc[0]


def directional_zone_labels(zones, country_col="country"):
    """Return names like 'KEN-North' based on centroid direction per country."""
    if "geometry" not in zones:
        raise ValueError("zones must include a geometry column")

    default_country = "__GLOBAL__"
    if country_col in zones:
        country_values = zones[country_col].fillna(default_country)
    else:
        country_values = pd.Series([default_country] * len(zones), index=zones.index)

    centers = {}
    for country in country_values.unique():
        mask = country_values == country
        subset = zones.loc[mask]
        if subset.empty:
            continue
        minx, miny, maxx, maxy = subset.total_bounds
        centers[country] = ((minx + maxx) / 2, (miny + maxy) / 2)

    records = []
    for idx in zones.index:
        centroid = zones.geometry.centroid.loc[idx]
        country = country_values.loc[idx]
        center = centers.get(country)
        dx = centroid.x - center[0] if center is not None else 0
        dy = centroid.y - center[1] if center is not None else 0
        base_dir = _cardinal_direction(dx, dy) if center is not None else "Center"
        records.append({
            "idx": idx,
            "country": country,
            "dx": dx,
            "dy": dy,
            "direction": base_dir,
            "final_direction": base_dir,
        })

    df = pd.DataFrame(records).set_index("idx")

    cardinal_variants = {
        "North": lambda dx, dy: "North-East" if dx >= 0 else "North-West",
        "South": lambda dx, dy: "South-East" if dx >= 0 else "South-West",
        "East": lambda dx, dy: "North-East" if dy >= 0 else "South-East",
        "West": lambda dx, dy: "North-West" if dy >= 0 else "South-West",
    }

    for (country, direction), group in df.groupby(["country", "direction"]):
        if len(group) <= 1 or direction not in cardinal_variants:
            continue
        variant = cardinal_variants[direction]
        for idx, row in group.iterrows():
            dx, dy = row["dx"], row["dy"]
            df.at[idx, "final_direction"] = variant(dx, dy)

    counts = defaultdict(int)
    labels = []
    for idx, row in df.iterrows():
        country = row["country"]
        direction = row["final_direction"]
        if direction == "Center":
            label = country if country != default_country else "Center"
        else:
            counts[(country, direction)] += 1
            ordinal = counts[(country, direction)]
            base_label = direction if ordinal == 1 else f"{direction}-{ordinal}"
            label = f"{country}-{base_label}" if country != default_country else base_label
        labels.append(label)

    return labels


def _cardinal_direction(dx, dy):
    if math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9):
        return "Center"
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    if angle >= 337.5 or angle < 22.5:
        return "East"
    if angle < 67.5:
        return "North-East"
    if angle < 112.5:
        return "North"
    if angle < 157.5:
        return "North-West"
    if angle < 202.5:
        return "West"
    if angle < 247.5:
        return "South-West"
    if angle < 292.5:
        return "South"
    if angle < 337.5:
        return "South-East"
    return "Center"
