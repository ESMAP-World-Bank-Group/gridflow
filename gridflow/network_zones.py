"""Build zones and their transfer capacities directly from the real
transmission-line topology -- the pipeline behind
`region.create_network(method="network")` (see `gridflow.model`).

Graph format used internally: `nodes` (GeoDataFrame, one row per node,
indexed by node id) and `edges` (DataFrame of `from_node, to_node, capacity,
line_ids`, one row per connected node pair).
"""
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import box, Point

from gridflow.data_readers import read_line_data
from gridflow.utils import verbose_log


def build_grid_nodes(countries, resolution_km=50, crs=None):
    """Cover the bounding box of `countries` with square cells of `resolution_km`.

    Uses the local UTM zone by default rather than a fixed global projection,
    since e.g. Web Mercator distorts distance by latitude.
    """
    if crs is None:
        crs = countries.estimate_utm_crs()
    minx, miny, maxx, maxy = countries.to_crs(crs).total_bounds
    step = resolution_km * 1000

    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    # (row, col) let adjacency be found by integer comparison later, rather
    # than by re-deriving it from geometry after the CRS round-trip below.
    records = [
        {"row": row, "col": col, "geometry": box(x, y, x + step, y + step)}
        for row, y in enumerate(ys)
        for col, x in enumerate(xs)
    ]

    nodes = gpd.GeoDataFrame(records, crs=crs).to_crs(countries.crs)
    nodes = nodes[nodes.intersects(countries.union_all())].reset_index(drop=True)
    nodes.index.name = "node"
    return nodes


def build_adjacency_edges(nodes, land, weight=1e-3, land_buffer_deg=0.02):
    """Tiny-weight edges between land-adjacent grid cells (8-connectivity),
    so every land cell is tied into the graph even with no real line through it.

    Checks the midpoint between two lattice-neighbor cells against `land`
    (a MultiPolygon) so cells on different islands don't get linked just
    because they're lattice neighbors across a strait. `weight` must stay far
    below real line capacity so it only breaks ties, never outweighs a real edge.
    """
    buffered_land = land.buffer(land_buffer_deg)
    centroids = nodes.geometry.centroid
    pos_to_node = {(r, c): idx for idx, r, c in nodes[["row", "col"]].itertuples()}
    # Only the "forward" half of each cell's 8 neighbors, so each pair is
    # counted once rather than twice.
    offsets = [(1, 0), (0, 1), (1, 1), (1, -1)]

    records = []
    for idx, r, c in nodes[["row", "col"]].itertuples():
        c_u = centroids.loc[idx]
        for dr, dc in offsets:
            neighbor = pos_to_node.get((r + dr, c + dc))
            if neighbor is None:
                continue
            c_v = centroids.loc[neighbor]
            midpoint = Point((c_u.x + c_v.x) / 2, (c_u.y + c_v.y) / 2)
            if not buffered_land.contains(midpoint):
                continue
            records.append({"from_node": idx, "to_node": neighbor,
                            "capacity": 0.0, "weight": weight, "line_ids": []})
    return pd.DataFrame(records, columns=["from_node", "to_node", "capacity", "weight", "line_ids"])


def condense_by_modularity(nodes, edges, weight="capacity", n_clusters=None):
    """Condense the graph into communities via greedy modularity maximization.
    """
    G = nx.Graph()
    G.add_nodes_from(nodes.index)
    for _, e in edges.iterrows():
        if G.has_edge(e["from_node"], e["to_node"]):
            G[e["from_node"]][e["to_node"]]["weight"] += e[weight]
        else:
            G.add_edge(e["from_node"], e["to_node"], weight=e[weight])

    if n_clusters is not None:
        n_clusters = max(n_clusters, nx.number_connected_components(G))

    kwargs = {"best_n": n_clusters, "cutoff": n_clusters} if n_clusters else {}
    communities = nx.community.greedy_modularity_communities(G, weight="weight", **kwargs)
    node_to_cluster = {n: i for i, community in enumerate(communities) for n in community}

    clustered = nodes.copy()
    clustered["cluster"] = clustered.index.map(node_to_cluster)
    condensed_nodes = clustered.dissolve(by="cluster")
    condensed_nodes.index.name = "node"

    cross = edges.copy()
    cross["from_cluster"] = cross["from_node"].map(node_to_cluster)
    cross["to_cluster"] = cross["to_node"].map(node_to_cluster)
    cross = cross[cross["from_cluster"] != cross["to_cluster"]]
    cross["cluster_pair"] = [tuple(sorted(p)) for p in zip(cross["from_cluster"], cross["to_cluster"])]

    grouped = cross.groupby("cluster_pair").agg(
        capacity=("capacity", "sum"),
        line_ids=("line_ids", lambda s: sum(s, [])),
    )
    # Explicit columns: if no cluster pair has any edge, grouped is empty and
    # a plain pd.DataFrame([]) would come back with no columns at all.
    condensed_edges = pd.DataFrame(
        [{"from_node": a, "to_node": b, "capacity": row.capacity, "line_ids": row.line_ids}
         for (a, b), row in grouped.iterrows()],
        columns=["from_node", "to_node", "capacity", "line_ids"],
    )
    return condensed_nodes, condensed_edges


def build_network_zones(region, n=10, resolution_km=50, weight="capacity", minkm=5,
                        adjacency_weight=1e-3, verbose=False):
    """Build zones and inter-zone flow for `region` directly from real line
    topology: a fine grid per country, condensed via modularity clustering.

    `n` is zones per country; a country with more disconnected landmasses
    than `n` comes back with more zones instead of merging unrelated ones.
    `adjacency_weight` must stay well below real line capacity (~1MW+) so it
    only breaks ties for cells with no real line.

    Returns (zones, lines, flow, flow_neighbor) -- `zones` matches
    `region.segment_re_zones`'s shape so `set_zone_data()` works unmodified.
    `flow_neighbor` is an empty (0-column) frame; this method doesn't
    yet compute interconnects to neighboring countries, and each modeled
    country is condensed independently (no cross-border merging between them).
    """
    verbose_log("NETWORK_SEGMENT",
                f"Reading lines and building a {resolution_km}km grid per country...", verbose)
    lines = read_line_data(region.grid.path, region, minkm=minkm)
    lines["capacity"] = region.grid._get_line_capacity(lines)

    all_zones = []
    all_edges = []
    for idx in range(len(region.countries)):
        country = region.countries.iloc[[idx]]
        iso = country["ISO_A3"].iloc[0]

        nodes = build_grid_nodes(country, resolution_km=resolution_km)
        country_lines = lines.copy()
        country_lines["nodes"] = country_lines.geometry.apply(
            lambda g: region.grid._get_line_zones(g, nodes)
        )
        country_lines = country_lines[country_lines["nodes"].apply(len) > 0]

        edges = defaultdict(lambda: {"capacity": 0.0, "line_ids": []})
        for _, line in country_lines.iterrows():
            path = line["nodes"]
            for i in range(1, len(path)):
                a, b = sorted((path[i - 1], path[i]))
                edge = edges[(a, b)]
                edge["capacity"] += line["capacity"]
                edge["line_ids"].append(line["id"])
        edges = pd.DataFrame(
            [{"from_node": a, "to_node": b, **data} for (a, b), data in edges.items()],
            columns=["from_node", "to_node", "capacity", "line_ids"],
        )
        # `weight` (clustering signal) starts as a copy of `capacity`; adjacency
        # edges add a small amount on top for pairs with no real line, without
        # touching `capacity` itself, so reported flow never includes the nudge.
        edges["weight"] = edges[weight]
        adjacency_edges = build_adjacency_edges(nodes, country.union_all(), weight=adjacency_weight)
        edges = pd.concat([edges, adjacency_edges], ignore_index=True)

        n_clusters = min(n, len(nodes)) if len(nodes) else 1
        condensed_nodes, condensed_edges = condense_by_modularity(
            nodes, edges, weight="weight", n_clusters=n_clusters
        )
        # Grid cells are whole squares; trim overshoot past the real coastline.
        condensed_nodes["geometry"] = condensed_nodes.geometry.intersection(country.union_all())

        label_map = {i: f"{iso}-{i}" for i in condensed_nodes.index}
        condensed_edges["from_node"] = condensed_edges["from_node"].map(label_map)
        condensed_edges["to_node"] = condensed_edges["to_node"].map(label_map)
        condensed_nodes = condensed_nodes.rename(index=label_map)
        condensed_nodes.index.name = "zone"
        condensed_nodes["country"] = iso
        condensed_nodes["zone_label"] = condensed_nodes.index

        verbose_log("NETWORK_SEGMENT",
                    f"  {iso}: condensed to {len(condensed_nodes)} zones, "
                    f"{len(condensed_edges)} inter-zone links.", verbose)

        all_zones.append(condensed_nodes)
        all_edges.append(condensed_edges)

    zones = gpd.GeoDataFrame(pd.concat(all_zones), geometry="geometry", crs=all_zones[0].crs)

    edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame(
        columns=["from_node", "to_node", "capacity", "line_ids"]
    )
    zidx = zones.index
    flow = pd.DataFrame(np.zeros((len(zidx), len(zidx))), index=zidx, columns=zidx)
    for _, edge in edges.iterrows():
        flow.loc[edge["from_node"], edge["to_node"]] += edge["capacity"]
        flow.loc[edge["to_node"], edge["from_node"]] += edge["capacity"]

    # Interconnects to neighboring countries aren't computed by this method yet.
    flow_neighbor = pd.DataFrame(index=zidx)

    return zones, lines, flow, flow_neighbor
