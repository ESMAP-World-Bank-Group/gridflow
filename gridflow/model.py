# Standard packages
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from collections import defaultdict

# Raster data packages
from rasterio.features import shapes
from rasterio.mask import mask
from rasterstats import zonal_stats

# For segmenting rasters
from skimage.segmentation import slic
from shapely.geometry import shape, mapping
from shapely.geometry import MultiLineString, LineString, Point
from pyproj import Transformer

from gridflow.data_readers import (
    get_country_raster,
    get_global_dataset_file_path,
    get_global_dataset_url,
    get_global_datasets_path,
    get_neighbors,
    get_zonal_re,
    read_borders,
    read_boundaries,
    read_line_data,
    read_country_demand,
    read_power_plants,
    ctry_to_zone_format
)
from gridflow.utils import verbose_log, directional_zone_labels, centroid_distance_km
from gridflow.network_zones import build_network_zones


class region:
    """
    Top-level class wrapping full modeled system and all parameters.

    This class represents the country or set of countries over which we are
    conducting capacity expansion planning. The region possesses zones --
    the spatial resolution at which the region is being modelled -- and a
    network -- the flow model of the current transmission network.


    Attributes
    ----------
    grid : network object
        Transmission network representation for the region. 

    zone_data : dict
        Contains zonal data.
        1. zone_list - geopandas df with basic zone data
        2. zone_demand - demand profiles by zone
        3. zone_re - renewable profiles by zone
    """

    def __init__(self, countries, include_neighbors=True, global_data_path=None, zone_stats_to_load=None,
                 year=None):
        """
        Parameters
        ----------
        countries : list of str
            ISO3 codes for the countries to model.
        year : int, optional
            The year the model represents. Defaults to the current year.
        global_data_path : str, optional
            Path to the folder containing borders, rasters, and grid data. Defaults
            to `input_data.global_datasets` from :file:`config.yaml`.
        zone_stats_to_load : list of str, optional
            Subset of zone statistics to compute. Defaults to all available stats
            (currently: ["population"]). Pass a list like ["population"] to pick
            specific datasets.
        """
        # The year the model represents
        self.year = date.today().year if year is None else year

        if global_data_path is None:
            global_data_path = get_global_datasets_path()
        self.global_data_path = global_data_path

        borders_path = get_global_dataset_file_path("borders", root=global_data_path)
        self.countries = read_borders(borders_path, countries)
        if include_neighbors is True:
            neighbor_list_path = get_global_dataset_file_path("neighbors", root=global_data_path)
            country_names_path = get_global_dataset_file_path("country", root=global_data_path)
            neighbors_list = get_neighbors(countries, neighbor_list_path, country_names_path)
            self.neighbors = read_borders(borders_path, neighbors_list)
        elif isinstance(include_neighbors, list):
            self.neighbors = read_borders(borders_path, include_neighbors)
        else:
            self.neighbors = gpd.GeoDataFrame(geometry=[])

        # The transmission system -- starts out empty
        grid_path = get_global_dataset_file_path("grid", root=global_data_path)
        self.grid = network(grid_path)
        self.grid.region = self

        # The zones -- start out empty
        self.zones = gpd.GeoDataFrame(geometry=[])

        # Define the zonal statistics; allow users to pick a subset to keep setup simple.
        population_path = get_global_dataset_file_path("population", root=global_data_path)
        gdp_path = get_global_dataset_file_path("gdp", root=global_data_path)
        available_zone_stats = {
            "population": zonedata("population", population_path, "density_sum"),
            "gdp": zonedata("gdp", gdp_path, "mean"),
        }
        if zone_stats_to_load is None:
            selected_stats = list(available_zone_stats.keys())
        else:
            invalid = [z for z in zone_stats_to_load if z not in available_zone_stats]
            if invalid:
                raise ValueError(
                    f"Unknown zone stats {invalid}. "
                    f"Options are {list(available_zone_stats.keys())}."
                )
            selected_stats = zone_stats_to_load

        self.zone_stat_specs = {k: available_zone_stats[k] for k in selected_stats}
        # Empty dataframe for zone statistics
        self.zone_stats = pd.DataFrame()
        # Empty dataframe for zone RE profiles
        self.zone_re = pd.DataFrame()
        # Empty dataframe for zone renewable capacity factors, zones x tech
        self.zone_re_potential = pd.DataFrame()
        # Empty dataframe for power plants assigned to zones
        self.zone_plants = pd.DataFrame()

        # Paths to necessary global datasets
        self.global_pv = get_global_dataset_file_path("pv", root=global_data_path)
        self.global_wind = get_global_dataset_file_path("wind", root=global_data_path)
        self.global_wind_cf = get_global_dataset_file_path("wind_cf", root=global_data_path)
        self.global_admin = get_global_dataset_file_path("admin", root=global_data_path)
        self.global_power_plants = get_global_dataset_file_path("power_plants", root=global_data_path)
        self.global_demand = get_global_dataset_file_path("demand", root=global_data_path)

    def create_zones(self, method="pv", n=10, verbose=False):
        """Define zones by multiple methods.

        method : string
            Name of zoning method to use. Options include:
            admin - Use administrative boundaries as zones
            pv - Segment zones based on pv potential map
            wind - Segment zones based on wind potential map
        """

        if method in ["pv", "wind"]:
            self.zones = self.segment_re_zones(n=n, method=method, verbose=verbose)
            return
        elif method == "admin":
            # load global boundary data
            boundaries = read_boundaries(self.global_admin, self.countries.ISO_A3.to_list())
            zones = boundaries[["shapeName", "shapeGroup", "geometry"]]
            # some of these boundaries may be invalid - let's make them valid
            for idx in zones.index:
                poly = zones.loc[idx, "geometry"]
                if not poly.is_valid:
                    zones.loc[idx, "geometry"] = poly.buffer(0)
            zones = zones.rename({"shapeName" : "zone_name", "shapeGroup" : "country"},
                                 axis = 1)
            zones = zones.reset_index(drop=True)
            # Match pv/wind/network zoning's zone_label/index convention;
            # prefixed with country in case two share an admin unit name.
            zone_labels = [f"{row.country}-{row.zone_name}" for row in zones.itertuples()]
            zones = zones.assign(zone_label=zone_labels)
            zones.index = pd.Index(zone_labels, name="zone")
        else:
            raise ValueError(f"Unknown zoning method '{method}'. Use 'pv', 'wind', or 'admin'.")

        self.zones = zones


    def segment_re_zones(self, n=10, method="pv", verbose=False):
        """Segments region into zones and returns them (does not set self.zones).

        Parameters
        ----------
        n : int
            Number of zones per country to create.

        method : string
            Name of segmentation method to use. Options include:
            pv - Segment zones based on pv potential map
            wind - Segment zones based on wind potential map
        """
        verbose_log("ZONE_SEGMENT", f"Starting segmentation for method '{method}', target {n} zones per country.", verbose)

        all_zones = []
        for idx in range(len(self.countries)):
            country = self.countries.iloc[[idx]]

            if method=="pv":
                # Open the solar potential raster
                ras = get_country_raster(country, self.global_pv)
            elif method=='wind':
                # Open the wind potential raster
                ras = get_country_raster(country, self.global_wind)
            else:
                raise ValueError(
                    f"Unknown segmentation method '{method}'. "
                    "Use 'pv' or 'wind'."
                )
            # Segment raster
            zones = _segment_raster(ras, n=n)
            # Assign country to zones
            zones["country"] = country["ISO_A3"].iloc[0]
            all_zones.append(zones)
 
        zones = gpd.GeoDataFrame(pd.concat(all_zones, ignore_index=True),
                                 geometry="geometry",
                                 crs=all_zones[0].crs).drop(columns=["label"])
        zone_names = directional_zone_labels(zones, country_col="country", verbose=verbose)
        zones = zones.assign(zone_label=zone_names)
        zones.index = pd.Index(zone_names, name="zone")
        country_list = ", ".join(self.countries["ISO_A3"].tolist())
        verbose_log("ZONE_SEGMENT", f"Created {len(zones)} zones for {country_list}.", verbose)
        sample_count = min(5, len(zone_names))
        sample_labels = ", ".join(zone_names[:sample_count])
        extra = f" +{len(zone_names)-sample_count} more" if len(zone_names) > sample_count else ""
        verbose_log(
            "ZONE_LABELS",
            f"Method '{method}' assigned labels (first {sample_count}): {sample_labels}{extra}",
            verbose,
        )
        return zones

    
    def set_zone_data(self, demand_scaleby="population", verbose=False):
        """
        Compute country demand, zonal statistics and demand shares, renewable
        potential and renewables profiles, via `set_country_demand()`,
        `set_zone_demand_shares()`, `set_zone_re_potential()` and
        `set_zone_re()`, which can also be called separately.

        Parameters
        ----------
        demand_scaleby : str
            Zone statistic to distribute demand by (see `set_zone_demand_shares`).
        verbose : bool
            If True, print which datasets are loaded and where they are stored.
        """
        self.set_country_demand(verbose=verbose)
        self.set_zone_demand_shares(scaleby=demand_scaleby, verbose=verbose)
        self.set_zone_re_potential(verbose=verbose)
        self.set_zone_re(verbose=verbose)

    def set_country_demand(self, verbose=False):
        """Add each country's annual demand (TWh, Ember) for `self.year` to
        `self.countries` as `demand_twh`, with the data year used in `demand_year`."""
        isos = self.countries["ISO_A3"].tolist()
        demand = read_country_demand(self.global_demand, isos, self.year)
        missing = sorted(set(isos) - set(demand.index))
        if missing:
            print(f"WARNING: no Ember demand data at or before {self.year} for {missing}.")
        self.countries = self.countries.assign(
            demand_twh=self.countries["ISO_A3"].map(demand["demand_twh"]),
            demand_year=self.countries["ISO_A3"].map(demand["demand_year"]).astype("Int64"),
        )
        for iso, row in demand.iterrows():
            verbose_log("COUNTRY_DEMAND", f"{iso}: {row.demand_twh:.1f} TWh "
                        f"(Ember {row.demand_year:.0f}, model year {self.year})", verbose)

    def set_zone_re_potential(self, verbose=False):
        """Compute each zone's average renewable capacity factor per tech into
        `self.zone_re_potential`: zones as rows, PyPSA technology-data tech
        names (as used in seer) as columns, values always capacity factors.

        solar-utility is the Global Solar Atlas daily PV output (kWh/kWp) / 24;
        onwind is the Global Wind Atlas IEC class 2 capacity factor.

        Parameters
        ----------
        verbose : bool
            If True, print which datasets are loaded.
        """
        specs = [
            zonedata("solar-utility", self.global_pv, "mean", scale=1 / 24),
            zonedata("onwind", self.global_wind_cf, "mean"),
        ]
        potential = pd.DataFrame(index=self.zones.index)
        for zdata in specs:
            verbose_log("ZONE_RE_POTENTIAL", f"Loading '{zdata.name}' from {zdata.path}", verbose)
            potential = zdata.get_zone_values(self, outputdf=potential)
        potential.columns.name = "tech"
        self.zone_re_potential = potential
        verbose_log("ZONE_RE_POTENTIAL", f"Saved {list(potential.columns)} to region.zone_re_potential", verbose)

    def set_zone_demand_shares(self, scaleby="population", verbose=False):
        """Compute zonal statistics into `self.zone_stats`, plus a
        `demand_share` column: each zone's share of its country's demand,
        proportional to `scaleby`. Shares sum to 1 within each country.

        Parameters
        ----------
        scaleby : str
            Zone statistic to distribute demand by; must be one of the
            loaded `zone_stat_specs` (e.g. "population").
        verbose : bool
            If True, print which datasets are loaded and where they are stored.
        """
        if scaleby not in self.zone_stat_specs:
            raise ValueError(
                f"Can't distribute demand by '{scaleby}': it isn't a loaded zone stat. "
                f"Loaded: {list(self.zone_stat_specs.keys())}."
            )
        # Iterate through datasets, and obtain aggregate subregion statistics
        zstats = pd.DataFrame(index=self.zones.index)
        for zdata in self.zone_stat_specs.values():
            verbose_log("ZONE_STATS", f"Loading zone stat '{zdata.name}' from {zdata.path}", verbose)
            zstats = zdata.get_zone_values(self, outputdf=zstats)

        country_total = zstats[scaleby].groupby(self.zones["country"]).transform("sum")
        zstats["demand_share"] = zstats[scaleby] / country_total

        self.zone_stats = zstats
        verbose_log("ZONE_STATS", f"Saved stats with columns: {list(self.zone_stats.columns)}", verbose)

    def set_zone_re(self, verbose=False):
        """Fetch renewables profiles for each zone from renewables.ninja into
        `self.zone_re`, one column per zone.

        Parameters
        ----------
        verbose : bool
            If True, print each zone as it's queried.
        """
        verbose_log("ZONE_RE", "Loading renewables profiles for each zone from renewables.ninja (default: pv).", verbose)
        transformer = Transformer.from_crs("EPSG:3857", self.zones.crs, always_xy=True)
        for zidx in range(len(self.zones)):
            zone = self.zones.iloc[[zidx]]
            country_iso = zone["country"].iloc[0] if "country" in zone else "?"
            zone_proj = zone.to_crs(epsg=3857)
            zone_area = zone_proj.geometry.area.iloc[0]
            centroid_proj = zone_proj.geometry.centroid.iloc[0]
            centroid_lonlat = Point(transformer.transform(centroid_proj.x, centroid_proj.y))
            verbose_log(
                "ZONE_RE",
                f"  - Querying zone {zidx} ({country_iso}) area {zone_area:.2f}, centroid ({centroid_lonlat.x:.3f}, {centroid_lonlat.y:.3f}).",
                verbose,
            )
            if zidx == 0:
                self.zone_re = get_zonal_re(zone)
            else:
                self.zone_re = pd.concat([self.zone_re, get_zonal_re(zone)], axis=1)
        verbose_log("ZONE_RE", "Saved renewables profiles to region.zone_re", verbose)

    def set_zone_plants(self, max_distance_km=50, verbose=False):
        """Assign the region's power plants to zones and store them in
        `self.zone_plants`, one row per plant with a `zone` column.

        Plants inside a zone are assigned to it; plants outside every zone
        (offshore wind, coastal plants, plants just past a trimmed border) go
        to the nearest zone within `max_distance_km`, and are dropped with a
        warning beyond that.

        Parameters
        ----------
        max_distance_km : float
            Furthest a plant outside all zones may be from its assigned zone.
        verbose : bool
            If True, print how plants were assigned.
        """
        countries = self.countries["ISO_A3"].tolist()
        plants = read_power_plants(self.global_power_plants, countries,
                                   url=get_global_dataset_url("power_plants"))
        verbose_log("ZONE_PLANTS", f"Loaded {len(plants)} plants for {', '.join(countries)} "
                    f"from {self.global_power_plants}", verbose)

        zones = self.zones[["geometry"]].rename_axis("zone").reset_index().to_crs(plants.crs)
        inside = gpd.sjoin(plants, zones, how="left", predicate="within")
        zone = inside.loc[~inside.index.duplicated(), "zone"]

        outside = zone.isna()
        if outside.any():
            # Distances need a metric CRS; a local UTM zone keeps them accurate.
            utm = plants.estimate_utm_crs()
            nearest = gpd.sjoin_nearest(plants[outside].to_crs(utm), zones.to_crs(utm),
                                        how="left", max_distance=max_distance_km * 1e3)
            nearest = nearest.loc[~nearest.index.duplicated(), "zone"]
            zone = zone.fillna(nearest)
            verbose_log("ZONE_PLANTS", f"Assigned {nearest.notna().sum()} plants "
                        f"outside all zones to the nearest zone.", verbose)

        dropped = zone.isna()
        if dropped.any():
            print(f"WARNING: {dropped.sum()} plants ({plants.loc[dropped, 'Capacity'].sum():.0f} MW) "
                  f"are more than {max_distance_km} km from any zone and were dropped.")

        self.zone_plants = plants[~dropped].assign(zone=zone[~dropped])
        verbose_log("ZONE_PLANTS", f"Saved {len(self.zone_plants)} plants to region.zone_plants", verbose)

    def zone_capacity(self, status="operating", by="tech"):
        """Return installed capacity (MW) per zone, with zones as rows and
        `by` (e.g. "tech", "fuel") as columns.

        Parameters
        ----------
        status : str, list of str, or None
            Plant statuses to include (e.g. "operating", "construction").
            None includes all.
        by : str
            Column of `self.zone_plants` to split capacity by.
        """
        plants = self.zone_plants
        if status is not None:
            plants = plants[plants["Status"].isin(np.atleast_1d(status))]
        return (plants.pivot_table(index="zone", columns=by, values="Capacity",
                                   aggfunc="sum", fill_value=0)
                      .reindex(self.zones.index, fill_value=0))


    def create_network(self, method=None, n=10, resolution_km=50, weight="capacity", minkm=5,
                       adjacency_weight=1e-3, min_km2=1000, verbose=False):
        """Create the network flow model for the region.

        With no `method`, assumes zones already exist (e.g. via `create_zones()`)
        and just computes flow between them. Passing `method` makes this a
        one-call alternative that (re)defines zones too, so `set_zone_data()`
        still works afterward either way.

        method : {"pv", "wind", "admin", "network", None}
            "pv"/"wind"/"admin": delegates to `create_zones(method, n)`, then
            computes flow the existing way. "network": zones and capacities are
            both derived from the real line topology instead (see
            `gridflow.network_zones`); `n` is target zones per country, and
            `resolution_km`/`weight`/`minkm`/`adjacency_weight`/`min_km2`
            only apply here.
            None (default): use whatever is already in `self.zones`.
        """
        if method == "network":
            self.zones, self.grid.lines, self.grid.flow, self.grid.flow_neighbor = build_network_zones(
                self, n=n, resolution_km=resolution_km, weight=weight, minkm=minkm,
                adjacency_weight=adjacency_weight, min_km2=min_km2,
                verbose=verbose
            )
            self.grid.set_link_lengths(self.zones)
            return

        if method is not None:
            self.create_zones(method=method, n=n, verbose=verbose)

        zones = self.zones.assign(type="region-zone")
        neigh = ctry_to_zone_format(self.neighbors).assign(type="region-neighbor")
        self.grid.create_lines(pd.concat([zones, neigh]))
        self.grid.set_link_lengths(self.zones, neigh)


class network:
    """
    Holds network data for the full region. 

    After subregions have been defined, this class' methods will determine
    the subregions traversed by individual lines and then aggregate individual
    lines into a flow model that captures total capacities between subregions.

    Attributes
    ----------
    region : region object
       The region to which this network belongs. 

    path : str
       Path to network data from open infrastructure maps.

    lines : GeoDataFrame
       Set of lines from open infrastructure maps.

    flow : numpy matrix
       A square matrix representing the flow model. Dimensions are number of
       subregions; values are estimated total interconnect capacity between
       regions.

    length : DataFrame
       Same shape as `flow`; centroid-to-centroid distance in km between zones.
       Use entries where `flow` > 0 as the link lengths.

    """

    def __init__(self, path):
        """Initialize network metadata."""
        # The region this network is in
        self.region = None
        # Path to the network data file
        self.path = path
        # No lines between zones yet
        self.lines = None
        # No flow model
        self.flow = None
        # No flow model to neighbors
        self.flow_neighbor = None
        # No link lengths (km) yet; same shapes as flow / flow_neighbor
        self.length = None
        self.length_neighbor = None

    def set_link_lengths(self, zones, neighbors=None):
        """Set centroid-to-centroid km distances for zone-zone and zone-neighbor pairs."""
        self.length = centroid_distance_km(zones.geometry, zones.geometry)
        has_neigh = neighbors is not None and len(neighbors) > 0
        self.length_neighbor = (centroid_distance_km(zones.geometry, neighbors.geometry)
                                if has_neigh else pd.DataFrame(index=zones.index))

    def create_lines(self, zones, minkm=5):
        """Assign each power line to zone sequences and compute the flow model."""
        # Load the list of power lines - filter short lines
        self.lines = read_line_data(self.path, self.region, minkm=5)
        
        # Add columns in the lines dataframe for regions and capacities
        self.lines["zones"] = None
        self.lines["capacity"] = None

        # Determine the sequence of zones the line traverses
        nlines = len(self.lines)
        for i in range(nlines):
            linepath = self.lines.loc[i].geometry
            zidx = self._get_line_zones(linepath, zones)
            self.lines.at[i, "zones"] = zidx

        # Get the capacity of the line
        self.lines["capacity"] = self._get_line_capacity(self.lines)
        # Create the flow model representation of the network
        # for the lines created
        self.flow, self.flow_neighbor = self.get_flow_model(zones)
    
    def get_flow_model(self, zones):
        """Build the symmetric flow matrix from inter-zone lines."""
        nzones = len(zones)
        zidx = zones.index
        flow_mat = pd.DataFrame(data=np.zeros([nzones, nzones]),
                                index = zidx, 
                                columns = zidx)
        for _, line in self.lines.iterrows():
            zpath = line.zones
            npath = len(zpath)
            capmw = line.capacity
            if npath > 1:
                for i in range(1, npath):
                    # Add capacity to (both) of the corresponding entries
                    # in the flow model matrix
                    from_zone = zpath[i - 1]
                    to_zone = zpath[i]
                    flow_mat.loc[from_zone, to_zone] += capmw
                    flow_mat.loc[to_zone, from_zone] += capmw
        
        # split the flow model into zones and neighbor zones.
        zones_idxs = zones[zones["type"] == "region-zone"].index
        neigh_idxs = zones[zones["type"] == "region-neighbor"].index
        
        zone_flow = flow_mat.loc[zones_idxs, zones_idxs]
        neigh_flow = flow_mat.loc[zones_idxs, neigh_idxs]

        return zone_flow, neigh_flow
        
    def _get_line_capacity(self, lines):
        """Estimate per-line capacities using surge impedance loading heuristics."""
        # This is a preliminary model mapping line parameters from openinframaps
        # to line capacities. This model could be significantly enhanced based on
        # engineering rules of thumb or operational data.

        lines = lines.to_crs(epsg=3857)
        length = lines.geometry.length / 1e3
        z = pd.Series(name="z", index=lines.index)
        z.loc[lines.index[lines.cables <= 1]] = 400
        z.loc[lines.index[lines.cables > 1]] = 300
        
        # Surge Impedance Loading in MW
        sil_mw = (lines.voltage**2 / z) / 1e6
        return sil_mw
    
    def _get_line_zones(self, line, zones):
        """Return an ordered, deduplicated list of zone ids crossed by the line."""
        # This obtains an ordered sequence of zones traversed by a line.
        # Generalizes to non-convex regions and complex line pathways.
        int_points = []
        int_zones = []
        int_dist = []

        # Use the zones' spatial index to narrow down to candidates whose bounding
        # box could intersect the line before running the exact intersects test,
        # instead of testing every zone.
        candidate_pos = zones.sindex.query(line, predicate="intersects")
        onpath = zones.iloc[candidate_pos].copy()
        onpath["intersect"] = onpath.geometry.apply(lambda r: r.intersection(line))
        for zidx, intersection in onpath.iterrows():
            seg = intersection.intersect
            if isinstance(seg, MultiLineString):
                for el in seg.geoms:
                    int_zones.append(zidx)
                    int_points.append([Point(el.coords[0]), Point(el.coords[-1])]);
            elif isinstance(seg, LineString):
                int_zones.append(zidx)
                int_points.append([Point(seg.coords[0]), Point(seg.coords[-1])]);

        for pt in int_points:
            int_dist.append(min(line.project(pt[0]), line.project(pt[1])))

        order_zones = [int_zones[idx] for idx in np.argsort(int_dist)]
        # [r1, r1, r3, r5, r5, r1] => [r1, r3, r5, r1]
        order_zones = _streamline(order_zones)
        return order_zones


class zonedata:
    def __init__(self, name, path, agg, scale=1.0):
        """Describe a raster-based zone statistic.

        `scale` multiplies raster values before aggregating (e.g. 1/24 turns
        daily kWh/kWp into a capacity factor).
        """
        self.path = path
        self.agg = agg
        self.name = name
        self.scale = scale

    def aggregate(self, data):
        """Aggregate raster values according to the configured function.

        "density_sum" expects `data` already multiplied by pixel area (see
        `get_zone_values`), turning a per-km² density into a total.
        """
        data = data * self.scale
        if self.agg=="mean":
            return np.mean(data)
        elif self.agg in ("sum", "density_sum"):
            return np.sum(data)
    
    def get_zone_values(self, region, outputdf=None):
        """Aggregate the raster statistics per zone and append to `outputdf`."""
        # Read in the segment of the raster corresponding to the region
        ras = get_country_raster(region.countries, self.path)
        zones = region.zones.to_crs(ras.rio.crs)
        
        # Create dataframe to store the results -- intialize with zeros
        if outputdf is None:
            outputdf= pd.DataFrame(index=zones.index)
            outputdf[self.name] = 0
        
        for idx in zones.index:
            geom = zones.loc[idx].geometry

            # Clip the raster to the zone geometry
            zoneras = ras.rio.clip([geom], zones.crs)
            # Get valid raster values
            valid = _get_ras_mask(zoneras, zoneras.rio.nodata)
            data = zoneras.values[valid]
            if self.agg == "density_sum":
                data = data * _pixel_area_km2(zoneras)[valid]
            # Aggregate the data and fill in the results dataframe
            rez = self.aggregate(data) if data.size > 0 else np.nan
            outputdf.loc[idx, self.name] = rez
        return outputdf


"""Helper Functions"""

def _get_ras_mask(ras, nodata):
    """Boolean mask of valid (non-nodata) raster cells."""
    data = ras.values
    if pd.isna(nodata):
        return ~pd.isna(data)
    try:
        return data != nodata
    except TypeError:
        return pd.Series(data.ravel()).ne(nodata).values.reshape(data.shape)

def _pixel_area_km2(ras):
    """Area of each raster cell in km², broadcast to the raster's shape.

    In a geographic CRS, cells shrink with latitude, so area is computed
    per row on a sphere; in a projected CRS, every cell has the same area.
    """
    res_x, res_y = (abs(r) for r in ras.rio.resolution())
    if ras.rio.crs.is_geographic:
        earth_radius_km = 6371.0088
        lat = np.radians(ras.y.values)
        half = np.radians(res_y) / 2
        row_area = earth_radius_km**2 * np.radians(res_x) * (np.sin(lat + half) - np.sin(lat - half))
        area = np.broadcast_to(row_area[:, None], ras.shape[-2:])
    else:
        area = np.full(ras.shape[-2:], res_x * res_y / 1e6)
    return np.broadcast_to(area, ras.shape)
        
def _streamline(seq):
    """Collapse consecutive duplicates while preserving order."""
    if len(seq) == 0:
        return seq
    else:
        slseq = [seq[0]]
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]:
                slseq.append(seq[i])
    return slseq

def _segment_raster(ras, n=10):
    """Segment a raster into n segments"""
    data = ras.sel(band=1).values
    ### Create subregions through segmentation
    # Create a nan mask
    mask = ~np.isnan(data)
    # Replace nans by means
    data[np.isnan(data)] = np.nanmean(data)
    seg = slic(data, n_segments=n, compactness=0.1,
               enforce_connectivity=True, mask=mask, channel_axis=None)

    # Generate subregion boundaries from raster segmention
    # Use rasterio to extract polygons
    poly = (
        {'properties': {'label': v}, 'geometry': s}
        for s, v in shapes(seg.astype(np.int32), mask=mask, transform=ras.rio.transform())
    )
    gdf = gpd.GeoDataFrame.from_features(list(poly), crs=ras.rio.crs)
    gdf = gdf.dissolve(by="label", as_index=False)
    return gdf
