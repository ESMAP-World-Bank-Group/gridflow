# Standard packages
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import yaml
from collections import defaultdict

# Raster data packages
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import rioxarray
from rasterstats import zonal_stats

# For segmenting rasters
from skimage.segmentation import slic
from shapely.geometry import shape, mapping
from shapely.geometry import MultiLineString, LineString, Point
from pyproj import Transformer

from gridflow.data_readers import (
    get_country_raster,
    get_global_dataset_file_path,
    get_global_datasets_path,
    get_neighbors,
    get_zonal_re,
    read_borders,
    read_line_data,
    ctry_to_zone_format
)
from gridflow.utils import verbose_log, directional_zone_labels


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

    def __init__(self, countries, include_neighbors=True, global_data_path=None, zone_stats_to_load=None):
        """
        Parameters
        ----------
        countries : list of str
            ISO3 codes for the countries to model.
        global_data_path : str, optional
            Path to the folder containing borders, rasters, and grid data. Defaults
            to `input_data.global_datasets` from :file:`config.yaml`.
        zone_stats_to_load : list of str, optional
            Subset of zone statistics to compute. Defaults to all available stats
            (currently: ["population"]). Pass a list like ["population"] to pick
            specific datasets.
        """
        if global_data_path is None:
            global_data_path = get_global_datasets_path()
        self.global_data_path = global_data_path

        borders_path = get_global_dataset_file_path(
            "borders", "borders/WB_GAD_ADM0_complete.shp", root=global_data_path
        )
        self.countries = read_borders(borders_path, countries)
        if include_neighbors is True:
            self.neighbors = read_borders(borders_path, get_neighbors(countries))
        elif isinstance(include_neighbors, list):
            self.neighbors = read_borders(borders_path, include_neighbors)
        else:
            self.neighbors = gpd.GeoDataFrame(geometry=[])

        # The transmission system -- starts out empty
        grid_path = get_global_dataset_file_path("grid", "grid_sample.gpkg", root=global_data_path)
        self.grid = network(grid_path)
        self.grid.region = self
        
        # The zones -- start out empty
        self.zones = gpd.GeoDataFrame(geometry=[])

        # Define the zonal statistics; allow users to pick a subset to keep setup simple.
        population_path = get_global_dataset_file_path(
            "population", "population_2020.tif", root=global_data_path
        )
        gdp_path = get_global_dataset_file_path(
            "gdp", "GDP2005_1km.tif", root=global_data_path
        )
        available_zone_stats = {
            "population": zonedata("population", population_path, "sum"),
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
        
        # Paths to necessary global datasets
        self.global_pv = get_global_dataset_file_path("pv", "pv.tif", root=global_data_path)
        self.global_wind = get_global_dataset_file_path("wind", "wind.tif", root=global_data_path)

    
    def create_zones(self, n=10, method="pv", verbose=False):
        """Segments region into zones. 

        Number of zones is specified by modeller. Eventually will
        support multiple segmentation methods of varied complexity. 

        Parameters
        ----------
        n : int
            Number of zones per country to create.

        method : string
            Name of segmentation method to use. Options include:
            pv - Segment based on pv potential map
            wind - Segment based on wind potential map
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
 
        self.zones = gpd.GeoDataFrame(pd.concat(all_zones, ignore_index=True),
                                      geometry="geometry",
                                      crs=all_zones[0].crs).drop(columns=["label"])
        zone_names = directional_zone_labels(self.zones, country_col="country", verbose=verbose)
        self.zones = self.zones.assign(zone_label=zone_names)
        self.zones.index = pd.Index(zone_names, name="zone")
        country_list = ", ".join(self.countries["ISO_A3"].tolist())
        verbose_log("ZONE_SEGMENT", f"Created {len(self.zones)} zones for {country_list}.", verbose)
        sample_count = min(5, len(zone_names))
        sample_labels = ", ".join(zone_names[:sample_count])
        extra = f" +{len(zone_names)-sample_count} more" if len(zone_names) > sample_count else ""
        verbose_log(
            "ZONE_LABELS",
            f"Method '{method}' assigned labels (first {sample_count}): {sample_labels}{extra}",
            verbose,
        )

    
    def set_zone_data(self, verbose=False):
        """
        Compute zonal statistics and renewables profiles.

        Parameters
        ----------
        verbose : bool
            If True, print which datasets are loaded and where they are stored.
        """
        ### Generate statistics for each subregion
        # Iterate through datasets, and obtain aggregate subregion statistics
        zstats = pd.DataFrame(index=self.zones.index)
        for zdata in self.zone_stat_specs.values():
            verbose_log("ZONE_STATS", f"Loading zone stat '{zdata.name}' from {zdata.path}", verbose)
            zstats = zdata.get_zone_values(self, outputdf=zstats)
        
        self.zone_stats = zstats
        verbose_log("ZONE_STATS", f"Saved stats with columns: {list(self.zone_stats.columns)}", verbose)

        # Get renewables profiles for each zone
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


    def create_network(self):
        """Create the network flow models for defined subregions.

        After region has been segmented into subregions, we can define
        the routes of lines through subregions and accordingly estimate the
        flow representation of the network. 
        """
        zones = self.zones.assign(type="region-zone")
        neigh = ctry_to_zone_format(self.neighbors).assign(type="region-neighbor")
        self.grid.create_lines(pd.concat([zones, neigh]))

    def get_neighbor_capacities(self, verbose=False):
        """Return the per-country neighbor capacity map collected from the network."""
        return self.grid.get_neighbor_capacities(verbose=verbose)


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

        onpath = zones[zones.intersects(line)].copy()
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
    def __init__(self, name, path, agg):
        """Describe a raster-based zone statistic."""
        self.path = path
        self.agg = agg
        self.name = name
    
    def aggregate(self, data):
        """Aggregate raster values according to the configured function."""
        if self.agg=="mean":
            return np.mean(data)
        elif self.agg=="sum":
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
            data = _get_ras_values(zoneras, zoneras.rio.nodata)
            # Aggregate the data and fill in the results dataframe
            rez = self.aggregate(data) if data.size > 0 else np.nan
            outputdf.loc[idx, self.name] = rez
        return outputdf


"""Helper Functions"""

def _get_ras_values(ras, nodata):
    """Extract valid raster cells, excluding nodata."""
    data = ras.values
    if data.size == 0:
        return data
    if pd.isna(nodata):
        mask = ~pd.isna(data)
    else:
        try:
            mask = data != nodata
        except TypeError:
            mask = pd.Series(data.ravel()).ne(nodata).values.reshape(data.shape)
    return data[mask]
        
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
