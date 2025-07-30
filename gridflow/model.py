# Standard packages
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import yaml

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

from gridflow.data_readers import *


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

    def __init__(self, countries, global_data_path):
        self.countries = read_borders(global_data_path + "/borders/WB_GAD_ADM0_complete.shp",
                                      countries)

        # The transmission system -- starts out empty
        self.grid = network(global_data_path + "/grid.gpkg")
        self.grid.region = self
        
        # The zones -- start out empty
        self.zones = gpd.GeoDataFrame(geometry=[])
        self.zone_stats = pd.DataFrame()
        # Define the zonal statistics
        self.zone_stat_specs = {
            "population" : zonedata("population", global_data_path + "/population_2020.tif", "sum")
        }
        
        # Paths to necessary global datasets
        self.global_pv = global_data_path + "/pv.tif"
        self.global_wind = global_data_path + "/wind.tif"

    
    def create_zones(self, n=10, method="pv"):
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
        all_zones = []
        for idx in range(len(self.countries)):
            country = self.countries.iloc[[idx]]

            if method=="pv":
                # Open the solar potential raster
                ras = get_country_raster(country, self.global_pv)
            else:
                # Open the wind potential raster
                ras = get_country_raster(country, self.global_wind)
            # Segment raster
            zones = _segment_raster(ras, n=n)
            # Assign country to zones
            zones["country"] = country["ISO_A3"].iloc[0]
            all_zones.append(zones)
 
        self.zones = gpd.GeoDataFrame(pd.concat(all_zones, ignore_index=True),
                                      geometry="geometry",
                                      crs=all_zones[0].crs).drop(columns=["label"])

    
        ### Generate statistics for each subregion
        # Iterate through datasets, and obtain aggregate subregion statistics
        zstats = pd.DataFrame(index=self.zones.index)
        for zdata in self.zone_stat_specs.values():
            zstats = zdata.get_zone_values(self, outputdf=zstats)
        
        self.zone_stats = zstats
    
    def create_network(self):
        """Create the network flow models for defined subregions.

        After region has been segmented into subregions, we can define
        the routes of lines through subregions and accordingly estimate the
        flow representation of the network. 
        """

        self.grid.create_lines(self.zones)


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
        # The region this network is in
        self.region = None
        # Path to the network data file
        self.path = path
        # No lines between zones yet
        self.lines = None
        # No flow model
        self.flow = None
        
    def create_lines(self, zones, minkm=5):
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
        self.flow = self.get_flow_model()
    
    def get_flow_model(self):
        nzones = len(self.region.zones)
        zidx = self.region.zones.index
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
                    flow_mat.iloc[zpath[i-1], zpath[i]] += capmw
                    flow_mat.iloc[zpath[i], zpath[i-1]] += capmw
        return flow_mat
        
    def _get_line_capacity(self, lines):
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
        self.path = path
        self.agg = agg
        self.name = name
    
    def aggregate(self, data):
        if self.agg=="mean":
            return np.mean(data)
        elif self.agg=="sum":
            return np.sum(data)
    
    def get_zone_values(self, region, outputdf=None):
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
    data = ras.values
    if np.isnan(nodata):
        data = data[~np.isnan(data)]
    else:
        data = data[data != nodata]
    return data
        
def _streamline(seq):
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
        
