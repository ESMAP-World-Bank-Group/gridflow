# Standard packages
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

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

from gridflow.data_readers import oim_reader


class region:
    """
    Top-level class wrapping full modeled system and all parameters.

    This class represents the country or set of countries over which we are
    conducting capacity expansion planning. The region posses subregions --
    the spatial resolution at which the region is being modelled -- and a
    network -- the flow model of the current transmission network.


    Attributes
    ----------
    region_data : dict
        Contains spatial data for the country; used to define subregions,
        disaggregate load, etc.

    grid : network object
        Transmission network representation for the region. 

    subregions : GeoDataFrame
        Dataframe containing list of subregions within region.
    """

    def __init__(self, name, data_path=""):
        self.name = name
        
        # Define paths of necessary spatial datasets for the region
        self.region_data = {
            "pv" : regiondata("pv", data_path + "/pv/pv.tif", "mean"),
            "wind" : regiondata("wind", data_path + "/wind.tif", "mean"),
            "population" : regiondata("population", data_path + "/pop.tif", "sum")}

        # The transmission system -- starts out empty
        self.grid = network(data_path + "/grid.gpkg")
        self.grid.country = self
        
        # The subregions -- start out empty
        self.subregions = gpd.GeoDataFrame(geometry=[])
    
    def create_subregions(self, n=10, method="pv"):
        """Segments region into subregions. 

        Number of subregions is specified by modeller. Eventually will
        support multiple segmentation methods of varied complexity. 

        Parameters
        ----------
        n : int
            Number of subregions to create.

        method : string
            Name of segmentation method to use. Options include:
            pv - Segment based on pv potential map
            wind - Segment based on wind potential map
        """

        if method=="pv":
            # Open the solar potential raster
            pvpath = self.region_data["pv"].path
            ras = rioxarray.open_rasterio(pvpath, masked=True)
        else:
            # Open the wind potential raster
            windpath = self.region_data["wind"].path
            ras = rioxarray.open_rasterio(windpath, masked=True)

        data = ras.sel(band=1).values
        ### Create subregions through segmentation
        # Create a nan mask
        mask = ~np.isnan(data)
        # Remove nans
        data[np.isnan(data)] = 0
        seg = slic(data, n_segments=n, compactness=0.01,
                   enforce_connectivity=True, mask=mask)
        
        # Generate subregion boundaries from raster segmention
        # Use rasterio to extract polygons
        poly = (
            {'properties': {'label': v}, 'geometry': s}
            for s, v in shapes(seg.astype(np.int32), mask=mask, transform=ras.rio.transform())
        )
        gdf = gpd.GeoDataFrame.from_features(list(poly), crs=ras.rio.crs)
        
        ### Generate statistics for each subregion
        # Iterate through datasets, and obtain aggregate subregion statistics
        subregionstats = pd.DataFrame(index=gdf.index)
        for data in self.region_data.values():
            subregionstats = data.get_subregion_value(gdf, outputdf=subregionstats)
        
        gdf = gdf.join(subregionstats)
        self.subregions = gdf
        
        return gdf
    
    def create_network(self):
        self.grid.create_lines(self.subregions)


class network:
    def __init__(self, path):
        # The country this network is in
        self.country = None
        # Path to the network data file
        self.path = path
        # No lines between regions created
        self.lines = None
        # No flow model
        self.flow = None
        
    def create_lines(self, subregions):
        # Load the list of power lines
        self.lines = oim_reader.read_line_data(self.path)
        # Add columns in the lines dataframe for regions and capacities
        self.lines["subregions"] = None
        self.lines["capacity"] = None
        nlines = len(self.lines)
        # Determine the sequence of subregions the line traverses
        for i in range(nlines):
            linepath = self.lines.loc[i].geometry
            ridx = self._get_line_regions(linepath, subregions)
            self.lines.at[i, "subregions"] = ridx
        # Get the capacity of the line
        self.lines["capacity"] = self._get_line_capacity(self.lines)
        # Create the flow model representation of the network
        # for the lines created
        self.flow = self.get_flow_model()
    
    def get_flow_model(self):
        nsubregions = len(self.country.subregions)
        regidx = self.country.subregions.index
        flow_mat = pd.DataFrame(data=np.zeros([nsubregions, nsubregions]),
                                index = regidx, 
                                columns = regidx)
        for _, line in self.lines.iterrows():
            rpath = line.subregions
            npath = len(rpath)
            capmw = line.capacity
            if npath > 1:
                for i in range(1, npath):
                    flow_mat.iloc[rpath[i-1], rpath[i]] += capmw
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
    
    def _get_line_regions(self, line, subregions):
        # This obtains an ordered sequence of regions traversed by a line.
        # Generalizes to non-convex regions and complex line pathways.
        int_points = []
        int_subregions = []
        int_dist = []

        onpath = subregions[subregions.intersects(line)].copy()
        onpath["intersect"] = onpath.geometry.apply(lambda r: r.intersection(line))
        for ridx, intersection in onpath.iterrows():
            seg = intersection.intersect
            if isinstance(seg, MultiLineString):
                for el in seg.geoms:
                    int_subregions.append(ridx)
                    int_points.append([Point(el.coords[0]), Point(el.coords[-1])]);
            elif isinstance(seg, LineString):
                int_subregions.append(ridx)
                int_points.append([Point(seg.coords[0]), Point(seg.coords[-1])]);

        for pt in int_points:
            int_dist.append(min(line.project(pt[0]), line.project(pt[1])))

        order_subregions = [int_subregions[idx] for idx in np.argsort(int_dist)]
        # [r1, r1, r3, r5, r5, r1] => [r1, r3, r5, r1]
        order_subregions = _streamline(order_subregions)
        return order_subregions
        


class regiondata:
    def __init__(self, name, path, agg):
        self.path = path
        self.agg = agg
        self.name = name
    
    def aggregate(self, data):
        if self.agg=="mean":
            return np.mean(data)
        elif self.agg=="sum":
            return np.sum(data)
    
    def get_subregion_value(self, subregions, outputdf=None):
        # Iterate through regions dataframe and obtain aggregate data for each region
        ras = rasterio.open(self.path)
        subregions = subregions.to_crs(ras.crs)
        
        # Create dataframe to store the results -- intialize with zeros
        if outputdf is None:
            outputdf= pd.DataFrame(index=subregions.index)
            outputdf[self.name] = 0
        
        for idx in subregions.index:
            geom = subregions.loc[idx].geometry
            try:
                # Mask the raster based on the region geometry
                out_ras, out_transform = mask(ras, [mapping(geom)], crop=True)
                # Get valid raster values
                data = out_ras[0]
                data = _clean_raster(data, ras.nodata)
                # Aggregate the data and fill in the results dataframe
                rez = self.aggregate(data) if data.size > 0 else np.nan
            except Exception as e:
                rez = np.nan
                #warnings.warn("Error when getting region statistics")

            outputdf.loc[idx, self.name] = rez
        return outputdf

    
def _clean_raster(ras, nodata):
    if np.isnan(nodata):
        ras = ras[~np.isnan(ras)]
    else:
        ras = ras[ras != nodata]
    return ras
        
def _streamline(seq):
    if len(seq) == 0:
        return seq
    else:
        slseq = [seq[0]]
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]:
                slseq.append(seq[i])
    return slseq
    
        