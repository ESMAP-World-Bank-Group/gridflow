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


class country:
    def __init__(self, name, data_path=""):
        self.name = name
        
        # Load the data for the country
        self.region_data = {
            "pv" : regiondata("pv", data_path + "/pv/pv.tif", "mean"),
            "wind" : regiondata("wind", data_path + "/wind.tif", "mean"),
            "population" : regiondata("population", data_path + "/pop.tif", "sum")}
        # Empty grid
        self.grid = network(data_path + "/grid.gpkg")
        self.grid.country = self
        
        # Empty regions
        self.regions = gpd.GeoDataFrame(geometry=[])
    
    def create_regions(self, n=10, method="pv"):
        if method=="pv":
            # Open the solar potential raster
            pvpath = self.region_data["pv"].path
            ras = rioxarray.open_rasterio(pvpath, masked=True)
        else:
            # Open the wind potential raster
            windpath = self.region_data["wind"].path
            ras = rioxarray.open_rasterio(windpath, masked=True)

        data = ras.sel(band=1).values
        ### Create regions through segmentation
        # Create a nan mask
        mask = ~np.isnan(data)
        # Remove nans
        data[np.isnan(data)] = 0
        seg = slic(data, n_segments=n, compactness=0.01,
                   enforce_connectivity=True, mask=mask)
        
        # Generate region boundaries from raster segmention
        # Use rasterio to extract polygons
        poly = (
            {'properties': {'label': v}, 'geometry': s}
            for s, v in shapes(seg.astype(np.int32), mask=mask, transform=ras.rio.transform())
        )
        gdf = gpd.GeoDataFrame.from_features(list(poly), crs=ras.rio.crs)
        
        ### Generate statistics for each region
        regionstats = pd.DataFrame(index=gdf.index)
        for data in self.region_data.values():
            regionstats = data.get_region_value(gdf, regionstats=regionstats)
        
        gdf = gdf.join(regionstats)
        self.regions = gdf
        
        return gdf
    
    def create_network(self):
        self.grid.create_lines(self.regions)


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
        
    def create_lines(self, regions):
        # Load the list of power lines
        self.lines = oim_reader.read_line_data(self.path)
        # Add columns in the lines dataframe for regions and capacities
        self.lines["regions"] = None
        self.lines["capacity"] = None
        nlines = len(self.lines)
        for i in range(nlines):
            linepath = self.lines.loc[i].geometry
            ridx = self._get_line_regions(linepath, regions)
            self.lines.at[i, "regions"] = ridx
        self.lines["capacity"] = self._get_line_capacity(self.lines)
    
    def create_flow_model(self):
        nregions = len(self.country.regions)
        regidx = self.country.regions.index
        flow_mat = pd.DataFrame(index = regidx, 
                                columns = regidx)
        
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
    
    def _get_line_regions(self, line, regions):
        # This obtains an ordered sequence of regions traversed by a line.
        # Generalizes to non-convex regions and complex line pathways.
        int_points = []
        int_regions = []
        int_dist = []

        onpath = regions[regions.intersects(line)].copy()
        onpath["intersect"] = onpath.geometry.apply(lambda r: r.intersection(line))
        for ridx, intersection in onpath.iterrows():
            seg = intersection.intersect
            if isinstance(seg, MultiLineString):
                for el in seg.geoms:
                    int_regions.append(ridx)
                    int_points.append([Point(el.coords[0]), Point(el.coords[-1])]);
            elif isinstance(seg, LineString):
                int_regions.append(ridx)
                int_points.append([Point(seg.coords[0]), Point(seg.coords[-1])]);

        for pt in int_points:
            int_dist.append(min(line.project(pt[0]), line.project(pt[1])))

        order_regions = [int_regions[idx] for idx in np.argsort(int_dist)]
        return order_regions
        


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
    
    def get_region_value(self, regions, regionstats=None):
        # Iterate through regions dataframe and obtain aggregate data for each region
        ras = rasterio.open(self.path)
        regions = regions.to_crs(ras.crs)
        
        # Create the empty dataframe to store the results
        if regionstats is None:
            regionstats = pd.DataFrame(index=regions.index)
            regionstats[self.name] = 0
        
        for idx in regions.index:
            geom = regions.loc[idx].geometry
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

            regionstats.loc[idx, self.name] = rez
        return regionstats

    
def _clean_raster(ras, nodata):
    if np.isnan(nodata):
        ras = ras[~np.isnan(ras)]
    else:
        ras = ras[ras != nodata]
    return ras
        
        