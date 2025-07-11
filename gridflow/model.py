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
        self.grid.region = self
        
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
        """Create the network flow models for defined subregions.

        After region has been segmented into subregions, we can define
        the routes of lines through subregions and accordingly estimate the
        flow representation of the network. 
        """

        self.grid.create_lines(self.subregions)

    def generate_epm_inputs(self, raw_inputs_path, n_subregions=None):
    """
    Generate EPM inputs with subregion modifications.
    
    This function processes CSV files by:
    1. Replicating data for each subregion
    2. Adding a 'subregion' column
    3. Adding a 'region' column (previous zone)
    
    Parameters
    ----------
    raw_inputs_path : str or Path
        Path to the raw inputs directory containing CSV files
    n_subregions : int, optional
        Number of subregions to create. If None, defaults to 4.
    
    Returns
    -------
    dict
        Dictionary containing processed dataframes for EPM
    """
    import pandas as pd
    from pathlib import Path
    
    raw_inputs_path = Path(raw_inputs_path)
    epm_inputs = {}
    
    # Determine number of subregions
    if n_subregions is not None:
        # User-specified number of subregions
        subregion_names = [f"subregion_{i}" for i in range(n_subregions)]
        print(f"Using user-specified {n_subregions} subregions")
    else:
        # Default to 4 subregions
        n_subregions = 4
        subregion_names = [f"subregion_{i}" for i in range(n_subregions)]
        print(f"No subregions specified, defaulting to {n_subregions}")
    
    # Process all CSV files
    for csv_file in raw_inputs_path.glob('*.csv'):
        print(f"\nProcessing {csv_file.name}...")
        
        # Read the original CSV
        df_original = pd.read_csv(csv_file)
        print(f"  Original shape: {df_original.shape}")
        
        # Check if zone column exists
        if 'zone' in df_original.columns:
            # Get unique zones for region mapping
            zones = df_original['zone'].unique().tolist()
            
            # Create region mapping (previous zone)
            region_map = {}
            for i, zone in enumerate(zones):
                prev_idx = (i - 1) % len(zones)
                region_map[zone] = zones[prev_idx]
            
            # Create list to store replicated dataframes
            dfs_list = []
            
            # Replicate data for each subregion
            for subregion in subregion_names:
                # Create a copy of the original dataframe
                df_copy = df_original.copy()
                
                # Add the subregion column
                df_copy['subregion'] = subregion
                
                # Add the region column
                df_copy['region'] = df_copy['zone'].map(region_map)
                
                # Add to list
                dfs_list.append(df_copy)
            
            # Concatenate all dataframes
            df_final = pd.concat(dfs_list, ignore_index=True)
            
            # Reorder columns to put subregion and region after zone
            cols = df_final.columns.tolist()
            cols.remove('subregion')
            cols.remove('region')
            zone_idx = cols.index('zone')
            cols.insert(zone_idx + 1, 'region')
            cols.insert(zone_idx + 2, 'subregion')
            df_final = df_final[cols]
            
            print(f"  Final shape: {df_final.shape} ({n_subregions} subregions × {len(df_original)} original rows)")
            epm_inputs[csv_file.stem] = df_final
        else:
            # If no zone column, just store original
            print(f"  No 'zone' column found, keeping original data")
            epm_inputs[csv_file.stem] = df_original
    
    return epm_inputs
        
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
        nsubregions = len(self.region.subregions)
        regidx = self.region.subregions.index
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


"""Helper Functions"""

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
    
        
