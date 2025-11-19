# gridflow overview

## Purpose
`gridflow` builds a spatially aware network-flow representation of a power system so it can feed into the World Bank’s EPM expansion and market modeling framework. The project couples raw geospatial inputs with segmentation, line parsing, and data transformation utilities to move from borders and rasters to zonal statistics, a simplified transmission matrix, and EPM-ready CSVs.

## Core code components

### `gridflow.model`
- `region`: top-level object that ingests ISO-3 country borders, links to a `network`, and holds the zonal geometry + metadata. `create_zones` segments each country on a PV or wind potential raster and caches the resulting GeoDataFrame. `set_zone_data` computes zonal statistics and renewable profiles from the raster data and `get_zonal_re`.
- `network`: once zones exist, `create_lines` pulls OpenInfraMaps line data, filters and projects it, scopes each line to the ordered sequence of zones it crosses, estimates per-line capacities, and assembles the symmetric flow matrix (`flow`) that EPM can treat as aggregated inter-zone transmission.
- `zonedata` helper: parameterized aggregator for rasters (e.g., population) with `get_zone_values` that clips datasets to each zone and applies mean/sum aggregations.
- segmentation helpers: `_segment_raster` uses `skimage.slic` to carve the potential maps, `_get_line_zones` determines the ordered zone path for a line, and `_get_line_capacity` uses simplified SIL-based heuristics to turn line geometry into MW.

### `gridflow.data_readers`
- Reads from the bundled global datasets: country borders, power lines (`grid.gpkg`), and large raster stacks (`pv.tif`, `wind.tif`, `population_2020.tif`). The readers clip rasters and lines to the modeling region, infer CRS-bounded bounding boxes, and clean/massage the line attributes (`circuits`, `cables`, `max_voltage`).
- `get_zonal_re` / `get_reninja_data`: sample random points inside each zone (via `utils.get_random_points`) and fetch hourly PV or wind time series from Renewables.ninja; API key stored in `config.yaml`.
- `get_config_data`: helper to read credentials from the root config file.

### `gridflow.epm_input_generator`
- Transforms raw EPM CSVs from `data/epm_inputs_raw` into zonal inputs that mirror the split defined by `region.zones`. `generate_epm_inputs` orchestrates `zone_replicate` (duplicate country-level tables per zone) and `zone_distribute` (scale values like annual demand by zone population share) before copying over untouched files.
- `_copy_remaining_inputs` is a safety net that detects CSVs still containing zonal columns but not explicitly processed and copies non-zonal tables verbatim.

### `gridflow.utils`
- `get_random_points`: creates evenly distributed sample points inside a zone polygon by projecting into EPSG:32633, sampling within the buffered bounds, and filtering for containment.
- `get_bb`: returns buffered bounding boxes in a desired CRS.
- `country_code_map`: simple ISO3-to-name map backed by `data/global_datasets/country_names.csv`.

## Supporting files
- `config.yaml`: defines what countries to model by default (`grid_params.countries`), where to find the `global_datasets`, and holds the `"renewables_ninja"` API key placeholder.
- `requirements.txt`: pinned dependencies (e.g., geopandas, rasterio, skimage) matched to Python 3.9; see README instructions for installing GDAL/Rtree/PyProj first.
- `README.md`: project overview plus runtime instructions for segmentation & network construction via `gridflow.model.region`.
- `DemoNotebook.ipynb`: interactive notebook demonstrating how to build a region, create zones, and inspect outputs.

## Data structure

Refer to `data/global_datasets/data_documentation.txt` for the detailed provenance, expected formats, and download links noted in this overview.

- `data/global_datasets/`: contains the sample inputs that mirror the World Bank’s standard dataset layout:
  - `borders/WB_GAD_ADM0_complete.*`: shapefile components that provide ADM0 country boundaries for segmentation and region clipping.
  - `grid.gpkg`: OpenInfraMap transmission lines filtered to the target region with layer `power_line`.
  - `pv.tif`, `wind.tif`, `population_2020.tif`: global raster layers used for zone segmentation, renewable profiles, and population statistics.
  - `country_names.csv`: ISO3-to-country name mapping referenced by the EPM generator.
  - `data_documentation.txt`: authoritative dataset descriptions and download sources.
- `data/epm_inputs_raw/`: template CSV folders (`load/`, `supply/`, `constraint/`, etc.) that get processed into `output_base_dir` by `generate_epm_inputs`.

### Required dataset table

| Dataset | Description | Format / Notes |
| --- | --- | --- |
| `wind.tif` | Global Wind Atlas power density used for wind-based segmentation | GeoTIFF raster (1 km resolution), derived from `power_density_cog_50m.tif` |
| `pv.tif` | Global Solar Atlas PV potential for PV-based segmentation | GeoTIFF raster, 1 km resolution |
| `borders/WB_GAD_ADM0_complete.*` | Official World Bank country boundaries (ADM0) | ESRI shapefile bundle (SHP, SHX, DBF, PRJ, CPG) |
| `population_2020.tif` | Gridded 2020 population density | GeoTIFF raster (people per km²) |
| `grid.gpkg` | OpenInfraMap transmission lines | GeoPackage with `power_line` layer |
| `country_names.csv` | ISO2/ISO3 to country-name mapping | CSV (sourced from ISO-3166 repo) |

## Workflow summary
1. Configure `config.yaml` and ensure `data/global_datasets` contains the desired countries’ rasters/grids.
2. Instantiate `gridflow.model.region` with ISO3s and call `create_zones` (PV or wind) plus `set_zone_data`.
3. Generate the simplified zone-to-zone `network.flow` matrix via `create_network`.
4. Use `gridflow.epm_input_generator.generate_epm_inputs` to emit CSVs for EPM that respect the newly defined zonal split.

See `README.md` and `DemoNotebook.ipynb` for hands-on examples of these steps.
