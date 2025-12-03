# gridflow
Building a network flow model for planning

The `gridflow` library uses geospatial data to build out a network flow model for a given country or region. It then outputs this representation in a form that can serve as an input to EPM, the GAMS based capacity expansion and market modeling tool used by the World Bank.

## Installation

> **Important:** Use the pinned environment definition to lock dependencies to Python 3.9 and the matching native libraries.

1. Create and configure the environment from the provided `environment.yml`:
   ```
   conda env create -f environment.yml
   ```
2. Activate the environment before working on the project:
   ```
   conda activate gridflow_env
   ```
3. (Optional) Verify the installation by importing `gridflow` in a Python shell or running `conda list`/`pip list`.


## Run

To run gridflow, clone the repository locally, and update the files in the `data/global_datasets` folder to include the region you are interested in modeling. The files already included in the repository are not global, but provide data samples that allow for testing out the functionality of the repository on small data sizes. For World Bank internal staff, full global files - named and structured to match the requirements of gridflow input data readers - are available on an [internal drive](https://worldbankgroup.sharepoint.com/:f:/r/teams/PowerSystemPlanning-WBGroup/Shared%20Documents/2.%20Knowledge%20Products/19.%20Databases/Gridflow/global_datasets/global_datasets?csf=1&web=1&e=TI8SuT
), or can be obtained by contacting the modeling team. 

For external users, most of the data is openly accessible, and the sources can be found in the `data/global_datasets/data_documentation.txt` file.

Once you have the necessary input data, setting up a gridflow model is easy: 
```
from gridflow import model

# Specify the countries you wish to include, in ISO-3 format
countries = ["TUR", "SYR"]

mod = model.region(countries, 'data/global_datasets')
# Specify the number of zones you want to create
mod.create_zones(n=5)

# Build the power network "pipes" of the flow model, based on zones.
mod.create_network()
```
The `DemoNotebook` provides more examples of creating and interacting with the gridflow model.

If you want to run the whole pipeline (zones, zonal statistics, flows, and EPM inputs) without writing a script, simply call the CLI entry point:

```
python main.py --countries TUR,IRQ --n-zones 6 --generate-plots --epm-input-raw data/epm_inputs_raw --epm-output-dir data/epm_inputs
```

Ensure `data/epm_inputs_raw` contains the EPM template CSVs you want to split across zones; the CLI mirrors `model.region` + `generate_epm_inputs` and lets you control the demand scaling column with `--demand-scaleby` (defaults to `population`).

## Core components at a glance

- `gridflow.model.region`: the high-level pipeline driver. During `__init__` it reads borders, loads the sample grid, and wires up the available rasters/statistics. The common flow is `create_zones()` → `set_zone_data()` → `create_network()` → `epm_input_generator.generate_epm_inputs()` to emit zonalized EPM templates.
- `gridflow.model.network`: encapsulates the transmission representation built in `create_network()`. It stores the lines, nodes, and flow matrix used by the visualization helpers and neighbor summaries, and exposes helpers (e.g., `_get_line_capacity`) needed to keep the network data consistent across plots and exports.
- `gridflow.model.zonedata`: parameterized aggregator for raster statistics. Each entry records a raster path, zonal aggregation method, and metadata so `region.set_zone_data()` can clip each raster by zone and compute sums/means without repeating the logic everywhere.
- `gridflow.data_readers`: helpers that load geopandas/raster data (borders, grid files, raster clips, renewables.ninja queries) using the paths configured in `config.yaml`, keeping CLI and tests on the same dataset layout. It also exposes optional background map loading for `gridflow.visuals`.
- `gridflow.visuals`: visualization utilities for zone segmentation, choropleths, network layout, and flow heatmaps. They draw optional country/populated-place backgrounds and clamp the axes to the modeled zones so the diagnostics remain readable even when only a subset of the world is modeled.

`main.py` / `gridflow/main.py` stitch these pieces together: parse CLI args, build a `region`, run the chosen flow, and write map PDFs or neighbor summaries as requested.
