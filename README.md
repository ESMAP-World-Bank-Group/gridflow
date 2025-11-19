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

To run gridflow, clone the repository locally, and update the files in the `data/global_datasets` folder to include the region you are interested in modeling. The files already included in the repository are not global, but provide data samples that allow for testing out the
functionality of the repository on small data sizes. For World Bank internal staff, full global files - named and structured to match the requirements of gridflow input data readers - are available on an internal drive, and can be obtained by contacting the modeling team. For
external users, most of the data is openly accessible, and the sources can be found in the `data/global_datasets/data_documentation.txt` file.

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
