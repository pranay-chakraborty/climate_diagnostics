# Climate Diagnostics Toolkit

A Python package for analyzing and visualizing climate data from NetCDF files with a focus on time series analysis and spatial pattern visualization.

## Overview

Climate Diagnostics Toolkit provides powerful tools to process, analyze, and visualize climate data. The package offers functionality for seasonal filtering, spatial averaging with proper area weighting, trend analysis, and time series decomposition to help climate scientists extract meaningful insights from large climate datasets.

## Features

* **Data Loading and Processing**:
  * Load and process NetCDF climate data files with automatic chunking using dask
  * Filter data by meteorological seasons (Annual, DJF, MAM, JJAS)
  * Select data by latitude, longitude, level, and time range

* **Time Series Analysis**:
  * Plot time series with proper area weighting to account for grid cell size differences
  * Calculate and visualize spatial standard deviations
  * Decompose time series into trend, seasonal, and residual components using STL
  * Extract and analyze trends with statistical significance testing

* **Visualization**:
  * Generate spatial maps of means and standard deviations
  * Create publication-quality figures with Cartopy map projections
  * Visualize time series decompositions and trends

## Installation

### Using pip

```bash
pip install climate-diagnostics
```

### From Source
```bash
git clone https://github.com/username/climate-diagnostics.git
cd climate-diagnostics
pip install -e .
```

## Usage Examples

### Loading and Plotting Time Series
``` Python
from climate_diagnostics import TimeSeries

# Load data
ts = TimeSeries("path/to/climate_data.nc")

# Plot time series for a specific region and season
ts.plot_time_series(
    latitude=slice(-30, 30),
    longitude=slice(0, 180),
    level=850,  # hPa pressure level
    variable="air",
    season="jjas"  # June-July-August-September

```

### Analyzing Trends

``` Python
from climate_diagnostics import Trends

# Initialize trends analyzer
trends = Trends("path/to/climate_data.nc")

# Calculate and visualize trends with area-weighted averaging
results = trends.calculate_trend(
    variable="air",
    latitude=slice(-30, 30),
    longitude=slice(0, 180),
    level=850,
    season="annual",
    area_weighted=True,
    plot=True,
    return_results=True
)

# Access trend statistics
print(f"Trend slope: {results['trend_statistics'].loc['slope', 'value']}")
print(f"P-value: {results['trend_statistics'].loc['p_value', 'value']}")

```


### Creating Spatial Maps

``` Python
from climate_diagnostics import Plots

# Initialize plots
plots = Plots("path/to/climate_data.nc")

# Plot mean of a variable
plots.plot_mean(
    variable="air",
    level=850,
    season="djf"  # December-January-February
)

```

## Dependencies

- xarray
- dask
- netCDF4
- matplotlib
- numpy
- scipy
- cartopy
- statsmodels
- scikit-learn

## Development

### Setting up the development environment

``` bash
git clone https://github.com/username/climate-diagnostics.git
cd climate-diagnostics
conda env create -f environment.yml
conda activate climate-diagnostics
pip install -e ".[dev]"
```
### Running Tests

``` bash
pytest
```

## License

[MIT LICENSE](https://github.com/pranay-chakraborty/climate_diagnostics/blob/master/LICENSE)

## Citation

If you use Climate Diagnostics Toolkit in your research, please cite:

```
Chakraborty, P. (2025). Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data.
```


For LaTeX users:

```bibtex
@software{chakraborty2025climate,
  author = {Chakraborty, P.},
  title = {Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/pranay-chakraborty/climate_diagnostics},
  note = {[Computer software]}
} 
```