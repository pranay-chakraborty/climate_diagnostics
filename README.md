# Climate Diagnostics Toolkit

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-0.2.2-brightgreen.svg)
![Status](https://img.shields.io/badge/status-alpha-orange.svg)
[![PyPI version](https://img.shields.io/pypi/v/climate_diagnostics.svg)](https://pypi.org/project/climate_diagnostics/)



A Python package for analyzing and visualizing climate data from NetCDF files with a focus on time series and trend analysis, and spatial pattern visualization, integrated directly with xarray via accessors.

## Overview

Climate Diagnostics Toolkit provides powerful tools to process, analyze, and visualize climate data using xarray accessors. The package offers functionality for seasonal filtering, spatial averaging with proper area weighting, trend analysis, and time series decomposition to help climate scientists extract meaningful insights from large climate datasets directly from their xarray Dataset objects.

## Features

* **Data Loading and Processing**:
  * Load NetCDF climate data files using `xarray.open_dataset`.
  * Access toolkit functions via `.climate_timeseries`, `.climate_trends`, and `.climate_plots` accessors on xarray Datasets.
  * Filter data by meteorological seasons (Annual, DJF, MAM, JJAS, etc.) using accessor methods.
  * Select data by latitude, longitude, level, and time range.

* **Time Series Analysis (via `.climate_timeseries`)**:
  * Plot time series with proper area weighting to account for grid cell size differences.
  * Calculate and visualize spatial standard deviations over time.
  * Decompose time series into trend, seasonal, and residual components using STL.

* **Trend Analysis (via `.climate_trends`)**:
  * Calculate and visualize spatially averaged trends with statistical significance testing.
  * Compute and plot spatial maps of trends per grid point using STL decomposition.

* **Visualization (via `.climate_plots`)**:
  * Generate spatial maps of means and standard deviations using Cartopy.
  * Create publication-quality figures with map projections.
  * Optionally smooth spatial plots using Gaussian filters.
  * Option to plot data only over land.

## Installation

### Via pip (recommended)
```bash
pip install climate-diagnostics
```

### From Source
```bash
git clone https://github.com/pranay-chakraborty/climate_diagnostics_toolkit.git 
cd climate_diagnostics_toolkit
pip install -e .
```

## Usage Examples

### Loading Data and Plotting Time Series
```python
import xarray as xr
from climate_diagnostics import accessors

# Load data using xarray
ds = xr.open_dataset("path/to/climate_data.nc")

# Plot time series for a specific region and season using the accessor, given a 3D dataset
ds.climate_timeseries.plot_time_series(
    latitude=slice(40, 6),
    longitude=slice(60,110 ),
    level=850,  # hPa pressure level
    variable="air",
    season="jjas"  # June-July-August-September
)
```

### Analyzing Trends
```python
import xarray as xr
from climate_diagnostics import accessors

# Load data
ds = xr.open_dataset("path/to/climate_data.nc")

# Calculate and visualize trends with area-weighted averaging using the accessor
results = ds.climate_trends.calculate_trend(
    variable="air",
    latitude=slice(40, 6),
    longitude=slice(60,110 ),
    level=850,
    season="annual",
    area_weighted=True,
    plot=True,
    return_results=True
)

# Access trend statistics if return_results=True
if results:
    trend_stats = results['trend_statistics']
    slope = trend_stats.loc[trend_stats['statistic'] == 'slope', 'value'].item()
    p_value = trend_stats.loc[trend_stats['statistic'] == 'p_value', 'value'].item()
    print(f"Trend slope: {slope}")
    print(f"P-value: {p_value}")

# Calculate and plot spatial trends per decade
spatial_trends = ds.climate_trends.calculate_spatial_trends(
    variable="air",
    level=850,
    season="annual",
    frequency="M", # Assuming monthly data
    num_years=10,  # Trend per decade
    plot_map=True,
    n_workers = 8,
    latitude = slice(40,6),
    longitude = slice(60,110)
    
)
```

### Creating Spatial Maps
```python
import xarray as xr
from climate_diagnostics import accessors

# Load data
ds = xr.open_dataset("path/to/climate_data.nc")

# Plot mean of a variable using the accessor
ds.climate_plots.plot_mean(
    variable="air",
    level=850,
    season="djf",  # December-January-February
    latitude = slice(40,6),
    longitude = slice(60,110)
)

# Plot standard deviation over time, smoothed, over land only
ds.climate_plots.plot_std_time(
    variable="air",
    season="jjas",
    gaussian_sigma=1.5, # Apply smoothing
    land_only=True,      # Mask oceans
    latitude = slice(40,6),
    longitude = slice(60,110),
    levels = 500
)
```

## Dependencies

- xarray
- dask
- netCDF4
- bottleneck
- matplotlib
- numpy
- scipy
- cartopy
- statsmodels
- scikit-learn

## Development

### Setting up the development environment

```bash
git clone https://github.com/pranay-chakraborty/climate_diagnostics.git 
cd climate_diagnostics
conda env create -f environment.yml
conda activate climate-diagnostics
pip install -e ".[dev]"
```
### Running Tests

```bash
pytest
```

## License

[MIT LICENSE](LICENSE)

## Citation

If you use Climate Diagnostics Toolkit in your research, please cite:

```
Chakraborty, P. (2025) & Muhammed I. K., A. (2025). Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data using xarray accessors. Version 0.2.1. https://github.com/pranay-chakraborty/climate_diagnostics
```

For LaTeX users:

```bibtex
@software{chakraborty2025climate,
  author = {Chakraborty, Pranay and Muhammed I. K., Adil},
  title = {{Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data using xarray accessors}},
  year = {2025},
  version = {0.2.1},
  publisher = {GitHub},
  url = {https://github.com/pranay-chakraborty/climate_diagnostics},
  note = {[Computer software]}
}
```