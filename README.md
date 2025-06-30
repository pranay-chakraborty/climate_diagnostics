
# Climate Diagnostics Toolkit

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-1.1-brightgreen.svg)
![Status](https://img.shields.io/badge/status-alpha-orange.svg)
[![PyPI version](https://img.shields.io/pypi/v/climate_diagnostics.svg)](https://pypi.org/project/climate_diagnostics/)

A comprehensive Python toolkit for analyzing, processing, and visualizing climate data from model output, reanalysis, and observations. Built on xarray, it provides specialized accessors for time series, trends, and spatial diagnostics, with robust support for parallel processing and publication-quality figures.

## Key Features

- **Seamless xarray Integration**: Access all features via `.climate_plots`, `.climate_timeseries`, and `.climate_trends` on xarray Datasets.
- **Temporal Analysis**: Trend detection, STL decomposition, and variability analysis.
- **Spatial Visualization**: Publication-quality maps with Cartopy, custom projections, and area-weighted statistics.
- **Statistical Diagnostics**: Advanced methods for climate science, including ETCCDI indices.
- **Multi-model Analysis**: Compare and evaluate climate model outputs.
- **Performance**: Dask-powered parallel processing for large datasets.

## Installation

### With pip
```bash
pip install climate-diagnostics
```

### With conda (recommended for all dependencies)
```bash
conda env create -f environment.yml
conda activate climate-diagnostics
pip install -e .
```

## Quick Start

```python
import xarray as xr
from climate_diagnostics import accessors

# Open a dataset
ds = xr.open_dataset("/path/to/air.mon.mean.nc")

# Plot a mean map
ds.climate_plots.plot_mean(variable="air", season="djf")

# Analyze trends
ds.climate_trends.calculate_spatial_trends(
    variable="air",
    num_years=10,
    latitude=slice(40, 6),
    longitude=slice(60, 110)
)
```

## API Overview

### Accessors

- `climate_plots`: Geographic and statistical visualizations
- `climate_timeseries`: Time series analysis and decomposition
- `climate_trends`: Trend calculation and significance testing

### Example: Time Series
```python
ds.climate_timeseries.plot_time_series(
    latitude=slice(40, 6),
    longitude=slice(60, 110),
    level=850,
    variable="air",
    season="jjas"
)
```

### Example: Climate Indices
```python
ds.climate_plots.plot_consecutive_wet_days(
    variable="prate",
    threshold=1.0,
    latitude=slice(40, 6),
    longitude=slice(60, 110)
)
```

## Documentation

Full API documentation and usage examples are available in the [`docs/`](docs/) folder. To build and view locally:

```bash
cd docs
make html
# Then open _build/html/index.html in your browser
```

## Development & Testing

```bash
git clone https://github.com/pranay-chakraborty/climate_diagnostics.git
cd climate_diagnostics
conda env create -f environment.yml
conda activate climate-diagnostics
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the [MIT LICENSE](LICENSE).

## Citation

If you use Climate Diagnostics Toolkit in your research, please cite:

```
Chakraborty, P. (2025) & Muhammed I. K., A. (2025). Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data using xarray accessors. Version 1.1. https://github.com/pranay-chakraborty/climate_diagnostics
```

For LaTeX users:

```bibtex
@software{chakraborty2025climate,
  author = {Chakraborty, Pranay and Muhammed I. K., Adil},
  title = {{Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data using xarray accessors}},
  year = {2025},
  version = {1.1},
  publisher = {GitHub},
  url = {https://github.com/pranay-chakraborty/climate_diagnostics},
  note = {[Computer software]}
}
```