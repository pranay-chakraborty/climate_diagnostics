
# Climate Diagnostics Toolkit

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://pranay-chakraborty.github.io/climate_diagnostics/)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.1-brightgreen.svg)](https://github.com/pranay-chakraborty/climate_diagnostics/releases)
[![Status](https://img.shields.io/badge/status-stable-green.svg)](https://github.com/pranay-chakraborty/climate_diagnostics)
[![GitHub Actions](https://github.com/pranay-chakraborty/climate_diagnostics/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/pranay-chakraborty/climate_diagnostics/actions/workflows/docs.yml)
[![Issues](https://img.shields.io/github/issues/pranay-chakraborty/climate_diagnostics.svg)](https://github.com/pranay-chakraborty/climate_diagnostics/issues)
[![GitHub Stars](https://img.shields.io/github/stars/pranay-chakraborty/climate_diagnostics.svg)](https://github.com/pranay-chakraborty/climate_diagnostics/stargazers)

A Python toolkit for analyzing, processing, and visualizing climate data from model output, reanalysis, and observations. Built on xarray, it provides specialized accessors for time series analysis, trend calculations, and spatial diagnostics with chunking optimization and parallel processing support.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Overview](#api-overview)
- [Documentation](#documentation)
- [Contributing & Support](#contributing--support)
- [Development & Testing](#development--testing)
- [License](#license)
- [Citation](#citation)

## Key Features

- **xarray Integration**: Access features via `.climate_plots`, `.climate_timeseries`, and `.climate_trends` accessors on xarray Datasets.
- **Chunking Optimization**: Memory-efficient chunking strategies for large datasets.
- **Temporal Analysis**: Trend detection, STL decomposition, and variability analysis.
- **Spatial Visualization**: Map plotting with Cartopy and custom projections.
- **Statistical Diagnostics**: Climate science methods including ETCCDI indices.
- **Multi-model Analysis**: Compare and evaluate climate model outputs.
- **Performance**: Dask-powered parallel processing for large datasets.

## Installation

### With pip
```bash
pip install climate_diagnostics
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

# Open a large dataset
ds = xr.open_dataset("/path/to/air.mon.mean.nc")

# Optimize chunking for analysis
ds = ds.climate_timeseries.optimize_chunks_advanced(
    operation_type='timeseries',
    performance_priority='balanced'
)

# Plot a mean map
ds.climate_plots.plot_mean(variable="air", season="djf")

# Analyze trends
ds.climate_trends.calculate_spatial_trends(
    variable="air",
    num_years=10,
    latitude=slice(40, 60),
    longitude=slice(60, 110),
    optimize_chunks=True
)

# Get chunking recommendations
ds.climate_timeseries.analyze_chunking_strategy()
```

## API Overview

### Accessors

- `climate_plots`: Geographic and statistical visualizations
- `climate_timeseries`: Time series analysis and decomposition
- `climate_trends`: Trend calculation and significance testing

### Example: Time Series
```python
ds.climate_timeseries.plot_time_series(
    latitude=slice(40, 60),
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
    latitude=slice(40, 60),
    longitude=slice(60, 110)
)
```

## Documentation

[Complete Documentation](https://pranay-chakraborty.github.io/climate_diagnostics/)

- [Quick Start Guide](https://pranay-chakraborty.github.io/climate_diagnostics/quickstart.html) - Get started quickly
- [API Reference](https://pranay-chakraborty.github.io/climate_diagnostics/api/) - Complete function documentation
- [User Guide](https://pranay-chakraborty.github.io/climate_diagnostics/user_guide/) - In-depth tutorials
- [Examples](https://pranay-chakraborty.github.io/climate_diagnostics/examples/) - Usage examples

### Local Documentation Build

To build and view documentation locally:

```bash
cd docs
make html
# Open build/html/index.html in your browser
```

## Contributing & Support

- [Report Issues](https://github.com/pranay-chakraborty/climate_diagnostics/issues) - Bug reports and feature requests
- [Discussions](https://github.com/pranay-chakraborty/climate_diagnostics/discussions) - Questions and community support
- [Contributing Guide](https://pranay-chakraborty.github.io/climate_diagnostics/contributing.html) - How to contribute

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

---

<div align="center">

[Documentation](https://pranay-chakraborty.github.io/climate_diagnostics/) | [Issues](https://github.com/pranay-chakraborty/climate_diagnostics/issues) | [Discussions](https://github.com/pranay-chakraborty/climate_diagnostics/discussions)

![WeCLiMb Logo](https://pranay-chakraborty.github.io/climate_diagnostics/_static/WeCLiMb_LOGO_1.png)

</div>