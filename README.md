# Climate Diagnostics Toolkit

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://pranay-chakraborty.github.io/climate_diagnostics/)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.3.0-brightgreen.svg)](https://github.com/pranay-chakraborty/climate_diagnostics/releases)
[![Status](https://img.shields.io/badge/status-stable-green.svg)](https://github.com/pranay-chakraborty/climate_diagnostics)
[![GitHub Actions](https://github.com/pranay-chakraborty/climate_diagnostics/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/pranay-chakraborty/climate_diagnostics/actions/workflows/docs.yml)
[![Issues](https://img.shields.io/github/issues/pranay-chakraborty/climate_diagnostics.svg)](https://github.com/pranay-chakraborty/climate_diagnostics/issues)
[![GitHub Stars](https://img.shields.io/github/stars/pranay-chakraborty/climate_diagnostics.svg)](https://github.com/pranay-chakraborty/climate_diagnostics/stargazers)

A Python toolkit for analyzing, processing, and visualizing climate data from model output, reanalysis, and observations. Built on xarray, it provides specialized accessors for time series analysis, trend calculations, and spatial diagnostics with chunking optimization and parallel processing support.

**Version 1.3.0** - Features a refactored architecture with centralized data processing, enhanced exception handling, and optimized performance for scientific computing workflows.

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

- **Production-Ready Architecture**: Centralized data processing pipeline with robust exception handling and optimized performance.
- **xarray Integration**: Access features via `.climate_plots`, `.climate_timeseries`, and `.climate_trends` accessors on xarray Datasets.
- **Chunking Optimization**: Memory-efficient chunking strategies for large datasets.
- **Enhanced Reliability**: Specific exception handling provides clear error messages and improved debugging experience.
- **Temporal Analysis**: Trend detection, STL decomposition, and variability analysis.
- **Spatial Visualization**: Map plotting with Cartopy and custom projections.
- **Statistical Diagnostics**: Climate science methods including ETCCDI indices.
- **Radiative Equilibrium Models**: Functions for simulating radiative and radiative-convective equilibrium.
- **Multi-model Analysis**: Compare and evaluate climate model outputs.
- **Performance**: Dask-powered parallel processing for large datasets.

## Installation

### With pip (from PyPI)
```bash
pip install climate_diagnostics
```

### With pip (from source)
For the latest development version:
```bash
git clone https://github.com/pranay-chakraborty/climate_diagnostics.git
cd climate_diagnostics
pip install -e .
```

### With conda
If you prefer to manage your environment with conda, you can use the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate climate-diagnostics
pip install -e .
```

## Quick Start

This example demonstrates a typical workflow for analyzing and visualizing climate data using this toolkit.

```python
import xarray as xr
from climate_diagnostics import accessors
from climate_diagnostics.models.rce import create_rce_model

# 1. Load a dataset
ds = xr.tutorial.load_dataset("air_temperature")

# 2. Calculate and plot spatial trends with a custom projection
trends = ds.climate_trends.calculate_spatial_trends(
    variable="air",
    frequency="Y",  # Annual trends
    projection="Robinson"
)

# 3. Plot the mean temperature for a specific season
fig = ds.climate_plots.plot_mean(
    variable="air", 
    season="JJA", 
    title="Mean Summer (JJA) Temperature"
)

# 4. Run a single-column Radiative-Convective Equilibrium (RCE) model
rce_model = create_rce_model(integrate_years=2)

```

## API Overview

### Accessors

- **`climate_plots`**: Geographic and statistical visualizations including:
  - `plot_mean()`, `plot_std()`, `plot_percentile()`
  - `plot_prcptot()`, `plot_sdii()`, `plot_days_above_threshold()`
  - `plot_cdd()`, `plot_wsdi()`, `plot_csdi()`
- **`climate_timeseries`**: Time series analysis and decomposition:
  - `plot_time_series()`, `plot_std_space()`
  - `decompose_time_series()`
- **`climate_trends`**: Trend calculation and significance testing:
  - `calculate_trend()`, `calculate_spatial_trends()`

### Models

- `create_rce_model()`: Simulate radiative-convective equilibrium
- `create_re_model()`: Simulate pure radiative equilibrium

### Example: Time Series Analysis
```python
# Decompose a time series into trend and seasonal components
decomposition = ds.climate_timeseries.decompose_time_series(
    variable="air",
    latitude=slice(30, 40),
    longitude=slice(-100, -90)
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
Chakraborty, P. (2025) & Muhammed I. K., A. (2025). Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data using xarray accessors. Version 1.3.0. https://github.com/pranay-chakraborty/climate_diagnostics
```

For LaTeX users:

```bibtex
@software{chakraborty2025climate,
  author = {Chakraborty, Pranay and Muhammed I. K., Adil},
  title = {{Climate Diagnostics Toolkit: Tools for analyzing and visualizing climate data using xarray accessors}},
  year = {2025},
  version = {1.3.0},
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