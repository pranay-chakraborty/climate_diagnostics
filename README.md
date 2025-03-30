# Climate Diagostics Toolkit

A Python package for analyzing and visualizing climate data time series from NetCDF files.

### Overview

Climate Diagnostics Toolkit provides tools to process, analyze, and
visualize climate data with an emphasis on time series analysis. The
package offers seasonal filtering, spatial averaging with proper area
weighting, standard deviation analysis, and time series decomposition
capabilities with many more to come in the near future.

### Features

* Load and process NetCDF climate data files
* Filter data by meteorological seasons (Annual, DJF, MAM, JJAS)
* Plot time series with proper area weighting to account for grid cell size differences
* Calculate and visualize spatial standard deviation
* Decompose time series into trend, seasonal, and residual components using STL
* Support for data selection by latitude, longitude, pressure level, and time range

### Dependencies

* xarray
* numpy
* pandas
* matplotlib
* dask
* statsmodels

### Development

Clone the repository : 

```bash
git clone https://github.com/username/climate-diagnostics.git
cd climate-diagnostics
pip install -e .
```

Run tests : 

```bash
pytest
```

### License

[MIT LICENSE](https://github.com/pranay-chakraborty/climate_diagnostics/blob/89efca34a2285014ab1b85393af19440c0247118/LICENSE)
