"""
Utilities module for climate_diagnostics.

This module provides the core infrastructure for the package, exposing helper 
functions for:
- Coordinate handling and standardization (coord_utils)
- Data selection, filtering, and spatial averaging (data_utils)
- Dask resource management (dask_utils)
- Plotting configuration and projections (plot_utils)
"""

from .dask_utils import get_or_create_dask_client, managed_dask_client
from .coord_utils import get_coord_name, filter_by_season
from .data_utils import select_process_data, get_spatial_mean
from .plot_utils import get_projection

__all__ = [
    # Dask Management
    'get_or_create_dask_client',
    'managed_dask_client',
    
    # Coordinate Handling
    'get_coord_name',
    'filter_by_season',
    
    # Data Processing
    'select_process_data',
    'get_spatial_mean',
    
    # Plotting Helpers
    'get_projection',
]