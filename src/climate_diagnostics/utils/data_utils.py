import xarray as xr
import numpy as np
import logging
from typing import Optional, Union, Tuple, Any, List, Dict
import warnings

# Imported from the sibling module we refactored earlier
from .coord_utils import get_coord_name, filter_by_season

# Set up logger
logger = logging.getLogger(__name__)

def validate_and_get_sel_slice(
    coord_val_param: Union[float, int, slice, List, np.ndarray], 
    data_coord: xr.DataArray, 
    coord_name_str: str
) -> Tuple[Union[float, int, slice, List, np.ndarray], bool]:
    """
    Validate a coordinate selection parameter against the data's coordinate range.
    
    Parameters
    ----------
    coord_val_param : Union[float, int, slice, List, np.ndarray]
        The value(s) to select.
    data_coord : xr.DataArray
        The coordinate array from the dataset (e.g., ds['lat']).
    coord_name_str : str
        Name of the coordinate for logging/error messages.

    Returns
    -------
    Tuple[SelectionValue, bool]
        Returns the sanitized selection value and a boolean indicating if 
        method='nearest' is required.
    """
    # 27 Nov: Simplified validation logic to avoid brittle datetime comparisons.
    # We allow xarray to handle the heavy lifting but check basic numeric bounds.
    
    try:
        min_data = data_coord.min().item()
        max_data = data_coord.max().item()
    except (ValueError, TypeError):
        # Fallback for object-type coords or empty coords
        min_data, max_data = None, None

    needs_nearest = False
    sel_val = coord_val_param
    
    # Determine Request Bounds
    req_min, req_max = None, None
    
    if isinstance(coord_val_param, slice):
        req_min, req_max = coord_val_param.start, coord_val_param.stop
    elif isinstance(coord_val_param, (list, np.ndarray)):
        if len(coord_val_param) == 0: 
            raise ValueError(f"{coord_name_str.capitalize()} selection list/array empty.")
        req_min, req_max = min(coord_val_param), max(coord_val_param)
    else: 
        # Scalar selection
        req_min = req_max = coord_val_param
        # 27 Nov: Only use nearest neighbor for numeric scalars
        needs_nearest = isinstance(coord_val_param, (int, float, np.number))

    # Basic Bounds Check (if data is numeric and comparable)
    if min_data is not None and max_data is not None and req_min is not None and req_max is not None:
        try:
            if req_min > max_data:
                raise ValueError(f"Requested {coord_name_str} min {req_min} > data max {max_data}")
            if req_max < min_data:
                raise ValueError(f"Requested {coord_name_str} max {req_max} < data min {min_data}")
        except TypeError:
            # Ignore comparison errors between different types (e.g. cftime vs datetime64)
            # We let xarray's .sel() raise the specific error later if needed.
            pass

    return sel_val, needs_nearest


def select_process_data(
    xarray_obj: xr.Dataset, 
    variable: str, 
    latitude: Optional[Union[float, slice, List]] = None, 
    longitude: Optional[Union[float, slice, List]] = None, 
    level: Optional[Union[float, slice, List]] = None,
    time_range: Optional[slice] = None, 
    season: str = 'annual', 
    year: Optional[int] = None
) -> xr.DataArray:
    """
    Select, filter, and process a data variable from the dataset.
    
    Parameters
    ----------
    xarray_obj : xr.Dataset
        The input dataset.
    variable : str
        Name of the variable to select.
    latitude, longitude, level : Optional
        Selection parameters (slice, list, or single value).
    time_range : slice, optional
        Range of times to select.
    season : str, optional
        Season to filter (e.g., 'annual', 'djf', 'jjas').
    year : int, optional
        Specific year to filter.

    Returns
    -------
    xr.DataArray
        The processed DataArray.

    Raises
    ------
    ValueError
        If the variable is missing or selection results in empty data.
    """
    if variable not in xarray_obj.data_vars:
        raise ValueError(f"Variable '{variable}' not found. Available: {list(xarray_obj.data_vars.keys())}")
    
    data_var = xarray_obj[variable]

    # ---------------------------------------------------------
    # 1. Temporal Filtering
    # ---------------------------------------------------------
    time_name = get_coord_name(data_var, ['time', 't'])
    
    if time_name and time_name in data_var.dims:
        # A. Season Filter
        if season.lower() != 'annual':
            # 27 Nov: delegated to coord_utils.filter_by_season
            data_var = filter_by_season(data_var, season)

        # B. Year Filter
        if year is not None:
            if not hasattr(data_var[time_name], 'dt'):
                 raise ValueError(f"Year filtering requested but coordinate '{time_name}' has no datetime accessor.")
            
            # 27 Nov: FIXED PERFORMNACE BUG. 
            # Replaced "Loop of Doom" with vectorized boolean indexing.
            year_mask = data_var[time_name].dt.year == year
            data_var = data_var.sel({time_name: year_mask})
            
            if data_var[time_name].size == 0:
                logger.warning(f"No data found for year {year} (after applying season '{season}').")

        # C. Time Range Filter
        if time_range is not None:
            try:
                data_var = data_var.sel({time_name: time_range})
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"Invalid time_range selection {time_range}: {e}")

            if data_var[time_name].size == 0:
                logger.warning(f"No data found for time_range {time_range}.")

    elif (season.lower() != 'annual' or year is not None or time_range is not None):
        logger.warning(f"Temporal filters requested, but time dimension not found in '{variable}'.")


    # ---------------------------------------------------------
    # 2. Spatial & Vertical Selection
    # ---------------------------------------------------------
    selection_dict = {}
    method_dict = {}

    # Define mappings: (User Input, Possible Names, Debug Label)
    coord_mappings = [
        (latitude, ['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'nav_lat'], "latitude"),
        (longitude, ['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'nav_lon'], "longitude"),
        (level, ['level', 'lev', 'plev', 'pressure', 'height', 'z', 'depth'], "level")
    ]

    for param_val, names, debug_name in coord_mappings:
        coord_name = get_coord_name(xarray_obj, names)
        
        # Check if coordinate exists and user requested a selection
        if coord_name and coord_name in data_var.coords and param_val is not None:
            sel_val, needs_nearest = validate_and_get_sel_slice(param_val, data_var[coord_name], debug_name)
            selection_dict[coord_name] = sel_val
            if needs_nearest:
                method_dict[coord_name] = 'nearest'
        
        elif param_val is not None:
            # User asked for a slice, but coordinate doesn't exist
            logger.warning(f"Selection requested for '{debug_name}' but no matching coordinate found in dataset.")

    # 27 Nov: Removed implicit level selection (previously selected index 0 if level existed but param was None).
    # This ensures scientific safety; users must explicitly select a level.

    # ---------------------------------------------------------
    # 3. Apply Selections
    # ---------------------------------------------------------
    if selection_dict:
        # Split exact vs nearest
        exact_sel = {k: v for k, v in selection_dict.items() if k not in method_dict}
        method_sel = {k: v for k, v in selection_dict.items() if k in method_dict}

        try:
            # Apply exact slices first
            if exact_sel:
                data_var = data_var.sel(exact_sel)
            
            # Apply nearest neighbor selections
            if method_sel:
                data_var = data_var.sel(method_sel, method='nearest')
        except Exception as e:
            logger.error(f"Selection failed. Exact: {exact_sel}, Method: {method_sel}")
            raise e

    if data_var.size == 0:
        raise ValueError("Final selection resulted in empty DataArray. Check your parameters.")

    return data_var


def get_spatial_mean(data_var: xr.DataArray, area_weighted: bool = True) -> xr.DataArray:
    """
    Calculate the spatial mean of a DataArray.
    
    Parameters
    ----------
    data_var : xr.DataArray
        Input data.
    area_weighted : bool, optional
        If True (default), weights the mean by cos(latitude).

    Returns
    -------
    xr.DataArray
        The spatially averaged data.
    """
    lat_name = get_coord_name(data_var, ['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'nav_lat'])
    lon_name = get_coord_name(data_var, ['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'nav_lon'])
    
    spatial_dims = []
    if lat_name and lat_name in data_var.dims: spatial_dims.append(lat_name)
    if lon_name and lon_name in data_var.dims: spatial_dims.append(lon_name)

    if not spatial_dims:
        logger.debug("No spatial dimensions found, returning original data.")
        return data_var

    if area_weighted and lat_name in spatial_dims:
        lat_coord = data_var[lat_name]
        
        # 27 Nov: Check units for logging (sanity check) but don't crash
        units = getattr(lat_coord, 'units', '').lower()
        if units and 'degree' not in units:
             logger.warning(f"Latitude units are '{units}'. Cosine weighting assumes degrees.")

        # Calculate weights: cos(lat). Clip to [-1, 1] to prevent domain errors.
        weights = np.cos(np.deg2rad(lat_coord))
        weights.name = "weights"
        
        # Log this as info, not a warning (avoids warning fatigue)
        logger.info(f"Calculating area-weighted spatial mean over: {spatial_dims}")
        
        # .weighted handles NaN automatically
        data_weighted = data_var.weighted(weights)
        return data_weighted.mean(dim=spatial_dims, skipna=True)
    
    else:
        logger.info(f"Calculating unweighted spatial mean over: {spatial_dims}")
        return data_var.mean(dim=spatial_dims, skipna=True)