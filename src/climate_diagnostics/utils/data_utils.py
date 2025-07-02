import xarray as xr
import numpy as np
from typing import Optional, Union, Tuple, Any, List
import warnings

from .coord_utils import get_coord_name, filter_by_season


def validate_and_get_sel_slice(
    coord_val_param: Union[float, int, slice, List, np.ndarray], 
    data_coord: xr.DataArray, 
    coord_name_str: str, 
    is_datetime_intent: bool = False
) -> Tuple[Union[float, int, slice, List, np.ndarray], bool]:
    """
    Validate a coordinate selection parameter against the data's coordinate range.
    
    Parameters
    ----------
    coord_val_param : Union[float, int, slice, List, np.ndarray]
        The coordinate selection parameter to validate
    data_coord : xr.DataArray
        The data coordinate array to validate against
    coord_name_str : str
        Name of the coordinate for error messages
    is_datetime_intent : bool, optional
        Whether the coordinate is intended to be datetime, by default False
        
    Returns
    -------
    Tuple[Union[float, int, slice, List, np.ndarray], bool]
        Validated selection value and whether nearest neighbor selection is needed
        
    Raises
    ------
    ValueError
        If coordinate validation fails
    """
    min_data_val_raw = data_coord.min().item()
    max_data_val_raw = data_coord.max().item()
    needs_nearest_for_this_coord = False
    sel_val = coord_val_param

    comp_req_min, comp_req_max = None, None
    comp_data_min, comp_data_max = min_data_val_raw, max_data_val_raw

    # Determine coordinate bounds based on selection type
    if isinstance(coord_val_param, slice):
        # Slice selection: validate start and stop bounds
        comp_req_min, comp_req_max = coord_val_param.start, coord_val_param.stop
    elif isinstance(coord_val_param, (list, np.ndarray)):
        # List/array selection: validate against min/max values in the selection
        if not len(coord_val_param): 
            raise ValueError(f"{coord_name_str.capitalize()} selection list/array empty.")
        comp_req_min, comp_req_max = min(coord_val_param), max(coord_val_param)
    else: 
        # Scalar selection: single value selection (will use nearest-neighbor if numeric)
        comp_req_min = comp_req_max = coord_val_param
        needs_nearest_for_this_coord = isinstance(coord_val_param, (int, float, np.number))

    # Special handling for datetime coordinates (time-related selections)
    if is_datetime_intent:
        comp_data_min, comp_data_max = _validate_datetime_coordinates(
            coord_val_param, data_coord, coord_name_str, 
            comp_req_min, comp_req_max, min_data_val_raw, max_data_val_raw
        )

    # Perform comprehensive bounds checking to catch invalid selections early
    _check_coordinate_bounds(
        comp_req_min, comp_req_max, comp_data_min, comp_data_max, 
        coord_name_str, coord_val_param, min_data_val_raw, max_data_val_raw
    )
    
    return sel_val, needs_nearest_for_this_coord


def _validate_datetime_coordinates(
    coord_val_param: Any, 
    data_coord: xr.DataArray, 
    coord_name_str: str,
    comp_req_min: Any, 
    comp_req_max: Any, 
    min_data_val_raw: Any, 
    max_data_val_raw: Any
) -> Tuple[Any, Any]:
    """
    Helper function to validate and convert datetime coordinates for comparison.
    
    Handles the complex logic of comparing user-provided datetime values with
    dataset coordinate values, accounting for different datetime formats and types.
    """
    comp_data_min, comp_data_max = min_data_val_raw, max_data_val_raw
    data_dtype = data_coord.dtype
    
    # Convert user-provided values to numpy datetime64 for comparison
    try:
        if comp_req_min is not None: 
            comp_req_min = np.datetime64(comp_req_min)
        if comp_req_max is not None: 
            comp_req_max = np.datetime64(comp_req_max)
        
        # Handle numpy datetime64 coordinate data
        if np.issubdtype(data_dtype, np.datetime64):
            # Convert integer timestamps to proper datetime64 objects using correct units
            if isinstance(min_data_val_raw, (int, np.integer)):
                unit = np.datetime_data(data_dtype)[0]  # Extract time unit from dtype
                comp_data_min = np.datetime64(min_data_val_raw, unit)
                comp_data_max = np.datetime64(max_data_val_raw, unit)
            else: 
                # Direct conversion for datetime objects
                comp_data_min = np.datetime64(min_data_val_raw)
                comp_data_max = np.datetime64(max_data_val_raw)
        elif hasattr(min_data_val_raw, 'year'):
            comp_data_min = np.datetime64(min_data_val_raw)
            comp_data_max = np.datetime64(max_data_val_raw)

    except Exception as e:
        warnings.warn(
            f"Could not fully process/validate {coord_name_str} range "
            f"'{coord_val_param}' against data bounds due to type issues: {e}. "
            "Relying on xarray's .sel() behavior.",
            UserWarning
        )
        comp_data_min, comp_data_max = None, None
    
    return comp_data_min, comp_data_max


def _check_coordinate_bounds(
    comp_req_min: Any, 
    comp_req_max: Any, 
    comp_data_min: Any, 
    comp_data_max: Any,
    coord_name_str: str, 
    coord_val_param: Any, 
    min_data_val_raw: Any, 
    max_data_val_raw: Any
) -> None:
    """Helper function to check coordinate bounds."""
    if comp_data_min is not None and comp_data_max is not None: 
        if comp_req_min is not None and comp_req_min > comp_data_max:
            raise ValueError(
                f"Requested {coord_name_str} min {coord_val_param} > data max {max_data_val_raw}"
            )
        if comp_req_max is not None and comp_req_max < comp_data_min:
            raise ValueError(
                f"Requested {coord_name_str} max {coord_val_param} < data min {min_data_val_raw}"
            )


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
        The input dataset
    variable : str
        Name of the variable to select
    latitude : Optional[Union[float, slice, List]], optional
        Latitude selection parameter, by default None
    longitude : Optional[Union[float, slice, List]], optional
        Longitude selection parameter, by default None
    level : Optional[Union[float, slice, List]], optional
        Level selection parameter, by default None
    time_range : Optional[slice], optional
        Time range selection, by default None
    season : str, optional
        Season filter, by default 'annual'
    year : Optional[int], optional
        Year filter, by default None
        
    Returns
    -------
    xr.DataArray
        Processed data variable
        
    Raises
    ------
    ValueError
        If variable not found or selection results in empty data
    """
    if variable not in xarray_obj.data_vars:
        raise ValueError(f"Variable '{variable}' not found. Available: {list(xarray_obj.data_vars.keys())}")
    
    data_var = xarray_obj[variable]

    time_name = get_coord_name(data_var, ['time', 't'])
    if time_name and time_name in data_var.dims:
        if season.lower() != 'annual':
            data_var = filter_by_season(data_var, season)
            if data_var[time_name].size == 0:
                raise ValueError(f"No data available after season ('{season}') filter.")
        
        if year is not None:
            try:
                year_match_bool = data_var[time_name].dt.year == year
            except (AttributeError, TypeError):
                year_match_bool = xr.DataArray(
                    [t.year == year for t in data_var[time_name].compute().data],
                    coords={time_name: data_var[time_name]}, dims=[time_name]
                )
            data_var = data_var.sel({time_name: year_match_bool})
            if data_var[time_name].size == 0:
                raise ValueError(f"No data for year {year} (after season '{season}' filter).")

        if time_range is not None:
            sel_val, _ = validate_and_get_sel_slice(time_range, data_var[time_name], "time", True)
            data_var = data_var.sel({time_name: sel_val})
            if data_var[time_name].size == 0:
                raise ValueError("No data after time_range selection.")
    elif season.lower() != 'annual' or year is not None or time_range is not None :
            print(f"Warning: Temporal filters (season, year, time_range) requested, "
                    f"but time dimension ('{time_name}') not found or not a dimension in variable '{variable}'.")

    selection_dict = {}
    method_dict = {}

    lat_name = get_coord_name(xarray_obj, ['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
    if latitude is not None and lat_name and lat_name in data_var.coords:
        sel_val, needs_nearest = validate_and_get_sel_slice(latitude, data_var[lat_name], "latitude")
        selection_dict[lat_name] = sel_val
        if needs_nearest: method_dict[lat_name] = 'nearest'

    lon_name = get_coord_name(xarray_obj, ['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
    if longitude is not None and lon_name and lon_name in data_var.coords:
        sel_val, needs_nearest = validate_and_get_sel_slice(longitude, data_var[lon_name], "longitude")
        selection_dict[lon_name] = sel_val
        if needs_nearest: method_dict[lon_name] = 'nearest'
    
    level_name = get_coord_name(xarray_obj, ['level', 'lev', 'plev', 'height', 'altitude', 'depth', 'z'])
    if level_name and level_name in data_var.dims:
        if level is not None:
            if isinstance(level, (slice, list, np.ndarray)): 
                sel_val, _ = validate_and_get_sel_slice(level, data_var[level_name], "level")
                print(f"Averaging over levels: {level}")
                with xr.set_options(keep_attrs=True):
                    data_to_avg = data_var.sel({level_name: sel_val})
                    if level_name in data_to_avg.dims and data_to_avg.sizes[level_name] > 1:
                            data_var = data_to_avg.mean(dim=level_name)
                    else:
                            data_var = data_to_avg
            else: 
                sel_val, needs_nearest = validate_and_get_sel_slice(level, data_var[level_name], "level")
                selection_dict[level_name] = sel_val
                if needs_nearest: method_dict[level_name] = 'nearest'
        elif data_var.sizes[level_name] > 1: 
            first_level_val = data_var[level_name].isel({level_name: 0}).item()
            selection_dict[level_name] = first_level_val
            print(f"Warning: Multiple levels found in '{variable}'. Using first level: {first_level_val}")
    elif level is not None:
        print(f"Warning: Level dimension '{level_name}' not found or not a dimension in '{variable}'. Ignoring 'level' parameter.")

    if selection_dict:
        if any(isinstance(v, slice) for v in selection_dict.values()) and method_dict:
            print("Note: Applying selections. Slices will be used directly, 'nearest' for scalar points if specified.")
        try:
            data_var = data_var.sel(selection_dict, method=method_dict if method_dict else None)
        except Exception as e:
            print(f"Error during final .sel() operation: {e}")
            print(f"Selection dictionary: {selection_dict}, Method dictionary: {method_dict}")
            raise

    if data_var.size == 0:
        print("Warning: Selection resulted in an empty DataArray.")
    return data_var


def get_spatial_mean(data_var, area_weighted=True):
    """
    Calculate the spatial mean of a DataArray.
    """
    lat_name = get_coord_name(data_var, ['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
    lon_name = get_coord_name(data_var, ['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
    
    spatial_dims_present = []
    if lat_name and lat_name in data_var.dims:
        spatial_dims_present.append(lat_name)
    if lon_name and lon_name in data_var.dims:
        spatial_dims_present.append(lon_name)

    if not spatial_dims_present:
        return data_var

    if area_weighted and lat_name in spatial_dims_present:
        weights = np.cos(np.deg2rad(data_var[lat_name]))
        weights.name = "weights"
        print("Calculating area-weighted spatial mean.")
        return data_var.weighted(weights).mean(dim=spatial_dims_present, skipna=True)
    else:
        weight_msg = "(unweighted)" if lat_name in spatial_dims_present and not area_weighted else ""
        print(f"Calculating simple spatial mean {weight_msg}.")
        return data_var.mean(dim=spatial_dims_present, skipna=True) 