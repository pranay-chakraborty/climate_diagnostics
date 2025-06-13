import xarray as xr
import numpy as np

from .coord_utils import get_coord_name, filter_by_season


def validate_and_get_sel_slice(coord_val_param, data_coord, coord_name_str, is_datetime_intent=False):
    """
    Validate a coordinate selection parameter against the data's coordinate range.
    """
    min_data_val_raw = data_coord.min().item()
    max_data_val_raw = data_coord.max().item()
    needs_nearest_for_this_coord = False
    sel_val = coord_val_param

    comp_req_min, comp_req_max = None, None
    comp_data_min, comp_data_max = min_data_val_raw, max_data_val_raw

    if isinstance(coord_val_param, slice):
        comp_req_min, comp_req_max = coord_val_param.start, coord_val_param.stop
    elif isinstance(coord_val_param, (list, np.ndarray)):
        if not len(coord_val_param): raise ValueError(f"{coord_name_str.capitalize()} selection list/array empty.")
        comp_req_min, comp_req_max = min(coord_val_param), max(coord_val_param)
    else: 
        comp_req_min = comp_req_max = coord_val_param
        needs_nearest_for_this_coord = isinstance(coord_val_param, (int, float, np.number))

    if is_datetime_intent:
        data_dtype = data_coord.dtype
        try:
            if comp_req_min is not None: comp_req_min = np.datetime64(comp_req_min)
            if comp_req_max is not None: comp_req_max = np.datetime64(comp_req_max)
            
            if np.issubdtype(data_dtype, np.datetime64):
                if isinstance(min_data_val_raw, (int, np.integer)):
                    unit = np.datetime_data(data_dtype)[0]
                    comp_data_min = np.datetime64(min_data_val_raw, unit)
                    comp_data_max = np.datetime64(max_data_val_raw, unit)
                else: 
                    comp_data_min = np.datetime64(min_data_val_raw)
                    comp_data_max = np.datetime64(max_data_val_raw)
            elif hasattr(min_data_val_raw, 'year'):
                comp_data_min = np.datetime64(min_data_val_raw)
                comp_data_max = np.datetime64(max_data_val_raw)

        except Exception as e:
            print(f"Warning: Could not fully process/validate {coord_name_str} range "
                  f"'{coord_val_param}' against data bounds due to type issues: {e}. "
                  "Relying on xarray's .sel() behavior.")
            comp_data_min, comp_data_max = None, None

    if comp_data_min is not None and comp_data_max is not None: 
        if comp_req_min is not None and comp_req_min > comp_data_max:
            raise ValueError(f"Requested {coord_name_str} min {coord_val_param} > data max {max_data_val_raw}")
        if comp_req_max is not None and comp_req_max < comp_data_min:
            raise ValueError(f"Requested {coord_name_str} max {coord_val_param} < data min {min_data_val_raw}")
    
    return sel_val, needs_nearest_for_this_coord


def select_process_data(xarray_obj, variable, latitude=None, longitude=None, level=None,
                        time_range=None, season='annual', year=None):
    """
    Select, filter, and process a data variable from the dataset.
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