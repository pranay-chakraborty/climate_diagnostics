import xarray as xr
import numpy as np


def get_coord_name(xarray_like_obj, possible_names):
    """
    Find the name of a coordinate in an xarray object from a list of possible names.

    This function checks for coordinate names in a case-sensitive manner first,
    then falls back to a case-insensitive check.

    Parameters
    ----------
    xarray_like_obj : xr.DataArray or xr.Dataset
        The xarray object to search for coordinates.
    possible_names : list of str
        A list of possible coordinate names to look for.

    Returns
    -------
    str or None
        The found coordinate name, or None if no matching coordinate is found.
    """
    if xarray_like_obj is None:
        return None
    for name in possible_names:
        if name in xarray_like_obj.coords:
            return name
    coord_names_lower = {name.lower(): name for name in xarray_like_obj.coords}
    for name in possible_names:
        if name.lower() in coord_names_lower:
            return coord_names_lower[name.lower()]
    return None


def filter_by_season(data_subset, season='annual'):
    """
    Filter climate data for a specific season using month-based selection.

    This function implements robust seasonal filtering that handles various
    time coordinate formats commonly found in climate datasets, including
    standard datetime64, cftime objects, and numeric time coordinates.

    Supported seasons are 'annual', 'jjas', 'djf', 'mam', 'son', 'jja'.

    Parameters
    ----------
    data_subset : xr.DataArray or xr.Dataset
        The climate data to filter by season.
    season : str, optional
        The season to filter by. Defaults to 'annual' (no filtering).
        Options:
        - 'annual': All months (no filtering)
        - 'jjas': June-July-August-September (monsoon season)
        - 'djf': December-January-February (winter/dry season)
        - 'mam': March-April-May (spring/pre-monsoon)
        - 'son': September-October-November (autumn/post-monsoon)
        - 'jja': June-July-August (summer)

    Returns
    -------
    xr.DataArray or xr.Dataset
        The filtered data containing only the specified season.
        
    Raises
    ------
    ValueError
        If time coordinate cannot be found or processed for seasonal filtering.

    Notes
    -----
    This function automatically detects time coordinate naming conventions
    and handles different datetime formats including CF-compliant cftime objects.
    """
    season_input = season
    season = season.lower()  # Normalize to lowercase for consistent processing
    
    # Return unfiltered data for annual analysis
    if season == 'annual':
        return data_subset

    # Step 1: Locate the time coordinate using flexible name detection
    time_coord_name = get_coord_name(data_subset, ['time', 't'])
    if time_coord_name is None:
        raise ValueError("Cannot find time coordinate for seasonal filtering.")

    # Step 2: Validate that time coordinate is a usable dimension
    if time_coord_name not in data_subset.dims:
        if time_coord_name in data_subset.coords:
            print(f"Warning: Time coordinate '{time_coord_name}' exists but is not a dimension. "
                  "Cannot filter by season. Returning unfiltered data.")
            return data_subset
        raise ValueError(f"Time dimension '{time_coord_name}' not found for seasonal filtering.")

    # Step 3: Extract month information using multiple approaches for robustness

    time_coord_da = data_subset[time_coord_name]
    if 'month' in data_subset.coords:
        month_coord = data_subset['month']
    elif hasattr(time_coord_da.dt, 'month'):
        month_coord = time_coord_da.dt.month
    elif time_coord_da.size > 0 and hasattr(time_coord_da.data[0], 'month'):
        try:
            time_values = time_coord_da.values
            months_list = [t.month for t in time_values]
            month_coord = xr.DataArray(months_list, coords={time_coord_name: time_coord_da}, dims=[time_coord_name])
            print("Warning: Extracted months from cftime objects manually. This might be slow for large Dask arrays.")
        except Exception as e:
                raise ValueError(f"Time coordinate '{time_coord_name}' (type: {time_coord_da.dtype}) "
                                 f"could not be processed for month extraction. Error: {e}")
    else:
        raise ValueError(f"Cannot determine month for seasonal filtering from time coordinate '{time_coord_name}' "
                         f"(dtype: {time_coord_da.dtype}). It's not datetime64-like with a .dt.month accessor, "
                         "not cftime-like with a .month attribute, and no 'month' coordinate exists.")


    season_months = {'jjas': [6, 7, 8, 9], 'djf': [12, 1, 2], 'mam': [3, 4, 5], 'son': [9, 10, 11], 'jja': [6, 7, 8]}
    selected_months = season_months.get(season)

    if selected_months:
        if not set(month_coord.dims).issubset(set(data_subset.dims)) or \
            not all(s1 == s2 for s1,s2 in zip(month_coord.shape, data_subset.shape) if month_coord.dims[0] == data_subset.dims[0]):
            month_coord = month_coord.reindex_like(data_subset[time_coord_name], method='nearest', tolerance='1D')


        filtered_data = data_subset.where(month_coord.isin(selected_months), drop=True)
        if filtered_data[time_coord_name].size == 0:
            print(f"Warning: No data found for season '{season_input.upper()}' within the selected time range.")
        return filtered_data
    else:
        print(f"Warning: Unknown season '{season_input}'. Supported: {list(season_months.keys())}. Returning unfiltered data.")
        return data_subset 