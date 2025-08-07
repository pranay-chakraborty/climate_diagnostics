import xarray as xr
import numpy as np
import warnings


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
    Filter climate data for a specific season using xarray's time accessors.

    This function implements robust seasonal filtering that handles various
    time coordinate formats, including standard datetime64 and cftime objects,
    in a performant, Dask-aware manner.

    Parameters
    ----------
    data_subset : xr.DataArray or xr.Dataset
        The climate data to filter by season. Must have a recognizable time dimension.
    season : str, optional
        The season to filter by. Defaults to 'annual'.
        Supported: 'annual', 'jjas', 'djf', 'mam', 'son', 'jja'.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The filtered data containing only the specified season.
        
    Raises
    ------
    ValueError
        If a usable time coordinate cannot be found or processed.

    Notes
    -----
    This function relies on xarray's `.dt` accessor, which works for both
    `numpy.datetime64` and `cftime` objects (if `cftime` is installed).
    """
    # Use a constant for season definitions for clarity
    SEASON_MONTHS = {
        'jjas': [6, 7, 8, 9],  # Monsoon
        'djf': [12, 1, 2],     # Winter
        'mam': [3, 4, 5],      # Pre-monsoon
        'son': [9, 10, 11],    # Post-monsoon
        'jja': [6, 7, 8]       # Summer
    }
    
    normalized_season = season.lower()
    
    if normalized_season == 'annual':
        return data_subset

    # Step 1: Locate the time coordinate.
    time_coord_name = get_coord_name(data_subset, ['time', 't'])
    if not time_coord_name or time_coord_name not in data_subset.dims:
        raise ValueError("A recognizable time dimension (e.g., 'time', 't') is required for seasonal filtering.")

    # Step 2: Use xarray's .dt accessor to get the month.
    # This is the key improvement: it works for both datetime64 and cftime objects
    # without needing a slow, memory-intensive manual loop.
    time_coord = data_subset[time_coord_name]
    if hasattr(time_coord.dt, 'month'):
        month_coord = time_coord.dt.month
    else:
        raise ValueError(
            f"Cannot extract 'month' from time coordinate '{time_coord_name}' (dtype: {time_coord.dtype}). "
            "Ensure it is a datetime-like coordinate. If using a non-standard calendar, "
            "make sure the 'cftime' library is installed."
        )

    # Step 3: Filter the data.
    selected_months = SEASON_MONTHS.get(normalized_season)
    if selected_months:
        # .isin() is a powerful and efficient way to select multiple values.
        filtered_data = data_subset.where(month_coord.isin(selected_months), drop=True)
        
        # Check if the filtering resulted in an empty dataset and warn the user.
        if filtered_data[time_coord_name].size == 0:
            warnings.warn(
                f"No data found for season '{season.upper()}' within the dataset's time range.",
                UserWarning
            )
        return filtered_data
    else:
        # Warn about the unsupported season and return the original data.
        supported_seasons = ['annual'] + list(SEASON_MONTHS.keys())
        warnings.warn(
            f"Unknown season '{season}'. Supported options are: {supported_seasons}. Returning unfiltered data.",
            UserWarning
        )
        return data_subset 