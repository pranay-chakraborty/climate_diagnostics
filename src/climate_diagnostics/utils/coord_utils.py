import xarray as xr
import numpy as np
import warnings
from typing import List, Optional, Union, Dict  # 27 Nov: Added type hints support


def get_coord_name(
    xarray_like_obj: Union[xr.DataArray, xr.Dataset], 
    possible_names: List[str]
) -> Optional[str]:  # 27 Nov: Added type annotations
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


def filter_by_season(
    data_subset: Union[xr.DataArray, xr.Dataset], 
    season: str = 'annual'
) -> Union[xr.DataArray, xr.Dataset]:  # 27 Nov: Added type annotations
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
        If a usable time coordinate cannot be found or processed, or if the season is invalid.

    Notes
    -----
    - This function relies on xarray's `.dt` accessor.
    - Note on 'DJF' (Winter): This function filters for Dec, Jan, and Feb based on the 
      month index. It does NOT automatically shift December to align with the Jan/Feb 
      of the following year.
    """
    # Use a constant for season definitions for clarity
    SEASON_MONTHS: Dict[str, List[int]] = {  # 27 Nov: Type annotation for dict
        'jjas': [6, 7, 8, 9],  # Monsoon
        'djf': [12, 1, 2],     # Winter
        'mam': [3, 4, 5],      # Pre-monsoon
        'son': [9, 10, 11],    # Post-monsoon
        'jja': [6, 7, 8]       # Summer
    }
    
    normalized_season = season.lower()
    
    # 27 Nov: Fail fast if season is invalid before doing expensive coordinate lookups
    supported_seasons = ['annual'] + list(SEASON_MONTHS.keys())
    if normalized_season not in supported_seasons:
         raise ValueError(f"Unknown season '{season}'. Supported options: {supported_seasons}")

    if normalized_season == 'annual':
        return data_subset

    # Step 1: Locate the time coordinate.
    time_coord_name = get_coord_name(data_subset, ['time', 't'])
    
    # 27 Nov: Relaxed check. It is sufficient if the name exists in coords, 
    # we don't strictly enforce it being in 'dims' (though it usually is).
    if not time_coord_name:
        raise ValueError("A recognizable time coordinate (e.g., 'time', 't') is required.")

    # Step 2: Use xarray's .dt accessor to get the month.
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
    selected_months = SEASON_MONTHS[normalized_season] # Safe access due to earlier check

    # 27 Nov: Optimized performance. Replaced .where(..., drop=True) with .sel().
    # .where() casts to float and masks data (memory intensive), .sel() simply subsets.
    try:
        mask = month_coord.isin(selected_months)
        filtered_data = data_subset.sel({time_coord_name: mask})
    except Exception as e:
        # Fallback in rare cases where .sel with boolean mask fails on non-dim coords
        filtered_data = data_subset.where(mask, drop=True)

    # Check if the filtering resulted in an empty dataset and warn the user.
    if filtered_data[time_coord_name].size == 0:
        warnings.warn(
            f"No data found for season '{season.upper()}' within the dataset's time range.",
            UserWarning
        )

    return filtered_data