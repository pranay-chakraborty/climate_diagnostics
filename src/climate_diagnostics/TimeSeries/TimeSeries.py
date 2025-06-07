import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from statsmodels.tsa.seasonal import STL

@xr.register_dataset_accessor("climate_timeseries")
class TimeSeriesAccessor:
    """
    Accessor for analyzing and visualizing climate time series from xarray datasets.
    Provides methods for extracting, processing, and visualizing time series
    with support for weighted spatial averaging, seasonal filtering, and time series decomposition.
    """

    # --------------------------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------------------------
    def __init__(self, xarray_obj):
        """Initialize the accessor with a Dataset object."""
        self._obj = xarray_obj

    # --------------------------------------------------------------------------
    # INTERNAL HELPER METHODS: DATA SELECTION & PREPARATION
    # --------------------------------------------------------------------------
    def _get_coord_name(self, possible_names):
        """
        Find the actual coordinate name in the dataset from a list of common alternatives.

        This function checks for coordinate names in a case-sensitive manner first,
        then falls back to a case-insensitive check on the lower-cased names.

        Parameters
        ----------
        possible_names : list of str
            A list of potential coordinate names to search for (e.g., ['lat', 'latitude']).

        Returns
        -------
        str or None
            The matching coordinate name found in the dataset, or None if no match is found.
        """
        if self._obj is None: return None
        for name in possible_names:
            if name in self._obj.coords:
                return name
        coord_names_lower = {name.lower(): name for name in self._obj.coords}
        for name in possible_names:
            if name.lower() in coord_names_lower:
                return coord_names_lower[name.lower()]
        return None

    def _filter_by_season(self, data_subset, season='annual', time_coord_name=None):
        """
        Filter a DataArray or Dataset for a specific meteorological season.

        Supported seasons are 'annual' (no filtering), 'jjas', 'djf', 'mam', 'son'.
        It identifies the month from the time coordinate and filters the data
        to include only the months corresponding to the specified season.

        Parameters
        ----------
        data_subset : xr.DataArray or xr.Dataset
            The data to be filtered, which must have a time dimension.
        season : str, optional
            The season to filter by. Defaults to 'annual'.
        time_coord_name : str, optional
            The name of the time coordinate. If None, it will be auto-detected.

        Returns
        -------
        xr.DataArray or xr.Dataset
            The data filtered for the specified season.

        Raises
        ------
        ValueError
            If a time coordinate cannot be found or if the data cannot be filtered.
        """
        # --- Step 1: Get the time coordinate name ---
        if time_coord_name is None:
            time_coord_name = self._get_coord_name(['time', 't'])
            if time_coord_name is None:
                raise ValueError("Cannot find time coordinate for seasonal filtering.")
        
        # --- Step 2: Handle 'annual' case and prepare for filtering ---
        season_input = season
        season = season.lower()
        if season == 'annual':
            return data_subset
            
        if time_coord_name not in data_subset.dims:
            raise ValueError(f"Cannot filter by season - '{time_coord_name}' dimension not found in the provided data_subset.")

        # --- Step 3: Extract month information from the time coordinate ---
        time_da = data_subset[time_coord_name]
        if 'month' in data_subset.coords and time_coord_name in data_subset['month'].coords:
            month_coord = data_subset['month']
        elif hasattr(time_da.dt, 'month'):
            month_coord = time_da.dt.month
        elif time_da.size > 0 and hasattr(time_da.data[0], 'month'):
            try:
                # Fallback for cftime objects that don't directly support .dt accessor
                time_values = time_da.compute().data if hasattr(time_da, 'chunks') and time_da.chunks else time_da.data
                months_list = [t.month for t in time_values]
                month_coord = xr.DataArray(months_list, coords={time_coord_name: time_da}, dims=[time_coord_name])
                print("Warning: Extracted months from cftime objects manually for seasonal filter.")
            except Exception as e:
                 raise ValueError(f"Time coordinate '{time_coord_name}' (type: {time_da.dtype}) "
                                  f"could not be processed for month extraction for seasonal filter. Error: {e}")
        else:
            raise ValueError(f"Cannot determine month for seasonal filtering from '{time_coord_name}' "
                             f"(dtype: {time_da.dtype}).")

        # --- Step 4: Define seasons and apply the filter ---
        season_months = {'jjas': [6, 7, 8, 9], 'djf': [12, 1, 2], 'mam': [3, 4, 5], 'son': [9, 10, 11]}
        selected_months = season_months.get(season)

        if selected_months:
            filtered_data = data_subset.where(month_coord.isin(selected_months), drop=True)
            if filtered_data[time_coord_name].size == 0:
                print(f"Warning: No data found for season '{season_input.upper()}' within the selected time range.")
            return filtered_data
        else:
            print(f"Warning: Unknown season '{season_input}'. Returning unfiltered data.")
            return data_subset

    def _validate_and_get_sel_slice(self, coord_val_param, data_coord, coord_name_str, is_datetime_intent=False):
        """
        Validate a coordinate selection parameter against the data's coordinate range.

        This helper function checks if the requested selection (a single value, slice, or list)
        is within the bounds of the data's coordinate. It also determines if a 'nearest'
        selection method is appropriate for scalar selections.

        Parameters
        ----------
        coord_val_param : int, float, slice, list, or np.ndarray
            The user-provided selection parameter for the coordinate.
        data_coord : xr.DataArray
            The coordinate DataArray from the dataset to validate against.
        coord_name_str : str
            The name of the coordinate for use in error messages (e.g., "latitude").
        is_datetime_intent : bool, optional
            If True, treat the values as datetime-like for comparison. Defaults to False.

        Returns
        -------
        sel_val : object
            The validated selection parameter, suitable for use with `.sel()`.
        needs_nearest_for_this_coord : bool
            True if the selection is a scalar that would benefit from `method='nearest'`.

        Raises
        ------
        ValueError
            If the requested selection range is entirely outside the data's coordinate range.
        """
        # --- Step 1: Initialize and get data coordinate bounds ---
        min_data_val_raw = data_coord.min().item()
        max_data_val_raw = data_coord.max().item()
        needs_nearest_for_this_coord = False
        sel_val = coord_val_param

        # --- Step 2: Determine requested selection range from input type ---
        comp_req_min, comp_req_max = None, None
        comp_data_min, comp_data_max = min_data_val_raw, max_data_val_raw

        if isinstance(coord_val_param, slice):
            comp_req_min, comp_req_max = coord_val_param.start, coord_val_param.stop
        elif isinstance(coord_val_param, (list, np.ndarray)):
            if not len(coord_val_param): raise ValueError(f"{coord_name_str.capitalize()} selection list/array empty.")
            comp_req_min, comp_req_max = min(coord_val_param), max(coord_val_param)
        else: 
            # Handle scalar selection
            comp_req_min = comp_req_max = coord_val_param
            needs_nearest_for_this_coord = isinstance(coord_val_param, (int, float, np.number))

        # --- Step 3: Convert to datetime objects for comparison if needed ---
        if is_datetime_intent:
            data_dtype = data_coord.dtype
            try:
                if comp_req_min is not None: comp_req_min = np.datetime64(comp_req_min)
                if comp_req_max is not None: comp_req_max = np.datetime64(comp_req_max)
                
                # Handle various raw data types (datetime64, cftime, etc.)
                if np.issubdtype(data_dtype, np.datetime64):
                    if isinstance(min_data_val_raw, (int, np.integer)): # Handle raw integer datetimes
                        unit = np.datetime_data(data_dtype)[0]
                        comp_data_min = np.datetime64(min_data_val_raw, unit)
                        comp_data_max = np.datetime64(max_data_val_raw, unit)
                    else: 
                        comp_data_min = np.datetime64(min_data_val_raw)
                        comp_data_max = np.datetime64(max_data_val_raw)
                elif hasattr(min_data_val_raw, 'year'): # cftime object check
                    comp_data_min = np.datetime64(min_data_val_raw)
                    comp_data_max = np.datetime64(max_data_val_raw)

            except Exception as e:
                print(f"Warning: Could not fully process/validate {coord_name_str} range "
                      f"'{coord_val_param}' against data bounds due to type issues: {e}. "
                      "Relying on xarray's .sel() behavior.")
                comp_data_min, comp_data_max = None, None

        # --- Step 4: Validate that the requested range overlaps with the data range ---
        if comp_data_min is not None and comp_data_max is not None: 
            if comp_req_min is not None and comp_req_min > comp_data_max:
                raise ValueError(f"Requested {coord_name_str} min {coord_val_param} > data max {max_data_val_raw}")
            if comp_req_max is not None and comp_req_max < comp_data_min:
                raise ValueError(f"Requested {coord_name_str} max {coord_val_param} < data min {min_data_val_raw}")
        
        # --- Step 5: Return the validated selection value and 'nearest' flag ---
        return sel_val, needs_nearest_for_this_coord

    def _select_process_data(self, variable, latitude=None, longitude=None, level=None,
                             time_range=None, season='annual', year=None):
        """
        Select, filter, and process a data variable from the dataset.

        This core internal function handles the multi-step process of:
        1. Selecting a variable.
        2. Applying temporal filters (season, year, time_range).
        3. Applying spatial and level selections.
        4. Handling level averaging or selection of a default level.

        Parameters
        ----------
        variable : str
            Name of the data variable to process.
        latitude : float, slice, or list, optional
            Latitude selection.
        longitude : float, slice, or list, optional
            Longitude selection.
        level : float, slice, or list, optional
            Level selection. If a slice or list, the data is averaged over the levels.
        time_range : slice, optional
            Time range selection.
        season : str, optional
            Seasonal filter to apply. Defaults to 'annual'.
        year : int, optional
            Specific year to filter the data for.

        Returns
        -------
        xr.DataArray
            The processed DataArray after all selections and operations.

        Raises
        ------
        ValueError
            If the variable is not found or if selections result in no data.
        """
        # --- Step 1: Validate variable and get the DataArray ---
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found. Available: {list(self._obj.data_vars.keys())}")
        
        data_var = self._obj[variable]

        # --- Step 2: Apply temporal filters (season, year, time_range) ---
        time_name = self._get_coord_name(['time', 't'])
        if time_name and time_name in data_var.dims:
            # Apply seasonal filter first
            if season.lower() != 'annual':
                data_var = self._filter_by_season(data_var, season, time_name)
                if data_var[time_name].size == 0:
                    raise ValueError(f"No data available after season ('{season}') filter.")
            
            # Apply year filter
            if year is not None:
                try: # Fast path for standard datetime types
                    year_match_bool = data_var[time_name].dt.year == year
                except (AttributeError, TypeError): # Slower fallback for cftime
                    year_match_bool = xr.DataArray(
                        [t.year == year for t in data_var[time_name].compute().data],
                        coords={time_name: data_var[time_name]}, dims=[time_name]
                    )
                data_var = data_var.sel({time_name: year_match_bool})
                if data_var[time_name].size == 0:
                    raise ValueError(f"No data for year {year} (after season '{season}' filter).")

            # Apply time range slice
            if time_range is not None:
                sel_val, _ = self._validate_and_get_sel_slice(time_range, data_var[time_name], "time", True)
                data_var = data_var.sel({time_name: sel_val})
                if data_var[time_name].size == 0:
                    raise ValueError("No data after time_range selection.")
        elif season.lower() != 'annual' or year is not None or time_range is not None :
             print(f"Warning: Temporal filters (season, year, time_range) requested, "
                   f"but time dimension ('{time_name}') not found or not a dimension in variable '{variable}'.")

        # --- Step 3: Prepare spatial and level selections ---
        selection_dict = {}
        method_dict = {}

        # Latitude selection
        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        if latitude is not None and lat_name and lat_name in data_var.coords:
            sel_val, needs_nearest = self._validate_and_get_sel_slice(latitude, data_var[lat_name], "latitude")
            selection_dict[lat_name] = sel_val
            if needs_nearest: method_dict[lat_name] = 'nearest'

        # Longitude selection
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        if longitude is not None and lon_name and lon_name in data_var.coords:
            sel_val, needs_nearest = self._validate_and_get_sel_slice(longitude, data_var[lon_name], "longitude")
            selection_dict[lon_name] = sel_val
            if needs_nearest: method_dict[lon_name] = 'nearest'
        
        # --- Step 4: Handle vertical level selection or averaging ---
        level_name = self._get_coord_name(['level', 'lev', 'plev', 'height', 'altitude', 'depth', 'z'])
        if level_name and level_name in data_var.dims:
            if level is not None:
                if isinstance(level, (slice, list, np.ndarray)): 
                    # Average over a range of levels
                    sel_val, _ = self._validate_and_get_sel_slice(level, data_var[level_name], "level")
                    print(f"Averaging over levels: {level}")
                    with xr.set_options(keep_attrs=True):
                        data_to_avg = data_var.sel({level_name: sel_val})
                        if level_name in data_to_avg.dims and data_to_avg.sizes[level_name] > 1:
                             data_var = data_to_avg.mean(dim=level_name)
                        else: # Only one level was selected by the slice
                             data_var = data_to_avg
                else: 
                    # Select a single level
                    sel_val, needs_nearest = self._validate_and_get_sel_slice(level, data_var[level_name], "level")
                    selection_dict[level_name] = sel_val
                    if needs_nearest: method_dict[level_name] = 'nearest'
            elif data_var.sizes[level_name] > 1: 
                # Default to first level if multiple exist and none are specified
                first_level_val = data_var[level_name].isel({level_name: 0}).item()
                selection_dict[level_name] = first_level_val
                print(f"Warning: Multiple levels found in '{variable}'. Using first level: {first_level_val}")
        elif level is not None:
            print(f"Warning: Level dimension '{level_name}' not found or not a dimension in '{variable}'. Ignoring 'level' parameter.")

        # --- Step 5: Apply all spatial and single-level selections ---
        if selection_dict:
            if any(isinstance(v, slice) for v in selection_dict.values()) and method_dict:
                print("Note: Applying selections. Slices will be used directly, 'nearest' for scalar points if specified.")
            try:
                data_var = data_var.sel(selection_dict, method=method_dict if method_dict else None)
            except Exception as e:
                print(f"Error during final .sel() operation: {e}")
                print(f"Selection dictionary: {selection_dict}, Method dictionary: {method_dict}")
                raise

        # --- Step 6: Final validation and return ---
        if data_var.size == 0:
            print("Warning: Selection resulted in an empty DataArray.")
        return data_var

    # --------------------------------------------------------------------------
    # INTERNAL HELPER METHODS: CALCULATIONS
    # --------------------------------------------------------------------------
    def _get_spatial_mean(self, data_var, area_weighted=True):
        """
        Calculate the spatial mean of a DataArray.

        The mean can be simple or area-weighted based on the cosine of the latitude.
        It operates over any available latitude and longitude dimensions.

        Parameters
        ----------
        data_var : xr.DataArray
            The input DataArray, which should have spatial dimensions (lat, lon).
        area_weighted : bool, optional
            If True, performs an area-weighted mean using latitude weights.
            Defaults to True.

        Returns
        -------
        xr.DataArray
            The DataArray after spatial averaging. If no spatial dimensions are
            present, the original DataArray is returned.
        """
        # --- Step 1: Find standard names for spatial coordinates ---
        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        
        # --- Step 2: Identify which spatial dimensions are present in the DataArray ---
        spatial_dims_present = []
        if lat_name and lat_name in data_var.dims:
            spatial_dims_present.append(lat_name)
        if lon_name and lon_name in data_var.dims:
            spatial_dims_present.append(lon_name)

        # --- Step 3: Return original data if no spatial dimensions are found ---
        if not spatial_dims_present:
            return data_var

        # --- Step 4: Calculate mean, applying area weighting if requested ---
        if area_weighted and lat_name in spatial_dims_present:
            # Use latitude-based weights for a more accurate spatial average
            weights = np.cos(np.deg2rad(data_var[lat_name]))
            weights.name = "weights"
            print("Calculating area-weighted spatial mean.")
            return data_var.weighted(weights).mean(dim=spatial_dims_present, skipna=True)
        else:
            # Calculate a simple, unweighted mean
            weight_msg = "(unweighted)" if lat_name in spatial_dims_present and not area_weighted else ""
            print(f"Calculating simple spatial mean {weight_msg}.")
            return data_var.mean(dim=spatial_dims_present, skipna=True)

    def _calculate_consecutive_days_index(self, data_var, time_name, threshold_val, is_wet_spell):
        """
        Calculate the annual maximum number of consecutive days meeting a condition.

        This helper function is used for CWD (Consecutive Wet Days) and CDD
        (Consecutive Dry Days) calculations. It takes a 1D time series,
        identifies consecutive days above/below a threshold within each year,
        and returns the maximum streak for each year.

        Parameters
        ----------
        data_var : xr.DataArray
            A 1D DataArray with a time dimension.
        time_name : str
            The name of the time coordinate.
        threshold_val : float
            The threshold value for the condition.
        is_wet_spell : bool
            If True, counts days >= threshold (wet). If False, counts days < threshold (dry).

        Returns
        -------
        xr.DataArray
            A DataArray containing the maximum consecutive day count for each year.
            The coordinate is 'year_group'.
        """
        # --- Step 1: Validate input and create boolean condition array ---
        if not isinstance(data_var, xr.DataArray) or data_var.ndim != 1 or time_name not in data_var.dims:
            raise ValueError("Input must be a 1D DataArray with a time dimension.")
        
        cond = (data_var >= threshold_val) if is_wet_spell else (data_var < threshold_val)
        
        # --- Step 2: Ensure data is in memory and get year for each time point ---
        if hasattr(cond, 'chunks') and cond.chunks:
            cond = cond.compute() 
        
        arr = cond.data
        
        time_coord_from_cond = cond[time_name]
        years = time_coord_from_cond.dt.year.data 
            
        # --- Step 3: Loop through each year to find the max consecutive streak ---
        unique_years = np.unique(years)
        counts = np.zeros(unique_years.shape, dtype=int)

        for idx, yr_val in enumerate(unique_years):
            mask_for_year = (years == yr_val)
            year_data_boolean = arr[mask_for_year]

            if not year_data_boolean.size: 
                counts[idx] = 0
                continue
            
            # Find the longest run of `True` values in the boolean array for the year
            max_streak_for_year = 0
            current_streak_for_year = 0
            for val_bool in year_data_boolean:
                if bool(val_bool): 
                    current_streak_for_year += 1
                else: 
                    max_streak_for_year = max(max_streak_for_year, current_streak_for_year)
                    current_streak_for_year = 0
            # Account for a streak that runs to the end of the year
            max_streak_for_year = max(max_streak_for_year, current_streak_for_year)
            counts[idx] = max_streak_for_year
            
        # --- Step 4: Return result as a new DataArray with 'year_group' coordinate ---
        return xr.DataArray(
            data=counts,
            coords={'year_group': unique_years},
            dims=['year_group']
        )

    # ==============================================================================
    # PUBLIC PLOTTING METHODS
    # ==============================================================================

    # --------------------------------------------------------------------------
    # A. Basic Time Series Plots
    # --------------------------------------------------------------------------
    def plot_time_series(self, variable='air', latitude=None, longitude=None, level=None,
                         time_range=None, season='annual', year=None,
                         area_weighted=True, figsize=(16, 10), save_plot_path=None):
        """
        Plot a time series of a spatially averaged variable.

        This function selects data for a given variable, performs spatial averaging
        over the specified domain, and plots the resulting time series.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to plot. Defaults to 'air'.
        latitude : float, slice, or list, optional
            Latitude range for spatial averaging.
        longitude : float, slice, or list, optional
            Longitude range for spatial averaging.
        level : float, slice, or list, optional
            Vertical level selection.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            Filter for a specific year.
        area_weighted : bool, optional
            If True, use latitude-based area weighting for the spatial mean. Defaults to True.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        save_plot_path : str or None, optional
            If provided, the path to save the plot figure.

        Returns
        -------
        matplotlib.axes.Axes or None
            The Axes object of the plot, or None if no data could be plotted.
        """
        # --- Step 1: Select and process the data ---
        data_selected = self._select_process_data(
            variable, latitude, longitude, level, time_range, season, year
        )
        time_name = self._get_coord_name(['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found for time series plot.")
        if data_selected.size == 0:
            print("No data to plot after selections.")
            return None

        # --- Step 2: Calculate the spatial mean time series ---
        ts_data = self._get_spatial_mean(data_selected, area_weighted)

        # --- Step 3: Ensure data is in memory for plotting ---
        if hasattr(ts_data, 'chunks') and ts_data.chunks:
            print("Computing time series...")
            with ProgressBar(): ts_data = ts_data.compute()
        
        if ts_data.size == 0 :
            print("Time series is empty after spatial averaging.")
            return None

        # --- Step 4: Create the plot ---
        plt.figure(figsize=figsize)
        ts_data.plot(marker='.')
        ax = plt.gca()

        # --- Step 5: Customize plot labels and title ---
        units = data_selected.attrs.get("units", "")
        long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
        ax.set_ylabel(f"{long_name} ({units})")
        ax.set_xlabel('Time')

        season_display = season.upper() if season.lower() != 'annual' else 'Annual'
        year_display = f" for {year}" if year is not None else ""
        weight_display = "Area-Weighted " if area_weighted and self._get_coord_name(['lat', 'latitude']) in data_selected.dims else ""
        ax.set_title(f"{season_display}{year_display}: {weight_display}Spatial Mean Time Series of {long_name} ({units})")

        # --- Step 6: Finalize and save the plot ---
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_plot_path:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        return ax

    def plot_std_space(self, variable='air', latitude=None, longitude=None, level=None,
                       time_range=None, season='annual', year=None,
                       area_weighted=True, figsize=(16, 10), save_plot_path=None, title=None):
        """
        Plot a time series of the spatial standard deviation of a variable.

        This function calculates the standard deviation across the spatial domain
        for each time step and plots the resulting time series. This can be used
        to analyze the spatial variability of a field over time.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to plot. Defaults to 'air'.
        latitude : float, slice, or list, optional
            Latitude range for the calculation.
        longitude : float, slice, or list, optional
            Longitude range for the calculation.
        level : float, slice, or list, optional
            Vertical level selection.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            Filter for a specific year.
        area_weighted : bool, optional
            If True, use latitude-based area weighting for the standard deviation. Defaults to True.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        save_plot_path : str or None, optional
            If provided, the path to save the plot figure.
        title : str or None, optional
            Custom plot title. A default title is generated if not provided.

        Returns
        -------
        matplotlib.axes.Axes or None
            The Axes object of the plot, or None if no data could be plotted.
        """
        # --- Step 1: Select and process the data ---
        data_selected = self._select_process_data(
             variable, latitude, longitude, level, time_range, season, year
        )
        time_name = self._get_coord_name(['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found for spatial standard deviation plot.")
        if data_selected.size == 0:
            print("No data to plot after selections.")
            return None

        # --- Step 2: Calculate the time series of spatial standard deviation ---
        lat_name = self._get_coord_name(['lat', 'latitude'])
        lon_name = self._get_coord_name(['lon', 'longitude'])
        spatial_dims = [d for d in [lat_name, lon_name] if d and d in data_selected.dims]
        if not spatial_dims:
            raise ValueError("No spatial dimensions found for standard deviation calculation.")

        std_ts_data = None
        if area_weighted and lat_name in spatial_dims:
            # Weighted standard deviation
            weights = np.cos(np.deg2rad(data_selected[lat_name]))
            weights.name = "weights"
            std_ts_data = data_selected.weighted(weights).std(dim=spatial_dims, skipna=True)
            print("Calculating area-weighted spatial standard deviation time series.")
        else:
            # Unweighted standard deviation
            std_ts_data = data_selected.std(dim=spatial_dims, skipna=True)
            weight_msg = "(unweighted)" if lat_name in spatial_dims else ""
            print(f"Calculating simple spatial standard deviation {weight_msg} time series.")

        # --- Step 3: Ensure data is in memory for plotting ---
        if hasattr(std_ts_data, 'chunks') and std_ts_data.chunks:
            print("Computing spatial standard deviation time series...")
            with ProgressBar(): std_ts_data = std_ts_data.compute()
        
        if std_ts_data.size == 0:
            print("Spatial standard deviation time series is empty.")
            return None

        # --- Step 4: Create the plot ---
        plt.figure(figsize=figsize)
        std_ts_data.plot(marker='.')
        ax = plt.gca()

        # --- Step 5: Customize plot labels and title ---
        units = data_selected.attrs.get("units", "")
        long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
        ax.set_ylabel(f"Spatial Std. Dev. ({units})" if units else "Spatial Std. Dev.")
        ax.set_xlabel('Time')

        if title is None:
            season_display = season.upper() if season.lower() != 'annual' else 'Annual'
            year_display = f" for {year}" if year is not None else ""
            weight_display = "Area-Weighted " if area_weighted and lat_name in spatial_dims else ""
            title = f"{season_display}{year_display}: Time Series of {weight_display}Spatial Std Dev of {long_name} ({units})"
        ax.set_title(title)

        # --- Step 6: Finalize and save the plot ---
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_plot_path:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        return ax

    # --------------------------------------------------------------------------
    # B. Climate Indices and Anomaly Plots
    # --------------------------------------------------------------------------
    def plot_consecutive_wet_days_time_series(self, variable='prate', threshold=None, threshold_percentile=None,
                                              latitude=None, longitude=None, level=None, time_range=None,
                                              season='annual', year=None,
                                              area_weighted=True, figsize=(16, 10),
                                              save_plot_path=None, title=None):
        """
        Plot a time series of the annual maximum Consecutive Wet Days (CWD).

        CWD is the longest period of consecutive days where a variable (e.g., precipitation)
        is at or above a given threshold. This function first computes a spatial mean
        of the variable, then calculates the CWD index for each year in the time series.

        The threshold can be an absolute value or determined by a percentile of wet days.
        When using a percentile, the threshold is calculated from the distribution of
        all wet days (> 0) in the selected time period to provide a consistent baseline.

        Parameters
        ----------
        variable : str, optional
            Name of the variable. Defaults to 'prate'.
        threshold : float, optional
            Absolute threshold for defining a wet day. If `threshold_percentile` is also
            provided, `threshold_percentile` will take precedence.
        threshold_percentile : int, optional
            Percentile (0-100) to define the wet day threshold, calculated from all days
            with a value > 0 in the selected period.
        latitude : float, slice, or list, optional
            Latitude range for spatial averaging.
        longitude : float, slice, or list, optional
            Longitude range for spatial averaging.
        level : float, slice, or list, optional
            Vertical level selection.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            Filter for a specific year.
        area_weighted : bool, optional
            If True, use area weighting for the spatial mean. Defaults to True.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        save_plot_path : str or None, optional
            Path to save the plot.
        title : str or None, optional
            Custom plot title.

        Returns
        -------
        matplotlib.axes.Axes or None
            The Axes object of the plot, or None if no data could be plotted.
        """
        # --- Step 1: Select and process the data ---
        data_selected = self._select_process_data(
            variable, latitude, longitude, level, time_range, season, year
        )
        time_name = self._get_coord_name(['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found.")
        if data_selected.size == 0: print("No data after selections."); return None

        # --- Step 2: Calculate the spatial mean time series ---
        ts_spatial_mean = self._get_spatial_mean(data_selected, area_weighted)
        if hasattr(ts_spatial_mean, 'chunks') and ts_spatial_mean.chunks : ts_spatial_mean = ts_spatial_mean.compute()
        if ts_spatial_mean.size == 0 : print("Time series empty after spatial mean."); return None

        # --- Step 3: Determine the threshold for a wet day ---
        final_threshold = threshold
        threshold_info_str = ""

        if threshold_percentile is not None:
            # Percentile-based threshold takes precedence
            if threshold is not None:
                print("Warning: Both 'threshold' and 'threshold_percentile' provided. Using 'threshold_percentile'.")

            # Calculate percentile on "wet" days only (>0) to avoid skew from dry days
            wet_days_for_percentile_calc = ts_spatial_mean.where(ts_spatial_mean > 0)
            if wet_days_for_percentile_calc.count() == 0:
                raise ValueError("Cannot calculate percentile threshold because there are no wet days (>0) in the time series.")
            
            final_threshold = wet_days_for_percentile_calc.quantile(
                threshold_percentile / 100.0, dim=time_name, skipna=True
            ).item()
            
            threshold_info_str = (f"{threshold_percentile}th %-ile of wet days: {final_threshold:.2f} "
                                  f"{data_selected.attrs.get('units','')}")
            print(f"Using threshold from percentile: {threshold_info_str}")

        elif threshold is None:
            raise ValueError("You must provide either 'threshold' or 'threshold_percentile'.")
        else:
            # Use the absolute threshold provided
            threshold_info_str = f"Abs Th: {threshold:.2f} {data_selected.attrs.get('units','')}"

        if final_threshold is None:
            raise ValueError("Threshold for CWD could not be determined.")

        # --- Step 4: Calculate the CWD index ---
        plot_data = self._calculate_consecutive_days_index(
            ts_spatial_mean, time_name, final_threshold, is_wet_spell=True
        )

        if plot_data.size == 0:
            print("No data to plot after CWD calculation.")
            return None

        # --- Step 5: Create the plot ---
        plt.figure(figsize=figsize)
        plot_data.plot(marker='o')
        ax = plt.gca()
        long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
        ax.set_ylabel("Max Consecutive Wet Days")
        ax.set_xlabel("Year")

        # --- Step 6: Customize title and finalize plot ---
        if title is None:
            season_display = season.upper() if season.lower() != 'annual' else 'Annual'
            year_filter_display = f" for year {year}" if year is not None else ""
            title = f"{season_display}{year_filter_display} Max Consecutive Wet Days\n{long_name} ({threshold_info_str})"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        if save_plot_path: plt.savefig(save_plot_path, bbox_inches='tight', dpi=300); print(f"Plot saved to: {save_plot_path}")
        return ax

    def plot_consecutive_dry_days_time_series(self, variable='prate', threshold=1.0,
                                              latitude=None, longitude=None, level=None, time_range=None,
                                              season='annual', year=None,
                                              area_weighted=True, figsize=(16, 10),
                                              save_plot_path=None, title=None):
        """
        Plot a time series of the annual maximum Consecutive Dry Days (CDD).

        CDD is the longest period of consecutive days where a variable (e.g., precipitation)
        is below a given threshold. This function first computes a spatial mean
        of the variable, then calculates the CDD index for each year. A dry day is commonly
        defined as a day with precipitation < 1.0 mm/day.

        Parameters
        ----------
        variable : str, optional
            Name of the variable. Defaults to 'prate'.
        threshold : float, optional
            Absolute threshold for defining a dry day (i.e., days with value < threshold).
            Defaults to 1.0, a common definition for precipitation.
        latitude : float, slice, or list, optional
            Latitude range for spatial averaging.
        longitude : float, slice, or list, optional
            Longitude range for spatial averaging.
        level : float, slice, or list, optional
            Vertical level selection.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            Filter for a specific year.
        area_weighted : bool, optional
            If True, use area weighting for the spatial mean. Defaults to True.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        save_plot_path : str or None, optional
            Path to save the plot.
        title : str or None, optional
            Custom plot title.

        Returns
        -------
        matplotlib.axes.Axes or None
            The Axes object of the plot, or None if no data could be plotted.
        """
        # --- Step 1: Select and process the data ---
        data_selected = self._select_process_data(
            variable, latitude, longitude, level, time_range, season, year
        )
        time_name = self._get_coord_name(['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found.")
        if data_selected.size == 0: print("No data after selections."); return None

        # --- Step 2: Calculate the spatial mean time series ---
        ts_spatial_mean = self._get_spatial_mean(data_selected, area_weighted)
        if hasattr(ts_spatial_mean, 'chunks') and ts_spatial_mean.chunks : ts_spatial_mean = ts_spatial_mean.compute()
        if ts_spatial_mean.size == 0 : print("Time series empty after spatial mean."); return None

        # --- Step 3: Validate the threshold ---
        if threshold is None:
             raise ValueError("A 'threshold' must be provided to define a dry day (e.g., threshold=1.0).")

        # --- Step 4: Calculate the CDD index ---
        plot_data = self._calculate_consecutive_days_index(
            ts_spatial_mean, time_name, threshold, is_wet_spell=False
        )
        
        if plot_data.size == 0:
            print("No data to plot after CDD calculation.")
            return None

        # --- Step 5: Create the plot ---
        plt.figure(figsize=figsize)
        plot_data.plot(marker='o')
        ax = plt.gca()
        long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
        ax.set_ylabel("Max Consecutive Dry Days")
        ax.set_xlabel("Year")

        # --- Step 6: Customize title and finalize plot ---
        if title is None:
            season_display = season.upper() if season.lower() != 'annual' else 'Annual'
            year_filter_display = f" for year {year}" if year is not None else ""
            thresh_disp = f"Threshold: < {threshold:.2f} {data_selected.attrs.get('units','')}"
            title = f"{season_display}{year_filter_display} Max Consecutive Dry Days\n{long_name} ({thresh_disp})"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        if save_plot_path: plt.savefig(save_plot_path, bbox_inches='tight', dpi=300); print(f"Plot saved to: {save_plot_path}")
        return ax

    def plot_precipitation_anomalies(self, variable='prate', latitude=None, longitude=None, level=None,
                                     time_range=None, season='annual', year=None,
                                     area_weighted=True, figsize=(16, 10),
                                     save_plot_path=None, title=None):
        """
        Plot a time series of precipitation anomalies.

        Anomalies are calculated as the deviation from the long-term mean of the
        selected data. If the time series spans multiple years, the anomalies are
        averaged annually and plotted as a bar chart. If a single year is selected,
        the daily or monthly anomalies are plotted as a line chart.

        Parameters
        ----------
        variable : str, optional
            Name of the precipitation variable. Defaults to 'prate'.
        latitude : float, slice, or list, optional
            Latitude range for spatial averaging.
        longitude : float, slice, or list, optional
            Longitude range for spatial averaging.
        level : float, slice, or list, optional
            Vertical level selection.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            If specified, plot anomalies for this year only. Otherwise, plot annual mean anomalies.
        area_weighted : bool, optional
            If True, use area weighting for the spatial mean. Defaults to True.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        save_plot_path : str or None, optional
            Path to save the plot.
        title : str or None, optional
            Custom plot title.

        Returns
        -------
        matplotlib.axes.Axes or None
            The Axes object of the plot, or None if no data could be plotted.
        """
        # --- Step 1: Select and process the data ---
        data_selected = self._select_process_data(
            variable, latitude, longitude, level, time_range, season, year
        )
        time_name = self._get_coord_name(['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found.")
        if data_selected.size == 0: print("No data after selections."); return None

        # --- Step 2: Calculate the spatial mean time series ---
        ts_spatial_mean = self._get_spatial_mean(data_selected, area_weighted)
        if hasattr(ts_spatial_mean, 'chunks') and ts_spatial_mean.chunks : ts_spatial_mean = ts_spatial_mean.compute()
        if ts_spatial_mean.size == 0 : print("Time series empty after spatial mean."); return None

        # --- Step 3: Calculate anomalies from the long-term mean ---
        long_term_mean_for_anomaly = ts_spatial_mean.mean(dim=time_name, skipna=True)
        anomalies = ts_spatial_mean - long_term_mean_for_anomaly

        # --- Step 4: Group anomalies by year if plotting a multi-year time series ---
        if year is None: 
            # Calculate annual mean anomalies
            years_coord = anomalies[time_name].dt.year
            plot_data = anomalies.groupby(years_coord).mean(dim=time_name, skipna=True).rename({'year':'year_group'})
        else: 
            # Use raw (e.g., daily/monthly) anomalies for a single year plot
            plot_data = anomalies

        # --- Step 5: Create the plot ---
        plt.figure(figsize=figsize)
        plot_data.plot(marker='o' if year is None else '.')
        ax = plt.gca()
        units = data_selected.attrs.get("units", "")
        long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
        ax.set_ylabel(f"Precipitation Anomaly ({units})")
        ax.set_xlabel("Year" if year is None else "Time")

        # --- Step 6: Customize title and finalize plot ---
        if title is None:
            season_display = season.upper() if season.lower() != 'annual' else 'Annual'
            year_display = f" for {year}" if year is not None else ""
            title = f"{season_display}{year_display} Precipitation Anomalies\n{long_name}"
        ax.set_title(title)
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        if save_plot_path: plt.savefig(save_plot_path, bbox_inches='tight', dpi=300); print(f"Plot saved to: {save_plot_path}")
        return ax

    def plot_extreme_precipitation_events(self, variable='prate', percentile=95,
                                          latitude=None, longitude=None, level=None, time_range=None,
                                          season='annual', year=None,
                                          area_weighted=True, figsize=(16, 10),
                                          save_plot_path=None, title=None):
        """
        Plot a time series of the count of extreme precipitation events.

        An extreme event is defined as a day where the spatially-averaged precipitation
        exceeds a given percentile threshold. The threshold is calculated over the entire
        selected time series. The function then plots the annual count of these events.

        Parameters
        ----------
        variable : str, optional
            Name of the precipitation variable. Defaults to 'prate'.
        percentile : int, optional
            The percentile (0-100) to use as the threshold for extreme events. Defaults to 95.
        latitude : float, slice, or list, optional
            Latitude range for spatial averaging.
        longitude : float, slice, or list, optional
            Longitude range for spatial averaging.
        level : float, slice, or list, optional
            Vertical level selection.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            If specified, calculate the count for only that year. Otherwise, calculates annual counts.
        area_weighted : bool, optional
            If True, use area weighting for the spatial mean. Defaults to True.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        save_plot_path : str or None, optional
            Path to save the plot.
        title : str or None, optional
            Custom plot title.

        Returns
        -------
        matplotlib.axes.Axes or None
            The Axes object of the plot, or None if no data could be plotted.
        """
        # --- Step 1: Select and process the data ---
        data_selected = self._select_process_data(
            variable, latitude, longitude, level, time_range, season, year
        )
        time_name = self._get_coord_name(['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found.")
        if data_selected.size == 0: print("No data after selections."); return None
        
        # --- Step 2: Calculate the spatial mean time series ---
        ts_spatial_mean = self._get_spatial_mean(data_selected, area_weighted)
        if hasattr(ts_spatial_mean, 'chunks') and ts_spatial_mean.chunks : ts_spatial_mean = ts_spatial_mean.compute()
        if ts_spatial_mean.size == 0 : print("Time series empty after spatial mean."); return None

        # --- Step 3: Determine the percentile threshold and identify extreme events ---
        actual_threshold = ts_spatial_mean.quantile(percentile / 100.0, dim=time_name, skipna=True).item()
        print(f"Using {percentile}th percentile threshold: {actual_threshold:.2f} {data_selected.attrs.get('units','')} (calculated on selected data)")

        extreme_event_mask = ts_spatial_mean > actual_threshold

        # --- Step 4: Count events per year ---
        if year is None:
            # Group by year and sum the boolean mask to get annual counts
            years_coord = extreme_event_mask[time_name].dt.year
            plot_data = extreme_event_mask.groupby(years_coord).sum(dim=time_name, skipna=True).rename({'year':'year_group'})
        else: 
            # Count for a single specified year
            plot_data_scalar = extreme_event_mask.sum(dim=time_name, skipna=True)
            year_val_for_plot = year if isinstance(year, int) else int(str(year)) 
            plot_data = xr.DataArray([plot_data_scalar.item()], coords={'year_group': [year_val_for_plot]}, dims=['year_group'])

        # --- Step 5: Create the plot ---
        plt.figure(figsize=figsize)
        plot_data.plot(marker='o')
        ax = plt.gca()
        long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
        ax.set_ylabel(f"Number of Extreme Events (>{percentile}th %-ile)")
        ax.set_xlabel("Year")

        # --- Step 6: Customize title and finalize plot ---
        if title is None:
            season_display = season.upper() if season.lower() != 'annual' else 'Annual'
            year_filter_display = f" for year {year}" if year is not None else ""
            title = f"{season_display}{year_filter_display} Count of Extreme Precipitation Events\n{long_name}"
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
        if save_plot_path: plt.savefig(save_plot_path, bbox_inches='tight', dpi=300); print(f"Plot saved to: {save_plot_path}")
        return ax

    # --------------------------------------------------------------------------
    # C. Time Series Decomposition
    # --------------------------------------------------------------------------
    def decompose_time_series(self, variable='air', level=None, latitude=None, longitude=None,
                              time_range=None, season='annual', year=None,
                              stl_seasonal=13, stl_period=12, area_weighted=True,
                              plot_results=True, figsize=(16, 10), save_plot_path=None):
        """
        Decompose a time series into trend, seasonal, and residual components using STL.

        Seasonal-Trend decomposition using LOESS (STL) is a robust method for
        decomposing a time series. This function first creates a spatially-averaged
        time series and then applies the STL algorithm.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to decompose. Defaults to 'air'.
        level : float, slice, or list, optional
            Vertical level selection.
        latitude : float, slice, or list, optional
            Latitude range for spatial averaging.
        longitude : float, slice, or list, optional
            Longitude range for spatial averaging.
        time_range : slice, optional
            Time range for the series.
        season : str, optional
            Seasonal filter. Defaults to 'annual'.
        year : int, optional
            Filter for a specific year.
        stl_seasonal : int, optional
            Length of the seasonal smoother for STL. Must be an odd integer. Defaults to 13.
        stl_period : int, optional
            The period of the seasonal component. For monthly data, this is typically 12. Defaults to 12.
        area_weighted : bool, optional
            If True, use area weighting for the spatial mean. Defaults to True.
        plot_results : bool, optional
            If True, plot the original series and its decomposed components. Defaults to True.
        figsize : tuple, optional
            Figure size for the plot. Defaults to (16, 10).
        save_plot_path : str or None, optional
            Path to save the decomposition plot.

        Returns
        -------
        dict or (dict, matplotlib.figure.Figure)
            If `plot_results` is False, returns a dictionary containing the
            'original', 'trend', 'seasonal', and 'residual' components as pandas Series.
            If `plot_results` is True, returns a tuple of (dictionary, figure object).
        """
        # --- Step 1: Select and process data for the time series ---
        data_selected = self._select_process_data(
             variable, latitude, longitude, level, time_range, season, year
        )
        time_name = self._get_coord_name(['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension required for decomposition.")
        if data_selected.size == 0: print("No data after selections."); return (None, None) if plot_results else None

        # --- Step 2: Compute the spatially-averaged time series ---
        ts_spatial_mean = self._get_spatial_mean(data_selected, area_weighted)
        if hasattr(ts_spatial_mean, 'chunks') and ts_spatial_mean.chunks:
            print("Computing mean time series for decomposition...")
            with ProgressBar(): ts_spatial_mean = ts_spatial_mean.compute()
        if ts_spatial_mean.size == 0 : print("Time series empty after spatial mean."); return (None, None) if plot_results else None
        
        # --- Step 3: Convert the xarray DataArray to a pandas Series for STL ---
        try:
            ts_pd = ts_spatial_mean.to_series()
        except Exception:
            # Handle cases where the data is not a simple 1D series
            if ts_spatial_mean.ndim > 1:
                squeezed_dims = [d for d in ts_spatial_mean.dims if d != time_name and ts_spatial_mean.sizes[d] == 1]
                if squeezed_dims:
                    ts_spatial_mean_squeezed = ts_spatial_mean.squeeze(dim=squeezed_dims)
                    if ts_spatial_mean_squeezed.ndim == 1:
                        ts_pd = ts_spatial_mean_squeezed.to_series()
                    else:
                         raise ValueError(f"Spatially averaged data for STL still has >1 non-squeezable dimension: {ts_spatial_mean.dims}")
                else:
                    raise ValueError(f"Spatially averaged data for STL still has >1 dimension: {ts_spatial_mean.dims}")

            elif ts_spatial_mean.ndim == 1 and time_name in ts_spatial_mean.coords:
                 try: # Fallback for cftime index
                    ts_pd = pd.Series(ts_spatial_mean.data, index=pd.to_datetime(ts_spatial_mean[time_name].data))
                 except Exception as e_pd:
                     raise ValueError(f"Could not convert to pandas Series for STL: {e_pd}")
            else:
                 raise ValueError("Data for STL is not a 1D time series.")

        # --- Step 4: Prepare the time series for STL (drop NaNs, check length) ---
        ts_pd = ts_pd.dropna()
        if ts_pd.empty:
            raise ValueError("Time series is empty or all NaN after processing for STL.")
        if len(ts_pd) <= 2 * stl_period: 
            raise ValueError(f"Time series length ({len(ts_pd)}) must be > 2 * stl_period ({2*stl_period}) for STL.")
        
        # --- Step 5: Perform STL decomposition ---
        if stl_seasonal % 2 == 0:
            stl_seasonal +=1
            print(f"Adjusted stl_seasonal to be odd: {stl_seasonal}")

        print(f"Performing STL decomposition (period={stl_period}, seasonal_smooth={stl_seasonal})...")
        try:
            stl_result = STL(ts_pd, seasonal=stl_seasonal, period=stl_period, robust=True).fit()
        except Exception as e:
             print(f"STL decomposition failed: {e}. Check time series properties (length, NaNs, period).")
             raise

        results_dict = {
            'original': stl_result.observed,
            'trend': stl_result.trend,
            'seasonal': stl_result.seasonal,
            'residual': stl_result.resid
        }

        # --- Step 6: Plot the results if requested ---
        if plot_results:
            print("Plotting decomposition results...")
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            units = data_selected.attrs.get("units", "")
            long_name = data_selected.attrs.get("long_name", variable.replace('_', ' ').capitalize())
            
            # Plot each component
            axes[0].plot(results_dict['original'].index, results_dict['original'].values, label='Observed')
            axes[0].set_ylabel(f"Observed ({units})")
            title_prefix = f'{season.upper() if season.lower() != "annual" else "Annual"}'
            year_info = f" for {year}" if year else ""
            axes[0].set_title(f'{title_prefix}{year_info} Time Series Decomposition: {long_name}')

            axes[1].plot(results_dict['trend'].index, results_dict['trend'].values, label='Trend')
            axes[1].set_ylabel(f"Trend ({units})")
            axes[2].plot(results_dict['seasonal'].index, results_dict['seasonal'].values, label='Seasonal')
            axes[2].set_ylabel(f"Seasonal ({units})")
            axes[3].plot(results_dict['residual'].index, results_dict['residual'].values, label='Residual', marker='.', linestyle='None', markersize=3, alpha=0.7)
            axes[3].axhline(0, color='grey', linestyle='--', alpha=0.5)
            axes[3].set_ylabel(f"Residual ({units})")
            axes[3].set_xlabel("Time")

            # Finalize and save the plot
            for ax_i in axes:
                ax_i.grid(True, linestyle='--', alpha=0.6)
                ax_i.legend(loc='upper left', fontsize='small')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            if save_plot_path:
                plt.savefig(save_plot_path, bbox_inches='tight', dpi=300); print(f"Plot saved to: {save_plot_path}")
            return results_dict, fig
        else:
            return results_dict

__all__ = ['TimeSeriesAccessor']
