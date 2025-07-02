import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from typing import cast
from dask.diagnostics import ProgressBar
from ..utils import get_or_create_dask_client, get_coord_name, filter_by_season
try:
    import cmocean
except ImportError:
    cmocean = None

@xr.register_dataset_accessor("climate_plots")
class PlotsAccessor:
    """
    A custom xarray accessor for creating climate-specific visualizations.

    This accessor extends xarray Dataset and DataArray objects with a `.climate_plots`
    namespace, providing a suite of plotting methods for common climate diagnostics.
    These methods simplify the process of selecting data, calculating indices,
    and generating publication-quality spatial plots.

    The accessor handles common data-wrangling tasks such as:
    - Finding coordinate names (e.g., 'lat', 'latitude').
    - Subsetting data by space, time, and vertical level.
    - Applying seasonal filters.
    - Calculating standard climate indices (e.g., Rx5day, CWD).
    - Generating descriptive titles and labels.

    Examples
    --------
    >>> import xarray as xr
    >>> # Assuming 'climate_diagnostics' is imported to register the accessor
    >>> import climate_diagnostics
    >>>
    >>> # Load a dataset
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>>
    >>> # Generate a plot of the mean air temperature for a specific time range
    >>> ds.climate_plots.plot_mean(
    ...     variable='air',
    ...     time_range=slice('2013-05', '2013-09'),
    ...     season='jja'
    ... )
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
    def _select_data(self, variable, latitude=None, longitude=None, level=None, time_range=None):
        """
        Select and subset data variable based on spatial, temporal, and vertical coordinates.

        This is a core utility method that handles the complex logic of coordinate
        selection across different climate datasets. It accommodates various coordinate
        naming conventions and data structures commonly found in climate model output.
        
        Key features:
        - Automatic coordinate name detection (lat/latitude, lon/longitude, etc.)
        - Flexible selection methods (single values, slices, lists)
        - Intelligent level handling with nearest-neighbor selection
        - Comprehensive error handling and validation
        - Support for both datetime and numeric time coordinates

        Parameters
        ----------
        variable : str
            The name of the data variable to select from the Dataset.
            Must exist in dataset.data_vars.
        latitude : float, slice, or list, optional
            Latitude selection. Can be:
            - Single value: nearest-neighbor selection
            - Slice: range selection (e.g., slice(30, 60))
            - List: specific values selection
        longitude : float, slice, or list, optional
            Longitude selection. Same formats as latitude.
        level : float or slice, optional
            Vertical level selection. If not specified and multiple levels exist,
            defaults to first level. Single values use nearest-neighbor matching.
        time_range : slice, optional
            Time range selection as slice of datetime-like objects or strings.
            E.g., slice('2000-01-01', '2010-12-31')

        Returns
        -------
        selected_data : xr.DataArray
            The selected and subsetted data variable with applied selections.
        level_dim_name_found : str or None
            Name of the level dimension found in the data (for reference).
        level_op : str or None
            Description of level operation performed:
            - 'single_selected': Single level chosen by user
            - 'range_selected': Level range selected
            - 'single_selected_default': First level chosen automatically

        Returns
        -------
        selected_data : xr.DataArray
            The selected and subsetted data variable.
        level_dim_name_found : str or None
            The name of the level dimension found in the data.
        level_op : str or None
            A string indicating the operation performed on the level dimension
            ('single_selected', 'range_selected', 'single_selected_default').

        Raises
        ------
        ValueError
            If the variable is not found or if coordinate selections are invalid.
            
        Notes
        -----
        This method implements robust coordinate validation that handles:
        - Different coordinate naming conventions (CF-compliant and others)
        - Datetime vs numeric coordinate systems
        - Boundary checking with informative error messages
        - Graceful handling of missing coordinates
        """
        # --- Step 1: Variable validation and initialization ---
        # Ensure the requested variable exists in the dataset
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found. Available: {list(self._obj.data_vars.keys())}")

        data_var = self._obj[variable]
        selection_dict = {}  # Store coordinate selections for xarray.sel()
        method_dict = {}     # Store method specifications (e.g., 'nearest' for exact matching)

        # --- Step 2: Coordinate name mapping and discovery ---
        # Build flexible mapping from standard names to actual coordinate names in the dataset
        # This handles different naming conventions across climate datasets (CF-compliant and others)
        coord_map = {
            'latitude': get_coord_name(data_var, ['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat']),
            'longitude': get_coord_name(data_var, ['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon']),
            'time': get_coord_name(data_var, ['time', 't']),
            'level': next((name for name in ['level', 'lev', 'plev', 'height', 'altitude', 'depth', 'z'] if name in data_var.dims or name in data_var.coords), None)
        }
        level_dim_name_found = coord_map['level']
        level_op = None  # Track level operations for metadata

        # Map user parameters to coordinate names and datetime handling flags
        coord_params_map = {
            'latitude': (latitude, False), 'longitude': (longitude, False),
            'time': (time_range, True), 'level': (level, False)
        }

        # --- Step 3: Process each coordinate selection with comprehensive validation ---
        for coord_type_name, (coord_val_param, is_param_datetime_intent) in coord_params_map.items():
            actual_coord_name_in_data = coord_map[coord_type_name]
            if coord_val_param is None:
                continue

            if actual_coord_name_in_data is None:
                if coord_type_name == "level": print(f"Warning: Level parameter provided, but no recognized level coordinate found. Ignoring.")
                else: raise ValueError(f"No {coord_type_name} coordinate found, but '{coord_type_name}' parameter was provided.")
                continue
            if actual_coord_name_in_data not in data_var.coords:
                print(f"Warning: Coord '{actual_coord_name_in_data}' not in variable '{variable}'. Skipping selection.")
                continue
            
            min_data_val_raw_item = data_var[actual_coord_name_in_data].min().item()
            max_data_val_raw_item = data_var[actual_coord_name_in_data].max().item()

            req_min_val, req_max_val = None, None
            is_scalar_request = not isinstance(coord_val_param, (slice, list, np.ndarray))
            if isinstance(coord_val_param, slice):
                req_min_val, req_max_val = coord_val_param.start, coord_val_param.stop
            elif isinstance(coord_val_param, (list, np.ndarray)):
                if not coord_val_param: raise ValueError(f"{coord_type_name.capitalize()} selection list/array empty.")
                req_min_val, req_max_val = min(coord_val_param), max(coord_val_param)
            else:
                req_min_val = req_max_val = coord_val_param

            comp_req_min, comp_req_max = req_min_val, req_max_val
            comp_data_min, comp_data_max = min_data_val_raw_item, max_data_val_raw_item

            data_coord_dtype = data_var[actual_coord_name_in_data].dtype
            data_coord_is_np_datetime = np.issubdtype(data_coord_dtype, np.datetime64)
            data_coord_is_cftime = False
            if not data_coord_is_np_datetime and data_var[actual_coord_name_in_data].size > 0:
                first_val = data_var[actual_coord_name_in_data].isel({data_var[actual_coord_name_in_data].dims[0]: 0}).item()
                if hasattr(first_val, 'year') and hasattr(first_val, 'month') and not isinstance(first_val, (np.datetime64, np.timedelta64)):
                     data_coord_is_cftime = True

            data_coord_is_datetime_like = data_coord_is_np_datetime or data_coord_is_cftime
            data_coord_is_numeric = np.issubdtype(data_coord_dtype, np.number)

            if is_param_datetime_intent:
                try:
                    if comp_req_min is not None: comp_req_min = np.datetime64(comp_req_min)
                    if comp_req_max is not None: comp_req_max = np.datetime64(comp_req_max)
                except Exception as e:
                    raise ValueError(f"Could not convert requested datetime value for {coord_type_name} ('{coord_val_param}') to np.datetime64: {e}")

                if data_coord_is_np_datetime:
                    if isinstance(min_data_val_raw_item, (int, np.integer)):
                        unit = np.datetime_data(data_coord_dtype)[0]
                        comp_data_min = np.datetime64(min_data_val_raw_item, unit)
                    elif min_data_val_raw_item is not None:
                        comp_data_min = np.datetime64(min_data_val_raw_item)
                    
                    if isinstance(max_data_val_raw_item, (int, np.integer)):
                        unit = np.datetime_data(data_coord_dtype)[0]
                        comp_data_max = np.datetime64(max_data_val_raw_item, unit)
                    elif max_data_val_raw_item is not None:
                        comp_data_max = np.datetime64(max_data_val_raw_item)

                elif data_coord_is_cftime:
                    try:
                        if comp_data_min is not None: comp_data_min = np.datetime64(comp_data_min)
                        if comp_data_max is not None: comp_data_max = np.datetime64(comp_data_max)
                    except Exception as c_e:
                        print(f"Warning: Could not convert cftime data bounds for {actual_coord_name_in_data} to np.datetime64 for comparison ({c_e}). Trusting xarray's .sel().")
                        comp_data_min, comp_data_max = None, None


                elif data_coord_is_numeric:
                    print(f"Note: Time parameter is datetime-like, but data coord '{actual_coord_name_in_data}' is numeric. "
                          "Extracting year from request for comparison.")
                    try:
                        if comp_req_min is not None: comp_req_min = comp_req_min.astype('datetime64[Y]').astype(int) + 1970
                        if comp_req_max is not None: comp_req_max = comp_req_max.astype('datetime64[Y]').astype(int) + 1970
                    except Exception as e_year:
                        print(f"Warning: Could not extract year from datetime request for {actual_coord_name_in_data}: {e_year}")
            
            if comp_req_min is not None and comp_data_max is not None:
                try:
                    if comp_req_min > comp_data_max:
                        raise ValueError(f"Requested {coord_type_name} minimum {req_min_val} (as {comp_req_min} type {type(comp_req_min).__name__}) "
                                         f"> data maximum {max_data_val_raw_item} (as {comp_data_max} type {type(comp_data_max).__name__})")
                except TypeError as e:
                    raise TypeError(f"Type mismatch comparing request min ({type(comp_req_min).__name__}) and data max ({type(comp_data_max).__name__}) for {coord_type_name}. Error: {e}")

            if comp_req_max is not None and comp_data_min is not None:
                try:
                    if comp_req_max < comp_data_min:
                         raise ValueError(f"Requested {coord_type_name} maximum {req_max_val} (as {comp_req_max} type {type(comp_req_max).__name__}) "
                                         f"< data minimum {min_data_val_raw_item} (as {comp_data_min} type {type(comp_data_min).__name__})")
                except TypeError as e:
                    raise TypeError(f"Type mismatch comparing request max ({type(comp_req_max).__name__}) and data min ({type(comp_data_min).__name__}) for {coord_type_name}. Error: {e}")

            selection_dict[actual_coord_name_in_data] = coord_val_param
            if coord_type_name == "level":
                level_op = 'range_selected'
                if is_scalar_request or isinstance(coord_val_param, (int, float, np.number)):
                    method_dict[actual_coord_name_in_data] = 'nearest'
                    level_op = 'single_selected'

        if level is None and level_dim_name_found and level_dim_name_found in data_var.dims and data_var.sizes.get(level_dim_name_found, 0) > 1:
            first_level_val = data_var[level_dim_name_found].isel({level_dim_name_found: 0}).item()
            selection_dict[level_dim_name_found] = first_level_val
            level_op = 'single_selected_default'
            print(f"Warning: Multiple levels found. Using first level: {first_level_val}")

        selected_data = data_var
        if method_dict:
            for coord_name, method_val in method_dict.items():
                if coord_name in selection_dict:
                    selected_data = selected_data.sel({coord_name: selection_dict[coord_name]}, method=method_val)
                    del selection_dict[coord_name]
        
        if selection_dict:
            selected_data = selected_data.sel(selection_dict)
        
        if level_op == 'single_selected_default': level_op = 'single_selected'
        return selected_data, level_dim_name_found, level_op

    # --------------------------------------------------------------------------
    # INTERNAL HELPER METHODS: PLOT LAYOUT & FINALIZATION
    # --------------------------------------------------------------------------
    def _setup_geographical_ax(self, figsize, land_only):
        """
        Set up a matplotlib Axes object for geographical plots.

        Configures a plot with coastlines, borders, and gridlines using Cartopy.

        Parameters
        ----------
        figsize : tuple
            The figure size for the plot.
        land_only : bool
            If True, oceans will be masked out.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object.
        ax : cartopy.mpl.geoaxes.GeoAxes
            The matplotlib Axes object with a geographical projection.
        """
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.Robinson())
        feature_zorder, ocean_zorder, gridline_zorder = 3, 2, 4
        ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, zorder=feature_zorder)
        ax.coastlines(zorder=feature_zorder)
        if land_only:
            ax.add_feature(cfeature.OCEAN, zorder=ocean_zorder, facecolor='white', edgecolor='none')
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=gridline_zorder)
        gl.top_labels = False
        gl.right_labels = False
        return fig, ax

    def _generate_title(self, base_operation_name, var_display_name, season,
                        level_op, level_dim_name_in_orig_var,
                        processed_data_coords,
                        original_var_accessor,
                        time_info_provider_data, time_coord_name_actual, time_range_requested,
                        time_format_type='day', index_specific_title_parts=""):
        """
        Generate a descriptive title for a plot.

        The title includes information about the operation, variable, season,
        level, and time range.

        Parameters
        ----------
        base_operation_name : str
            The name of the main operation being plotted (e.g., "Average").
        var_display_name : str
            The display name of the variable.
        season : str
            The season used for the calculation.
        level_op : str
            The operation performed on the level dimension.
        level_dim_name_in_orig_var : str
            The name of the level dimension.
        processed_data_coords : dict
            Coordinates of the data being plotted.
        original_var_accessor : xr.DataArray
            The original DataArray for accessing metadata like units.
        time_info_provider_data : xr.DataArray
            DataArray used to extract the time range for the title.
        time_coord_name_actual : str
            The name of the time coordinate.
        time_range_requested : slice
            The originally requested time range.
        time_format_type : str, optional
            Format for time display ('day' or 'year'). Defaults to 'day'.
        index_specific_title_parts : str, optional
            Additional parts to add to the title, specific to a climate index.

        Returns
        -------
        str
            The generated plot title.
        """
        # --- Part 1: Main Title Line (Season, Operation, Variable) ---
        season_map = {
            'annual': "Annual", 'djf': "Winter (DJF)", 'mam': "Spring (MAM)",
            'jja': "Summer (JJA)", 'jjas': "Summer Monsoon (JJAS)", 'son': "Autumn (SON)"}
        season_str = season_map.get(season.lower(), season.upper())
        title = f"{season_str} {base_operation_name} of {var_display_name}{index_specific_title_parts}"

        # --- Part 2: Level Information Sub-line ---
        level_info_parts = []
        if level_op == 'single_selected' and level_dim_name_in_orig_var and level_dim_name_in_orig_var in processed_data_coords:
            try:
                level_val = processed_data_coords[level_dim_name_in_orig_var].item()
                level_units = ""
                if level_dim_name_in_orig_var in original_var_accessor.coords:
                     level_units = original_var_accessor.coords[level_dim_name_in_orig_var].attrs.get('units', '')
                level_info_parts.append(f"Level: {level_val} {level_units}".strip())
            except Exception:
                level_info_parts.append(f"Level: {processed_data_coords.get(level_dim_name_in_orig_var, 'N/A')}")
        elif level_op == 'range_selected':
            level_info_parts.append("(Level Mean)")

        # --- Part 3: Time Information Sub-line ---
        time_info_parts = []
        if time_coord_name_actual and time_coord_name_actual in time_info_provider_data.coords and \
           time_info_provider_data[time_coord_name_actual].size > 0:
            coord = time_info_provider_data[time_coord_name_actual]
            if np.issubdtype(coord.dtype, np.number):
                min_tv = coord.min().item()
                max_tv = coord.max().item()
                if min_tv == max_tv:
                    time_info_parts.append(f"Time: {min_tv}")
                else:
                    time_info_parts.append(f"{min_tv} to {max_tv}")
            else:
                try:
                    times_np = coord.values.astype('datetime64[ns]')
                    fmt_unit = 'datetime64[Y]' if time_format_type == 'year' else 'datetime64[D]'
                    min_time = np.min(times_np).astype(fmt_unit)
                    max_time = np.max(times_np).astype(fmt_unit)
                    start_str = str(min_time)
                    end_str = str(max_time)
                    if start_str == end_str:
                        time_info_parts.append(f"Time: {start_str}")
                    else:
                        time_info_parts.append(f"{start_str} to {end_str}")
                except Exception as e:
                    print(f"Note: Could not format datetime time range for title: {e}")
        elif isinstance(time_range_requested, slice) and \
             time_range_requested.start is not None and time_range_requested.stop is not None:
            if isinstance(time_range_requested.start, (int, float)) and isinstance(time_range_requested.stop, (int, float)):
                time_info_parts.append(f"Requested: {time_range_requested.start} to {time_range_requested.stop}")
            else:
                try:
                    unit = 'Y' if time_format_type == 'year' else 'D'
                    start_str = np.datetime64(time_range_requested.start, unit).astype(str)
                    stop_str = np.datetime64(time_range_requested.stop, unit).astype(str)
                    time_info_parts.append(f"Requested: {start_str} to {stop_str}")
                except Exception:
                    time_info_parts.append(f"Requested: {time_range_requested.start} to {time_range_requested.stop}")

        # --- Part 4: Assemble Final Title ---
        if level_info_parts: title += f"\n{' '.join(level_info_parts)}"
        if time_info_parts: title += f"\n({' '.join(time_info_parts)})"
        return title

    def _finalize_plot(self, ax, plot_object, title_str, cbar_label,
                       data_for_extent, lon_name_plot, lat_name_plot,
                       save_plot_path, variable):
        """
        Finalize and optionally save the plot.

        This includes adding a colorbar, setting the title, adjusting the map extent,
        and saving the figure to a file if a path is provided.

        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxes
            The Axes object for the plot.
        plot_object : matplotlib.contour.QuadContourSet or None
            The plot object returned by a plotting function (e.g., contourf).
        title_str : str
            The title for the plot.
        cbar_label : str
            The label for the colorbar.
        data_for_extent : xr.DataArray
            DataArray used to determine the plot's geographical extent.
        lon_name_plot : str
            Name of the longitude coordinate.
        lat_name_plot : str
            Name of the latitude coordinate.
        save_plot_path : str or None
            Path to save the plot.
        variable : str
            The name of the variable being plotted (for warning messages).

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The finalized Axes object.
        """
        if plot_object:
            plt.colorbar(plot_object, label=cbar_label, orientation='vertical', pad=0.05, shrink=0.8, ax=ax)
        ax.set_title(title_str, fontsize=12, loc='center')
        if data_for_extent[lon_name_plot].size > 0 and data_for_extent[lat_name_plot].size > 0:
            try:
                min_lon = data_for_extent[lon_name_plot].min().item()
                max_lon = data_for_extent[lon_name_plot].max().item()
                min_lat = data_for_extent[lat_name_plot].min().item()
                max_lat = data_for_extent[lat_name_plot].max().item()
                if min_lon != max_lon and min_lat != max_lat :
                     ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            except Exception as e:
                print(f"Warning: Could not set extent for '{variable}': {e}")
        if save_plot_path:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300); print(f"Plot saved to: {save_plot_path}")
        return ax

    def _plot_spatial_data(self,
                           processed_data_to_plot,
                           original_variable_name,
                           original_selected_data_attrs,
                           original_var_accessor_for_coords,
                           data_season_for_time_info,
                           time_coord_name_actual, time_range_requested,
                           level_op, level_dim_name_in_orig_var,
                           season, contour,
                           figsize, cmap, land_only, levels, save_plot_path,
                           plot_operation_name, cbar_prefix="",
                           time_format_type='day', index_specific_title_parts=""):
        """
        A generic helper function for creating spatial plots.

        This function handles the common steps for plotting 2D spatial data, including
        setting up the map, generating a title, plotting the data (as contours or
        filled contours), and finalizing the plot.

        Parameters
        ----------
        processed_data_to_plot : xr.DataArray
            The 2D data to plot.
        original_variable_name : str
            The name of the original variable.
        original_selected_data_attrs : dict
            Attributes of the original selected data variable.
        original_var_accessor_for_coords : xr.DataArray
            The original DataArray, used for coordinate and metadata access.
        data_season_for_time_info : xr.DataArray
            The seasonally filtered data, used to get time info for the title.
        time_coord_name_actual : str
            The actual name of the time coordinate.
        time_range_requested : slice
            The time range originally requested by the user.
        level_op : str
            The operation performed on the level dimension.
        level_dim_name_in_orig_var : str
            The name of the level dimension in the original variable.
        season : str
            The season string.
        contour : bool
            Use contour lines if True, otherwise use filled contours.
        figsize : tuple
            Figure size.
        cmap : str
            Colormap name.
        land_only : bool
            Mask oceans if True.
        levels : int
            Number of contour levels.
        save_plot_path : str or None
            Path to save the plot.
        plot_operation_name : str
            Name of the operation performed on the data (e.g., "Average").
        cbar_prefix : str, optional
            A prefix for the colorbar label. Defaults to "".
        time_format_type : str, optional
            Time format for the title. Defaults to 'day'.
        index_specific_title_parts : str, optional
            Additional title parts for specific indices. Defaults to "".

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.
        """
        # --- Step 1: Set up the cartopy map axes and identify coordinate names ---
        fig, ax = self._setup_geographical_ax(figsize, land_only)
        data_zorder = 1
        lat_name_plot = get_coord_name(processed_data_to_plot, ['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name_plot = get_coord_name(processed_data_to_plot, ['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])

        if not lat_name_plot or not lon_name_plot:
            raise ValueError(f"Lat/Lon coordinates not found in processed data for '{original_variable_name}'.")
        
        # --- Step 2: Generate the plot title ---
        check_data = processed_data_to_plot.data
        if hasattr(check_data, 'compute'): check_data = check_data.compute()

        plot_object = None
        var_display_name_for_title = original_selected_data_attrs.get('long_name', original_variable_name.replace('_', ' ').capitalize())
        title_str_base = self._generate_title(
            plot_operation_name, var_display_name_for_title, season,
            level_op, level_dim_name_in_orig_var, processed_data_to_plot.coords,
            original_var_accessor_for_coords,
            data_season_for_time_info, time_coord_name_actual, time_range_requested,
            time_format_type, index_specific_title_parts
        )

        # --- Step 3: Plot the data, with a check for all-NaN data ---
        if np.all(np.isnan(check_data)):
            print(f"Warning: Data for '{original_variable_name}' ({plot_operation_name}) is all NaN. Cannot plot.")
            ax.set_title(title_str_base + "\n(Data is all NaN)")
        else:
            plot_kwargs = {
                'ax': ax, 'x': lon_name_plot, 'y': lat_name_plot,
                'transform': ccrs.PlateCarree(), 'levels': levels, 'cmap': cmap,
                'add_colorbar': False, 'zorder': data_zorder
            }
            if contour: plot_object = processed_data_to_plot.plot.contour(**plot_kwargs)
            else: plot_object = processed_data_to_plot.plot.contourf(**plot_kwargs, extend='both')

        # --- Step 4: Finalize the plot with colorbar, title, extent, and save if requested ---
        units = original_selected_data_attrs.get('units', '')
        cbar_label = f"{cbar_prefix}{var_display_name_for_title} ({units})" if units else f"{cbar_prefix}{var_display_name_for_title}"
        
        return self._finalize_plot(ax, plot_object, title_str_base, cbar_label,
                                   processed_data_to_plot, lon_name_plot, lat_name_plot,
                                   save_plot_path, original_variable_name)

    # --------------------------------------------------------------------------
    # INTERNAL HELPER METHODS: CLIMATE INDEX CALCULATIONS
    # --------------------------------------------------------------------------
    def _vectorized_consecutive_true_count(self, da, dim='time'):
        """
        Calculate the length of consecutive `True` runs in a boolean DataArray.
        This is a vectorized operation that is much faster than looping.
        """
        from xarray.core.computation import where
        # Get cumulative sum of `da` along `dim`. This increments for each `True`
        # in a consecutive block. We reset the count when `da` is `False`.
        cumulative_sum = da.cumsum(dim=dim)
        # Where `da` is `False`, the consecutive count is 0.
        # We find the `cumulative_sum` just before each `False` block.
        # This value needs to be subtracted from the `cumulative_sum` in the next `True` block.
        reset_points = where(da, 0, cumulative_sum).ffill(dim=dim)
        # Subtract the `reset_points` to get the length of each consecutive `True` run.
        consecutive_counts = cumulative_sum - reset_points
        return consecutive_counts

    def _apply_yearly_op_then_mean(self, data_for_yearly_op, time_coord_name, operation, op_kwargs=None, dask_op_name=""):
        """
        Apply a yearly operation (e.g., sum, max) and then compute the mean over the years.

        This is a helper function used for climate indices like Rx1day. It first groups
        the data by year, applies an operation within each year, and then calculates the
        mean of these yearly results.

        Parameters
        ----------
        data_for_yearly_op : xr.DataArray
            The input data array with a time dimension.
        time_coord_name : str
            The name of the time coordinate.
        operation : str
            The name of the operation to apply yearly (e.g., 'sum', 'max', 'mean').
        op_kwargs : dict, optional
            Additional keyword arguments for the operation.
        dask_op_name : str, optional
            A display name for the operation when printing progress for Dask computations.

        Returns
        -------
        xr.DataArray
            A DataArray containing the mean of the yearly operation results.
        """
        if op_kwargs is None: op_kwargs = {}
        year_coord_da = data_for_yearly_op[time_coord_name].dt.year
        grouped_data = data_for_yearly_op.groupby(year_coord_da.rename("year_for_grouping"))

        if data_for_yearly_op.chunks:
            print(f"Computing yearly {dask_op_name or operation} for Dask...")
            with ProgressBar(): yearly_data = getattr(grouped_data, operation)(dim=time_coord_name, skipna=True, **op_kwargs).compute()
            print(f"Computing mean of yearly {dask_op_name or operation} for Dask...")
            with ProgressBar(): mean_yearly_data = yearly_data.mean(dim='year_for_grouping', skipna=True).compute()
        else:
            yearly_data = getattr(grouped_data, operation)(dim=time_coord_name, skipna=True, **op_kwargs)
            mean_yearly_data = yearly_data.mean(dim='year_for_grouping', skipna=True)
        return mean_yearly_data

    def _calc_spell_counts(self, data_in, time_coord_name, threshold_val, min_consecutive_days, spell_type_is_above_thresh):
        """
        Calculate the average number of spells per year. Vectorized implementation.
        A "spell" is a period of consecutive days meeting a condition for at least a minimum number of days.
        """
        condition = (data_in > threshold_val) if spell_type_is_above_thresh else (data_in < threshold_val)
        
        # Calculate the length of each consecutive run
        consecutive_lengths = self._vectorized_consecutive_true_count(condition, dim=time_coord_name)
        
        # A spell of required duration is "born" when its length first equals the minimum duration.
        # This counts each spell exactly once.
        spell_is_born = (consecutive_lengths == min_consecutive_days)
        
        # The number of spells per year is the sum of these "births"
        return self._apply_yearly_op_then_mean(spell_is_born.astype(int), time_coord_name, 'sum', dask_op_name="spell counts")

    def _calc_days_above_or_below_threshold_mean(self, data_in, time_coord_name, threshold_val, is_above_op):
        """
        Calculate the mean annual number of days above or below a threshold.

        Helper function that counts days per year meeting a condition and then
        averages these counts over all years.

        Parameters
        ----------
        data_in : xr.DataArray
            Input data with a time dimension.
        time_coord_name : str
            Name of the time coordinate.
        threshold_val : float or xr.DataArray
            The threshold value.
        is_above_op : bool
            If True, counts days *above* the threshold. If False, counts days *below*.

        Returns
        -------
        xr.DataArray
            A DataArray with the mean annual number of days meeting the condition.
        """
        condition_met = (data_in >= threshold_val) if is_above_op else (data_in < threshold_val)
        return self._apply_yearly_op_then_mean(condition_met.astype(int), time_coord_name, 'sum', dask_op_name="days matching condition")

    def _calc_max_consecutive_days(self, data_in, time_coord_name, threshold_val, spell_type_is_above_thresh):
        """
        Calculate the mean of the annual maximum number of consecutive days
        above or below a threshold. Vectorized implementation.
        """
        condition = (data_in >= threshold_val) if spell_type_is_above_thresh else (data_in < threshold_val)
        
        # Get the length of each consecutive run of True values
        consecutive_lengths = self._vectorized_consecutive_true_count(condition, dim=time_coord_name)
        
        # Group by year and find the maximum length within each year, then average the maxima
        return self._apply_yearly_op_then_mean(consecutive_lengths, time_coord_name, 'max', dask_op_name="max consecutive days")

    def _calc_days_in_spell(self, data_in, time_coord_name, threshold_val, min_consecutive_days, spell_type_is_above_thresh):
        """
        Calculate the mean annual number of days in spells (e.g., WSDI).
        A "spell" is a period of consecutive days meeting a condition for at least a minimum number of days.
        This function counts the total number of days within such spells.
        """
        condition = (data_in >= threshold_val) if spell_type_is_above_thresh else (data_in < threshold_val)

        # Calculate the length of each consecutive run up to the current point
        consecutive_lengths = self._vectorized_consecutive_true_count(condition, dim=time_coord_name)

        # Identify the end of each consecutive run of True values.
        # A run ends if the current value is True and the next is False.
        is_spell_end = (condition & ~condition.shift({time_coord_name: -1}, fill_value=False))

        # Get the total length of each spell at the point where the spell ends.
        # Where it's not a spell end, this will be NaN.
        spell_end_lengths = consecutive_lengths.where(is_spell_end)

        # Back-fill the total spell length over the duration of each spell.
        # This propagates the final length of a spell to all days within that spell.
        total_spell_lengths = spell_end_lengths.bfill(dim=time_coord_name)

        # Mask out days that were not part of any spell to begin with.
        total_spell_lengths = total_spell_lengths.where(condition, 0)

        # Identify which days are part of a spell that meets the minimum duration requirement.
        is_in_qualifying_spell = (total_spell_lengths >= min_consecutive_days)

        # Sum the number of qualifying days per year and then average over the years.
        return self._apply_yearly_op_then_mean(is_in_qualifying_spell.astype(int), time_coord_name, 'sum', dask_op_name="days in spell")

    # ==============================================================================
    # PUBLIC PLOTTING METHODS
    # ==============================================================================

    # --------------------------------------------------------------------------
    # A. Basic Statistical Plots
    # --------------------------------------------------------------------------
    def plot_mean(self, variable='air', latitude=None, longitude=None, level=None,
                  time_range=None, season='annual', contour=False,
                  figsize=(16, 10), cmap='coolwarm', land_only=False,
                  levels=30, save_plot_path=None):
        """
        Plot the temporal mean of a variable over a specified period.

        Calculates and plots the mean of a given variable over the specified
        time, space, and level dimensions. This is a fundamental plot for
        understanding the basic climate state.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to plot. Defaults to 'air'.
        latitude : float, slice, or list, optional
            Latitude range for selection. Can be a single value, a list of values,
            or a slice object (e.g., slice(30, 60)).
        longitude : float, slice, or list, optional
            Longitude range for selection. Can be a single value, a list, or a slice
            (e.g., slice(-120, -80)).
        level : float or slice, optional
            Vertical level for data selection. A single value selects the nearest
            level. A slice (e.g., slice(500, 200)) will result in the data being
            averaged over that level range before the temporal mean is computed.
        time_range : slice, optional
            Time range for selection, specified as a slice of datetime-like objects
            or strings (e.g., slice('2000-01-01', '2010-12-31')).
        season : str, optional
            Season to calculate the mean for. Supported options are 'annual',
            'jjas', 'djf', 'mam', 'son', 'jja'. Defaults to 'annual'.
        contour : bool, optional
            If True, use contour lines instead of filled contours. Defaults to False.
        figsize : tuple, optional
            Figure size in inches (width, height). Defaults to (16, 10).
        cmap : str, optional
            Colormap name for the plot. Defaults to 'coolwarm'.
        land_only : bool, optional
            If True, mask out ocean areas, plotting data only over land.
            Defaults to False.
        levels : int, optional
            Number of contour levels for the plot. Defaults to 30.
        save_plot_path : str or None, optional
            If provided, the path to save the plot figure to.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot, allowing for further customization.

        See Also
        --------
        plot_std_time : Plot the temporal standard deviation.
        plot_percentile_spatial : Plot a specific temporal percentile.

        Examples
        --------
        >>> import xarray as xr
        >>> import climate_diagnostics
        >>> ds = xr.tutorial.load_dataset("air_temperature")
        >>> ds.climate_plots.plot_mean(
        ...     variable='air',
        ...     level=850,
        ...     time_range=slice('2013-01', '2013-12'),
        ...     season='djf'
        ... )
        """
        get_or_create_dask_client()
        # Step 1: Select the data based on user parameters
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        
        # Step 2: If a level range was selected, average over it first
        current_data_for_ops = selected_data
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
             current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)
             print(f"Averaging over selected levels for '{variable}'.")

        # Step 3: Apply seasonal filter
        data_season = filter_by_season(current_data_for_ops, season)
        if data_season.size == 0:
            raise ValueError(f"No data after selections and season filter ('{season}') for '{variable}'.")

        # Step 4: Calculate the temporal mean
        time_coord_name_actual = get_coord_name(data_season, ['time', 't'])
        
        mean_data = data_season
        if time_coord_name_actual and time_coord_name_actual in data_season.dims:
            if data_season.chunks:
                print(f"Computing time mean for '{variable}' using Dask...")
                with ProgressBar(): mean_data = data_season.mean(dim=time_coord_name_actual, skipna=True).compute()
            else:
                 mean_data = data_season.mean(dim=time_coord_name_actual, skipna=True)
        elif time_coord_name_actual:
            print(f"Warning: Time coord '{time_coord_name_actual}' not a dimension for averaging. Plotting as is.")
        else:
            print(f"Warning: No time coord for averaging. Plotting as is.")

        # Step 5: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            mean_data, variable, selected_data.attrs, self._obj[variable],
            data_season, time_coord_name_actual, time_range,
            level_op, level_dim_name, season, contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Average"
        )

    def plot_std_time(self, variable='air', latitude=None, longitude=None, level=None,
                      time_range=None, season='annual', contour=False,
                      figsize=(16,10), cmap='viridis', land_only = False,
                      levels=30, save_plot_path = None):
        """
        Plot the temporal standard deviation of a variable.

        Calculates and plots the standard deviation of a given variable over time,
        which is a key measure of climate variability.

        Parameters
        ----------
        variable : str, optional
            Name of the variable to plot. Defaults to 'air'.
        latitude : float, slice, or list, optional
            Latitude range for selection. Can be a single value, a list of values,
            or a slice object (e.g., slice(30, 60)).
        longitude : float, slice, or list, optional
            Longitude range for selection. Can be a single value, a list, or a slice
            (e.g., slice(-120, -80)).
        level : float or slice, optional
            Vertical level for data selection. A single value selects the nearest
            level. A slice (e.g., slice(500, 200)) will result in the data being
            averaged over that level range before the standard deviation is computed.
        time_range : slice, optional
            Time range for selection, specified as a slice of datetime-like objects
            or strings (e.g., slice('2000-01-01', '2010-12-31')).
        season : str, optional
            Season to calculate the standard deviation for. Supported options are 'annual',
            'jjas', 'djf', 'mam', 'son', 'jja'. Defaults to 'annual'.
        contour : bool, optional
            If True, use contour lines instead of filled contours. Defaults to False.
        figsize : tuple, optional
            Figure size in inches (width, height). Defaults to (16, 10).
        cmap : str, optional
            Colormap name for the plot. Defaults to 'viridis'.
        land_only : bool, optional
            If True, mask out ocean areas. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            If provided, the path to save the plot figure to.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.

        See Also
        --------
        plot_mean : Plot the temporal mean.
        """
        get_or_create_dask_client()
        # Step 1: Select the data based on user parameters
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        
        # Step 2: If a level range was selected, average over it first
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)
            print(f"Averaging across selected levels for '{variable}' before calculating std dev.")

        # Step 3: Apply seasonal filter
        time_coord_name_actual = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name_actual or time_coord_name_actual not in current_data_for_ops.dims:
             raise ValueError(f"Std dev requires time dimension for '{variable}'.")

        data_season = filter_by_season(current_data_for_ops, season)
        if data_season.size == 0:
            raise ValueError(f"No data after selections and season filter ('{season}') for '{variable}'.")
        if data_season.sizes[time_coord_name_actual] < 2:
             raise ValueError(f"Std dev requires at least 2 time points (found {data_season.sizes[time_coord_name_actual]}).")

        # Step 4: Calculate the temporal standard deviation
        if data_season.chunks:
            print(f"Computing std dev over time for '{variable}' using Dask...")
            with ProgressBar(): std_data = data_season.std(dim=time_coord_name_actual, skipna=True).compute()
        else:
            std_data = data_season.std(dim=time_coord_name_actual, skipna=True)

        # Step 5: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            std_data, variable, selected_data.attrs, self._obj[variable],
            data_season, time_coord_name_actual, time_range,
            level_op, level_dim_name, season, contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Temporal Standard Deviation", cbar_prefix="Std. Dev. of "
        )

    def plot_percentile_spatial(self, variable='prate', percentile=95, latitude=None, longitude=None,
                                level=None, time_range=None, contour=False, figsize=(16, 10),
                                cmap='Blues',
                                land_only=False, levels=30, save_plot_path=None):
        """
        Plot the spatial distribution of a temporal percentile for a variable.

        Calculates a given percentile (e.g., 95th) at each grid point over the
        time dimension and plots the resulting map. This is useful for identifying
        areas with extreme values.

        Parameters
        ----------
        variable : str, optional
            Name of the variable. Defaults to 'prate'.
        percentile : int, optional
            The percentile to calculate (0-100). Defaults to 95.
        latitude : float, slice, or list, optional
            Latitude range for selection. Can be a single value, a list of values,
            or a slice object.
        longitude : float, slice, or list, optional
            Longitude range for selection. Can be a single value, a list, or a slice.
        level : float or slice, optional
            Vertical level for data selection. A single value selects the nearest
            level. A slice will result in the data being averaged over that range.
        time_range : slice, optional
            Time range for selection as a slice of datetime-like objects or strings.
        contour : bool, optional
            If True, use contour lines instead of filled contours. Defaults to False.
        figsize : tuple, optional
            Figure size in inches (width, height). Defaults to (16, 10).
        cmap : str, optional
            Colormap name for the plot. Defaults to 'Blues'.
        land_only : bool, optional
            If True, mask out ocean areas. Defaults to False.
        levels : int, optional
            Number of contour levels for the plot. Defaults to 30.
        save_plot_path : str or None, optional
            If provided, the path to save the plot figure to.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.

        See Also
        --------
        plot_mean : Plot the temporal mean of a variable.
        """
        get_or_create_dask_client()
        # Step 1: Validate input
        if not 0 <= percentile <= 100:
            raise ValueError(f"Percentile must be 0-100, got {percentile}")
            
        # Step 2: Select the data and handle level-based averaging
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)
            print(f"Averaging across selected levels for '{variable}' before calculating percentile.")

        # Step 3: Calculate the temporal percentile
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"Percentile calculation requires time dimension for '{variable}'.")
        
        if current_data_for_ops.chunks:
            print(f"Computing {percentile}th percentile for '{variable}' using Dask...")
            with ProgressBar():
                percentile_data = current_data_for_ops.quantile(percentile/100.0, dim=time_coord_name, skipna=True).compute()
        else:
            percentile_data = current_data_for_ops.quantile(percentile/100.0, dim=time_coord_name, skipna=True)
        
        # Step 4: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            percentile_data, variable, selected_data.attrs, self._obj[variable],
            current_data_for_ops, time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name=f"Spatial Distribution of {percentile}th Percentile"
        )

    # --------------------------------------------------------------------------
    # B. Precipitation and Climate Indices (ETCCDI-style)
    # --------------------------------------------------------------------------
    def plot_annual_sum_mean(self, variable='prate', latitude=None, longitude=None, level=None,
                             time_range=None, contour=False, figsize=(16, 10),
                             cmap='Blues',
                             land_only=False, levels=30, save_plot_path=None):
        """
        Plot the mean of annual sums for a variable.

        This is often used for precipitation-related variables (e.g., 'prate')
        to show the average annual total precipitation, a key climate indicator.
        It works by summing the variable over each year and then averaging these
        annual sums over the entire time period.

        Parameters
        ----------
        variable : str, optional
            Name of the variable. Defaults to 'prate'.
        latitude : float, slice, or list, optional
            Latitude range for selection.
        longitude : float, slice, or list, optional
            Longitude range for selection.
        level : float or slice, optional
            Vertical level for selection. If a slice is given, data is averaged
            over the level range.
        time_range : slice, optional
            Time range for selection.
        contour : bool, optional
            Use contour lines if True. Defaults to False.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        cmap : str, optional
            Colormap. Defaults to 'Blues'.
        land_only : bool, optional
            Mask oceans if True. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            Path to save the plot.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.

        See Also
        --------
        plot_rx1day : Plot the mean annual maximum 1-day precipitation.
        """
        get_or_create_dask_client()
        # Step 1: Select the data based on user parameters
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)
            print(f"Averaging mean annual sum across selected levels for '{variable}'.")

        # Step 2: Calculate the mean of annual sums
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"Annual sum mean requires time dimension for '{variable}'.")
        
        mean_annual_sum = self._apply_yearly_op_then_mean(current_data_for_ops, time_coord_name, 'sum', dask_op_name="sums")

        # Step 3: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            mean_annual_sum, variable, selected_data.attrs, self._obj[variable],
            current_data_for_ops, time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Average Annual Sum", time_format_type='year'
        )

    def plot_rx1day(self, variable='prate', latitude=None, longitude=None, level=None,
                    time_range=None, contour=False, figsize=(16, 10),
                    cmap='Blues',
                    land_only=False, levels=30, save_plot_path=None):
        """
        Plot the mean annual maximum 1-day precipitation (Rx1day).

        Rx1day is a climate index from the Expert Team on Climate Change Detection
        and Indices (ETCCDI). It represents the maximum precipitation amount
        in a single day for each year, averaged over the specified time period.
        It is a key indicator of extreme precipitation events.

        Parameters
        ----------
        variable : str, optional
            Name of the precipitation variable. Defaults to 'prate'.
        latitude : float, slice, or list, optional
            Latitude range for selection.
        longitude : float, slice, or list, optional
            Longitude range for selection.
        level : float or slice, optional
            Vertical level for selection. If a slice is given, data is averaged
            over the level range.
        time_range : slice, optional
            Time range for selection.
        contour : bool, optional
            Use contour lines if True. Defaults to False.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        cmap : str, optional
            Colormap. Defaults to 'Blues'.
        land_only : bool, optional
            Mask oceans if True. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            Path to save the plot.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.

        See Also
        --------
        plot_rx5day : Plot the mean annual maximum 5-day precipitation.
        """
        get_or_create_dask_client()
        # Step 1: Select the data based on user parameters
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)
            print(f"Averaging Rx1day across selected levels for '{variable}'.")

        # Step 2: Calculate the mean of annual maxima
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"Rx1day requires time dimension for '{variable}'.")

        mean_rx1day = self._apply_yearly_op_then_mean(current_data_for_ops, time_coord_name, 'max', dask_op_name="maxima")

        # Step 3: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            mean_rx1day, variable, selected_data.attrs, self._obj[variable],
            current_data_for_ops, time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Maximum 1-day Precipitation (Rx1day)", time_format_type='year'
        )

    def plot_rx5day(self, variable='prate', latitude=None, longitude=None, level=None,
                    time_range=None, contour=False, figsize=(16, 10),
                    cmap='Blues',
                    land_only=False, levels=30, save_plot_path=None, title=None):
        """
        Plot the mean annual maximum 5-day precipitation (Rx5day).

        Rx5day is a climate index (ETCCDI) for the maximum precipitation amount
        in any consecutive 5-day period. This function finds the maximum 5-day
        sum for each year and then averages these maxima over the entire time period.
        It is an indicator of short-term precipitation extremes.

        Parameters
        ----------
        variable : str, optional
            Name of the precipitation variable. Defaults to 'prate'.
        latitude : float, slice, or list, optional
            Latitude range.
        longitude : float, slice, or list, optional
            Longitude range.
        level : float or slice, optional
            Vertical level.
        time_range : slice, optional
            Time range.
        contour : bool, optional
            Use contour lines if True. Defaults to False.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        cmap : str, optional
            Colormap. Defaults to 'Blues'.
        land_only : bool, optional
            Mask oceans if True. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            Path to save the plot.
        title : str or None, optional
            Custom title for the plot. If None, a default is generated.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.
        """
        get_or_create_dask_client()
        # Step 1: Select the data based on user parameters
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)
            print(f"Averaging across selected levels for '{variable}' before calculating Rx5day.")

        # Step 2: Calculate the 5-day rolling sum
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"Rx5day requires time dimension for '{variable}'.")

        if current_data_for_ops.chunks:
            print(f"Computing 5-day rolling sum for '{variable}' using Dask...")
            with ProgressBar():
                rolling_sum_data = current_data_for_ops.rolling({time_coord_name: 5}, center=True).sum(skipna=True).compute()
        else:
            rolling_sum_data = current_data_for_ops.rolling({time_coord_name: 5}, center=True).sum(skipna=True)
        
        # Step 3: Find the mean of the annual maxima of the rolling sum
        mean_rx5day = self._apply_yearly_op_then_mean(rolling_sum_data, time_coord_name, 'max', dask_op_name="5-day maxima")
        
        plot_op_name = title if title else "Max 5-day Precipitation (Rx5day)"

        # Step 4: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            mean_rx5day, variable, selected_data.attrs, self._obj[variable],
            current_data_for_ops, time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name=plot_op_name,
            time_format_type='year'
        )

    def plot_wet_spell_duration(self, variable='prate', threshold_percentile=90, min_consecutive_days=6,
                                latitude=None, longitude=None, level=None, time_range=None,
                                contour=False, figsize=(16, 10), cmap='Blues',
                                land_only=False, levels=30, save_plot_path=None):
        """
        Plot the Wet Spell Duration Index (WSDI).

        WSDI is the total number of days per year that are part of a wet spell.
        A wet spell is defined as a period of at least `min_consecutive_days`
        where precipitation is above a certain percentile threshold.

        Parameters
        ----------
        variable : str, optional
            Name of the precipitation variable. Defaults to 'prate'.
        threshold_percentile : int, optional
            The percentile for the precipitation threshold. Defaults to 90.
        min_consecutive_days : int, optional
            Minimum consecutive days for a wet spell. Defaults to 6.
        latitude : float, slice, or list, optional
            Latitude range.
        longitude : float, slice, or list, optional
            Longitude range.
        level : float or slice, optional
            Vertical level.
        time_range : slice, optional
            Time range.
        contour : bool, optional
            Use contour lines if True. Defaults to False.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        cmap : str, optional
            Colormap. Defaults to 'Blues'.
        land_only : bool, optional
            Mask oceans if True. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            Path to save the plot.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.
        """
        get_or_create_dask_client()
        # Step 1: Validate inputs
        if not 0 <= threshold_percentile <= 100: raise ValueError("Percentile 0-100.")
        if min_consecutive_days < 1: raise ValueError("Min days >= 1.")

        # Step 2: Select the data and handle level-based averaging
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)

        # Step 3: Calculate the percentile threshold
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"WSDI requires time dimension for '{variable}'.")

        frac = cast(float, threshold_percentile) / 100.0
        if current_data_for_ops.chunks:
            with ProgressBar(): actual_threshold = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True).compute()
        else:
            actual_threshold = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True)
        
        # Step 4: Calculate the mean annual number of days in wet spells
        mean_wsdi = self._calc_days_in_spell(current_data_for_ops, time_coord_name, actual_threshold, min_consecutive_days, spell_type_is_above_thresh=True)
        
        # Step 5: Pass to the generic spatial plotting function
        index_specific_parts = f"\n(Pctl Thresh: {threshold_percentile}th, Min Duration: {min_consecutive_days} days)"
        return self._plot_spatial_data(
            mean_wsdi, variable, selected_data.attrs, self._obj[variable], current_data_for_ops,
            time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Annual Wet Spell Duration Index (WSDI)",
            index_specific_title_parts=index_specific_parts, time_format_type='year'
        )

    def plot_dry_spell_duration(self, variable='prate', threshold_percentile=10, min_consecutive_days=6,
                                latitude=None, longitude=None, level=None, time_range=None,
                                contour=False, figsize=(16, 10), cmap='YlOrBr',
                                land_only=False, levels=30, save_plot_path=None):
        """
        Plot the Dry Spell Duration Index.

        This index is the total number of days per year that are part of a dry spell.
        A dry spell is defined as a period of at least `min_consecutive_days`
        where precipitation is below a certain percentile threshold.

        Parameters
        ----------
        variable : str, optional
            Name of the precipitation variable. Defaults to 'prate'.
        threshold_percentile : int, optional
            The percentile for the precipitation threshold. Defaults to 10.
        min_consecutive_days : int, optional
            Minimum consecutive days for a dry spell. Defaults to 6.
        latitude : float, slice, or list, optional
            Latitude range.
        longitude : float, slice, or list, optional
            Longitude range.
        level : float or slice, optional
            Vertical level.
        time_range : slice, optional
            Time range.
        contour : bool, optional
            Use contour lines if True. Defaults to False.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        cmap : str, optional
            Colormap. Defaults to 'YlOrBr'.
        land_only : bool, optional
            Mask oceans if True. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            Path to save the plot.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.
        """
        get_or_create_dask_client()
        # Step 1: Validate inputs
        if not 0 <= threshold_percentile <= 100: raise ValueError("Percentile 0-100.")
        if min_consecutive_days < 1: raise ValueError("Min days >= 1.")

        # Step 2: Select the data and handle level-based averaging
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)

        # Step 3: Calculate the percentile threshold
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"DSDI requires time dimension for '{variable}'.")

        frac = cast(float, threshold_percentile) / 100.0
        if current_data_for_ops.chunks:
            with ProgressBar(): actual_threshold = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True).compute()
        else:
            actual_threshold = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True)
        
        # Step 4: Calculate the mean annual number of days in dry spells
        mean_dsdi = self._calc_days_in_spell(current_data_for_ops, time_coord_name, actual_threshold, min_consecutive_days, spell_type_is_above_thresh=False)
        
        # Step 5: Pass to the generic spatial plotting function
        index_specific_parts = f"\n(Pctl Thresh: {threshold_percentile}th, Min Duration: {min_consecutive_days} days)"
        return self._plot_spatial_data(
            mean_dsdi, variable, selected_data.attrs, self._obj[variable], current_data_for_ops,
            time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Annual Dry Spell Duration Index",
            index_specific_title_parts=index_specific_parts, time_format_type='year'
        )

    def plot_precipitation_above_threshold(self, variable='prate', threshold=None, threshold_percentile=None,
                                         latitude=None, longitude=None, level=None, time_range=None,
                                         contour=False, figsize=(16, 10), cmap='Blues',
                                         land_only=False, levels=30, save_plot_path=None):
        """
        Plot the average annual number of days a variable exceeds a threshold.

        This can be used for indices like the number of heavy precipitation days.
        The threshold can be an absolute value or a percentile.

        Parameters
        ----------
        variable : str, optional
            Name of the variable. Defaults to 'prate'.
        threshold : float, optional
            Absolute threshold value. One of `threshold` or `threshold_percentile` must be given.
        threshold_percentile : int, optional
            Percentile to use as the threshold (0-100).
        latitude : float, slice, or list, optional
            Latitude range.
        longitude : float, slice, or list, optional
            Longitude range.
        level : float or slice, optional
            Vertical level.
        time_range : slice, optional
            Time range.
        contour : bool, optional
            Use contour lines if True. Defaults to False.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        cmap : str, optional
            Colormap. Defaults to 'Blues'.
        land_only : bool, optional
            Mask oceans if True. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            Path to save the plot.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.
        """
        get_or_create_dask_client()
        # Step 1: Validate inputs
        if threshold is None and threshold_percentile is None:
            raise ValueError("Either threshold or threshold_percentile must be provided.")
        if threshold_percentile is not None and not 0 <= threshold_percentile <= 100:
            raise ValueError(f"Threshold percentile must be between 0 and 100.")

        # Step 2: Select the data and handle level-based averaging
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)

        # Step 3: Determine the threshold value
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"Calculation requires time dimension for '{variable}'.")

        actual_threshold_val = threshold
        var_units = current_data_for_ops.attrs.get('units', '')
        threshold_info_for_title = f"(Abs Thresh: {threshold} {var_units})"
        if threshold is None:
            frac = cast(float, threshold_percentile) / 100.0
            if current_data_for_ops.chunks:
                with ProgressBar(): actual_threshold_val = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True).compute()
            else:
                actual_threshold_val = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True)
            threshold_info_for_title = f"(Pctl Thresh: {threshold_percentile}th)"
        
        # Step 4: Calculate the mean annual number of days exceeding the threshold
        mean_days = self._calc_days_above_or_below_threshold_mean(current_data_for_ops, time_coord_name, actual_threshold_val, is_above_op=True)
        
        # Step 5: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            mean_days, variable, selected_data.attrs, self._obj[variable], current_data_for_ops,
            time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Average Annual Days Exceeding Threshold",
            index_specific_title_parts=f"\n{threshold_info_for_title}", time_format_type='year'
        )

    def plot_consecutive_wet_days(self, variable='prate', threshold=1.157e-5, threshold_percentile=None,
                                latitude=None, longitude=None, level=None, time_range=None,
                                contour=False, figsize=(16, 10), cmap='Blues',
                                land_only=False, levels=30, save_plot_path=None):
        """
        Plot the average annual maximum number of Consecutive Wet Days (CWD).

        CWD is the longest stretch of consecutive days in a year where a variable
        (e.g., precipitation) is at or above a threshold. The result is averaged
        over all years. The default threshold corresponds to 1 mm/day for
        precipitation rate in units of kg m-2 s-1, a common definition for CWD.

        Parameters
        ----------
        variable : str, optional
            Name of the variable. Defaults to 'prate'.
        threshold : float, optional
            Absolute threshold value. Defaults to 1.157e-5 (~1mm/day).
            Set to None if using `threshold_percentile`.
        threshold_percentile : int, optional
            Percentile to use as the threshold (0-100). Overrides `threshold` if set.
        latitude : float, slice, or list, optional
            Latitude range.
        longitude : float, slice, or list, optional
            Longitude range.
        level : float or slice, optional
            Vertical level.
        time_range : slice, optional
            Time range.
        contour : bool, optional
            Use contour lines if True. Defaults to False.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        cmap : str, optional
            Colormap. Defaults to 'Blues'.
        land_only : bool, optional
            Mask oceans if True. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            Path to save the plot.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.
        """
        get_or_create_dask_client()
        # Step 1: Validate inputs
        if threshold is None and threshold_percentile is None: raise ValueError("One of 'threshold' or 'threshold_percentile' must be provided.")
        if threshold_percentile is not None and not 0 <= threshold_percentile <= 100: raise ValueError("Percentile 0-100.")

        # Step 2: Select the data and handle level-based averaging
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)

        # Step 3: Determine the threshold value
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"CWD requires time dimension for '{variable}'.")

        actual_threshold_val = threshold
        var_units = current_data_for_ops.attrs.get('units', '')
        threshold_info_for_title = f"(Abs Thresh: {threshold} {var_units})"
        if threshold_percentile is not None:
            frac = cast(float, threshold_percentile) / 100.0
            if current_data_for_ops.chunks:
                with ProgressBar(): actual_threshold_val = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True).compute()
            else:
                actual_threshold_val = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True)
            threshold_info_for_title = f"(Pctl Thresh: {threshold_percentile}th)"
        
        # Step 4: Calculate the mean of the annual maximum consecutive wet days
        mean_cwd = self._calc_max_consecutive_days(current_data_for_ops, time_coord_name, actual_threshold_val, spell_type_is_above_thresh=True)
        
        # Step 5: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            mean_cwd, variable, selected_data.attrs, self._obj[variable], current_data_for_ops,
            time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Average Annual Maximum Consecutive Wet Days",
            index_specific_title_parts=f"\n{threshold_info_for_title}", time_format_type='year'
        )

    def plot_consecutive_dry_days(self, variable='prate', threshold=1.157e-5, threshold_percentile=None,
                                latitude=None, longitude=None, level=None, time_range=None,
                                contour=False, figsize=(16, 10), cmap='YlOrBr',
                                land_only=False, levels=30, save_plot_path=None):
        """
        Plot the average annual maximum number of Consecutive Dry Days (CDD).

        CDD is the longest stretch of consecutive days in a year where a variable
        (e.g., precipitation) is below a threshold. The result is averaged over
        all years. The default threshold corresponds to 1 mm/day for
        precipitation rate in units of kg m-2 s-1, a common definition for CDD.

        Parameters
        ----------
        variable : str, optional
            Name of the variable. Defaults to 'prate'.
        threshold : float, optional
            Absolute threshold value. Defaults to 1.157e-5 (~1mm/day).
            Set to None if using `threshold_percentile`.
        threshold_percentile : int, optional
            Percentile to use as the threshold (0-100). Overrides `threshold` if set.
        latitude : float, slice, or list, optional
            Latitude range.
        longitude : float, slice, or list, optional
            Longitude range.
        level : float or slice, optional
            Vertical level.
        time_range : slice, optional
            Time range.
        contour : bool, optional
            Use contour lines if True. Defaults to False.
        figsize : tuple, optional
            Figure size. Defaults to (16, 10).
        cmap : str, optional
            Colormap. Defaults to 'YlOrBr'.
        land_only : bool, optional
            Mask oceans if True. Defaults to False.
        levels : int, optional
            Number of contour levels. Defaults to 30.
        save_plot_path : str or None, optional
            Path to save the plot.

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxes
            The Axes object of the plot.
        """
        get_or_create_dask_client()
        # Step 1: Validate inputs
        if threshold is None and threshold_percentile is None: raise ValueError("One of 'threshold' or 'threshold_percentile' must be provided.")
        if threshold_percentile is not None and not 0 <= threshold_percentile <= 100: raise ValueError("Percentile 0-100.")

        # Step 2: Select the data and handle level-based averaging
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )
        current_data_for_ops = selected_data
        
        if level_op == 'range_selected' and level_dim_name and level_dim_name in current_data_for_ops.dims:
            current_data_for_ops = current_data_for_ops.mean(dim=level_dim_name, skipna=True)

        # Step 3: Determine the threshold value
        time_coord_name = get_coord_name(current_data_for_ops, ['time', 't'])
        if not time_coord_name or time_coord_name not in current_data_for_ops.dims:
            raise ValueError(f"CDD requires time dimension for '{variable}'.")

        actual_threshold_val = threshold
        var_units = current_data_for_ops.attrs.get('units', '')
        threshold_info_for_title = f"(Abs Thresh: {threshold} {var_units})"
        if threshold_percentile is not None:
            frac = cast(float, threshold_percentile) / 100.0
            if current_data_for_ops.chunks:
                with ProgressBar(): actual_threshold_val = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True).compute()
            else:
                actual_threshold_val = current_data_for_ops.quantile(frac, dim=time_coord_name, skipna=True)
            threshold_info_for_title = f"(Pctl Thresh: {threshold_percentile}th)"
        
        # Step 4: Calculate the mean of the annual maximum consecutive dry days
        mean_cdd = self._calc_max_consecutive_days(current_data_for_ops, time_coord_name, actual_threshold_val, spell_type_is_above_thresh=False)
        
        # Step 5: Pass to the generic spatial plotting function
        return self._plot_spatial_data(
            mean_cdd, variable, selected_data.attrs, self._obj[variable], current_data_for_ops,
            time_coord_name, time_range,
            level_op, level_dim_name, "Annual", contour,
            figsize, cmap, land_only, levels, save_plot_path,
            plot_operation_name="Average Annual Maximum Consecutive Dry Days",
            index_specific_title_parts=f"\n{threshold_info_for_title}", time_format_type='year'
        )

__all__ = ['PlotsAccessor']
