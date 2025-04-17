import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar
import scipy.ndimage as ndimage  # For gaussian filter

@xr.register_dataset_accessor("climate_plots")
class PlotsAccessor:
    """
    Geographical visualizations of climate data using contour plots.
    
    Provides methods for visualizing climate data with support for seasonal filtering,
    spatial calculations, smoothing, and various visualization options.
    
    Access via the .climate_plots attribute on xarray Datasets.
    
    Examples
    --------
    >>> ds = xr.open_dataset('climate_data.nc')
    >>> ds.climate_plots.plot_mean(variable='air')
    >>> ds.climate_plots.plot_mean(variable='air', season='djf', 
    ...                           latitude=slice(40, 6), longitude=slice(65,110))
    """

    def __init__(self, xarray_obj):
        """Initialize the climate plots accessor with an xarray Dataset."""
        self._obj = xarray_obj

    def _get_coord_name(self, possible_names):
        """
        Find the actual coordinate name in the dataset from a list of common alternatives.
        
        Parameters
        ----------
        possible_names : list of str
            List of possible coordinate names to search for (e.g., ['lat', 'latitude'])
            
        Returns
        -------
        str or None
            The first matching coordinate name found in the dataset, or None if no match
        """
        if self._obj is None: return None
    
        for name in possible_names:
            if name in self._obj.coords:
                return name
        coord_names = {name.lower(): name for name in self._obj.coords}
        for name in possible_names:
            if name.lower() in coord_names:
                return coord_names[name.lower()]
        
        return None

    def _filter_by_season(self, data_subset, season='annual'):
        """
        Filter data by meteorological season.
        
        Parameters
        ----------
        data_subset : xarray.Dataset or xarray.DataArray
            Input data containing a time dimension
        season : str, default 'annual'
            Options: 'annual', 'djf', 'mam', 'jjas', 'son'
            
        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Data filtered to include only the specified season
        """
        if season.lower() == 'annual':
            return data_subset
        
        
        time_coord_name = self._get_coord_name(['time', 't'])
        if time_coord_name is None:
                raise ValueError("Cannot find time coordinate for seasonal filtering.")
        
        if time_coord_name not in data_subset.dims:
            raise ValueError(f"Cannot filter by season - '{time_coord_name}' dimension not found.")

        if 'month' in data_subset.coords:
            month_coord = data_subset['month']
        elif np.issubdtype(data_subset[time_coord_name].dtype, np.datetime64):
            month_coord = data_subset[time_coord_name].dt.month
        else:
            raise ValueError("Cannot determine month for seasonal filtering.")

        season_months = {'jjas': [6, 7, 8, 9], 'djf': [12, 1, 2], 'mam': [3, 4, 5], 'son': [9, 10, 11]}
        selected_months = season_months.get(season.lower())

        if selected_months:
            filtered_data = data_subset.where(month_coord.isin(selected_months), drop=True)
            if filtered_data[time_coord_name].size == 0:
                print(f"Warning: No data found for season '{season.upper()}' within the selected time range.")
            return filtered_data
        else:
            print(f"Warning: Unknown season '{season}'. Returning unfiltered data.")
            return data_subset

    def _apply_gaussian_filter(self, data_array, gaussian_sigma):
        """
        Apply Gaussian smoothing filter to spatial dimensions of data.
        
        Parameters
        ----------
        data_array : xarray.DataArray
            Input data array to smooth
        gaussian_sigma : float or None
            Standard deviation for Gaussian kernel
            
        Returns
        -------
        xarray.DataArray
            Smoothed data array
        bool
            Whether smoothing was applied
        """
        if gaussian_sigma is None or gaussian_sigma <= 0:
            return data_array, False

        if data_array.ndim < 2:
            print("Warning: Gaussian filtering skipped (data < 2D).")
            return data_array, False

        # Apply smoothing to last two dimensions (spatial)
        sigma_array = [0] * (data_array.ndim - 2) + [gaussian_sigma] * 2

        try:
            # Ensure data is computed if dask array
            computed_data = data_array.compute() if hasattr(data_array, 'compute') else data_array
            smoothed_values = ndimage.gaussian_filter(computed_data.values, sigma=sigma_array, mode='nearest')

            smoothed_da = xr.DataArray(
                smoothed_values, coords=computed_data.coords, dims=computed_data.dims,
                name=computed_data.name, attrs=computed_data.attrs
            )
            smoothed_da.attrs['filter'] = f'Gaussian smoothed (sigma={gaussian_sigma})'
            return smoothed_da, True
        except Exception as e:
            print(f"Warning: Could not apply Gaussian filter: {e}")
            return data_array, False

    def _select_data(self, variable, latitude=None, longitude=None, level=None, time_range=None):
        """
        Select data subset based on variable name and dimension constraints.
        
        Parameters
        ----------
        variable : str
            Name of the variable to select
        latitude, longitude, level, time_range : optional
            Coordinate constraints
            
        Returns
        -------
        xarray.DataArray
            Selected data subset
        str or None
            Level dimension name if found
        str or None
            Level operation type: 'range_selected', 'single_selected', or None
        """
        if variable not in self._obj.data_vars:
            raise ValueError(f"Variable '{variable}' not found.")

        data_var = self._obj[variable]
        selection_dict = {}
        method_dict = {}

        # Find coordinate names
        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        time_name = self._get_coord_name(['time', 't'])
        level_names = ['level', 'lev', 'plev', 'height', 'altitude', 'depth', 'z']
        level_dim_name = next((dim for dim in level_names if dim in data_var.dims), None)
        level_op = None

        # Validate latitude coordinates
        if latitude is not None:
            if lat_name is None:
                raise ValueError("No latitude coordinate found. Expected one of: lat, latitude, etc.")
            
            lat_min, lat_max = data_var[lat_name].min().item(), data_var[lat_name].max().item()
            
            # Check if latitude selection is completely outside available range
            if isinstance(latitude, slice):
                if latitude.start is not None and latitude.start > lat_max:
                    raise ValueError(f"Requested latitude minimum {latitude.start} is greater than available maximum {lat_max}")
                if latitude.stop is not None and latitude.stop < lat_min:
                    raise ValueError(f"Requested latitude maximum {latitude.stop} is less than available minimum {lat_min}")
            elif isinstance(latitude, (list, np.ndarray)):
                if min(latitude) > lat_max or max(latitude) < lat_min:
                    raise ValueError(f"Requested latitudes [{min(latitude)}, {max(latitude)}] are outside available range [{lat_min}, {lat_max}]")
            else:  # scalar
                if latitude < lat_min or latitude > lat_max:
                    raise ValueError(f"Requested latitude {latitude} is outside available range [{lat_min}, {lat_max}]")
            
            selection_dict[lat_name] = latitude

        # Validate longitude coordinates
        if longitude is not None:
            if lon_name is None:
                raise ValueError("No longitude coordinate found. Expected one of: lon, longitude, etc.")
            
            lon_min, lon_max = data_var[lon_name].min().item(), data_var[lon_name].max().item()
            
            # Check if longitude selection is completely outside available range
            if isinstance(longitude, slice):
                if longitude.start is not None and longitude.start > lon_max:
                    raise ValueError(f"Requested longitude minimum {longitude.start} is greater than available maximum {lon_max}")
                if longitude.stop is not None and longitude.stop < lon_min:
                    raise ValueError(f"Requested longitude maximum {longitude.stop} is less than available minimum {lon_min}")
            elif isinstance(longitude, (list, np.ndarray)):
                if min(longitude) > lon_max or max(longitude) < lon_min:
                    raise ValueError(f"Requested longitudes [{min(longitude)}, {max(longitude)}] are outside available range [{lon_min}, {lon_max}]")
            else:  # scalar
                if longitude < lon_min or longitude > lon_max:
                    raise ValueError(f"Requested longitude {longitude} is outside available range [{lon_min}, {lon_max}]")
            
            selection_dict[lon_name] = longitude

        # Validate time range
        if time_range is not None and time_name and time_name in data_var.dims:
            try:
                time_min, time_max = data_var[time_name].min().values, data_var[time_name].max().values
                
                if isinstance(time_range, slice):
                    if time_range.start is not None:
                        if np.datetime64(time_range.start) > time_max:
                            raise ValueError(f"Requested start time {time_range.start} is after available maximum {time_max}")
                    if time_range.stop is not None:
                        if np.datetime64(time_range.stop) < time_min:
                            raise ValueError(f"Requested end time {time_range.stop} is before available minimum {time_min}")
                elif isinstance(time_range, (list, np.ndarray)):
                    t_min, t_max = np.min(time_range), np.max(time_range)
                    if np.datetime64(t_min) > time_max or np.datetime64(t_max) < time_min:
                        raise ValueError(f"Requested time range is outside available range [{time_min}, {time_max}]")
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not validate time range: {e}")
            
            selection_dict[time_name] = time_range

        # Handle level selection
        if level_dim_name:
            if level is not None:
                level_min, level_max = data_var[level_dim_name].min().item(), data_var[level_dim_name].max().item()
                
                if isinstance(level, (slice, list, np.ndarray)):
                    if isinstance(level, slice):
                        if level.start is not None and level.start > level_max:
                            raise ValueError(f"Requested level minimum {level.start} is greater than available maximum {level_max}")
                        if level.stop is not None and level.stop < level_min:
                            raise ValueError(f"Requested level maximum {level.stop} is less than available minimum {level_min}")
                    elif min(level) > level_max or max(level) < level_min:
                        raise ValueError(f"Requested levels [{min(level)}, {max(level)}] are outside available range [{level_min}, {level_max}]")
                    
                    selection_dict[level_dim_name] = level
                    level_op = 'range_selected'
                elif isinstance(level, (int, float)):
                    if level < level_min * 0.5 or level > level_max * 1.5:
                        print(f"Warning: Requested level {level} is far from available range [{level_min}, {level_max}]")
                    
                    selection_dict[level_dim_name] = level
                    method_dict[level_dim_name] = 'nearest'
                    level_op = 'single_selected'
                else:
                    selection_dict[level_dim_name] = level
                    level_op = 'single_selected'
            elif len(data_var[level_dim_name]) > 1:
                level_val = data_var[level_dim_name].values[0]
                selection_dict[level_dim_name] = level_val
                level_op = 'single_selected'
                print(f"Warning: Multiple levels found. Using first level: {level_val}")
        elif level is not None:
            print("Warning: Level dimension not found. Ignoring 'level' parameter.")

        # Perform selection
        selected_data = data_var
        for dim, method in method_dict.items():
            if dim in selection_dict:
                selected_data = selected_data.sel({dim: selection_dict[dim]}, method=method)
                del selection_dict[dim]  
        
        if selection_dict:
            selected_data = selected_data.sel(selection_dict)
            
        return selected_data, level_dim_name, level_op


    def plot_mean(self, 
                  variable='air', 
                  latitude=None, 
                  longitude=None, 
                  level=None,
                  time_range=None, 
                  season='annual', 
                  gaussian_sigma=None,
                  figsize=(16, 10), 
                  cmap='coolwarm',
                  land_only = False,
                  levels=30,
                  save_plot_path = None
                  ): 
        """
        Plot spatial mean of a climate variable with optional filtering and smoothing.
        
        Parameters
        ----------
        variable : str, default 'air'
            Name of the climate variable to plot
        latitude, longitude : optional
            Coordinate constraints
        level : optional
            Vertical level(s) to select
        time_range : optional
            Time range for temporal averaging
        season : str, default 'annual'
            Options: 'annual', 'djf', 'mam', 'jja', 'jjas', 'son'
        gaussian_sigma : float or None, default None
            Smoothing parameter for spatial filtering
        figsize : tuple, default (16, 10)
            Figure dimensions
        cmap : str or matplotlib colormap, default 'coolwarm'
            Colormap for the plot
        land_only : bool, default False
            Mask out ocean areas if True
        levels : int or array-like, default 30
            Number of contour levels or explicit boundaries
        save_plot_path : str, optional
             Path to save the plot
            
        Returns
        -------
        matplotlib.axes.Axes
            The plot axes object
            
        Examples
        --------
        >>> ds.climate_plots.plot_mean(variable='temperature')
        >>> ds.climate_plots.plot_mean(variable='air', season='jjas',
        ...     latitude=slice(40, 6), longitude=slice(65, 100))
        >>> ds.climate_plots.plot_mean(variable='air', level=500,
        ...     gaussian_sigma=1, land_only=True)
        """
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )

        # Average over levels if a range was selected
        if level_op == 'range_selected' and level_dim_name in selected_data.dims:
             selected_data = selected_data.mean(dim=level_dim_name)
             print(f"Averaging over selected levels.")

        # Filter by season
        data_season = self._filter_by_season(selected_data, season)
        if data_season.size == 0:
            raise ValueError(f"No data after selections and season filter ('{season}').")

        # Compute time mean
        if 'time' in data_season.dims:
            if data_season.chunks:
                print("Computing time mean...")
                with ProgressBar(): mean_data = data_season.mean(dim='time').compute()
            else:
                 mean_data = data_season.mean(dim='time')
        else:
            mean_data = data_season
            print("Warning: No time dimension found for averaging.")

        # Apply smoothing
        smoothed_data, was_smoothed = self._apply_gaussian_filter(mean_data, gaussian_sigma)

        # Create the plot
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add geographic features
        if land_only:
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, zorder=3)
            ax.coastlines(zorder=3)
        else:
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1)
            ax.coastlines()
            
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False; gl.right_labels = False

        
        # Get coordinate names from the data
        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        time_name = self._get_coord_name(['time', 't'])

        if not lat_name or not lon_name:
            raise ValueError("Could not find latitude/longitude coordinates in the data")
        
        
        # Prepare contourf plot
        lon_coords = smoothed_data[lon_name].values
        lat_coords = smoothed_data[lat_name].values
        plot_data_np = smoothed_data.values

        if np.all(np.isnan(plot_data_np)):
             print("Warning: Data is all NaN after calculations. Cannot plot contours.")
             ax.set_title(f'{variable.capitalize()} - Data is all NaN')
             return ax

        im = ax.contourf(lon_coords, lat_coords, plot_data_np,
                         levels=levels, cmap=cmap,
                         transform=ccrs.PlateCarree(), extend='both')
                         
        # Add ocean mask if requested
        if land_only:
            ax.add_feature(cfeature.OCEAN, zorder=2, facecolor='white')

        # Add Colorbar
        unit_label = smoothed_data.attrs.get('units', '')
        cbar_label = f"{smoothed_data.attrs.get('long_name', variable)} ({unit_label})"
        plt.colorbar(im, label=cbar_label, orientation='vertical', pad=0.05, shrink=0.8)

        # Build title
        season_map = {
        'annual': "Annual", 'djf': "Winter (DJF)", 'mam': "Spring (MAM)",
        'jja': "Summer (JJA)", 'jjas': "Summer Monsoon (JJAS)", 'son': "Autumn (SON)"
        }
        season_str = season_map.get(season.lower(), season.upper())
        var_name = variable.replace('_', ' ').capitalize()
        title = f"{season_str} Mean of {var_name}"

        # Add level info to title
        if level_op == 'single_selected' and level_dim_name:
            actual_level = smoothed_data[level_dim_name].values.item()
            level_unit = smoothed_data[level_dim_name].attrs.get('units', '')
            title += f"\nLevel={actual_level} {level_unit}"
        elif level_op == 'range_selected':
            title += " (Level Mean)"

        # Add time period to title
        try:
            start_time = data_season[time_name].min().dt.strftime('%Y').item()
            end_time = data_season[time_name].max().dt.strftime('%Y').item()
            title += f"\n({start_time}-{end_time})"
        except Exception:
            if time_range is not None and hasattr(time_range, 'start') and hasattr(time_range, 'stop'):
                try:
                    title += f"\n({time_range.start.strftime('%Y')}-{time_range.stop.strftime('%Y')})"
                except (AttributeError, TypeError):
                    pass

        # Add smoothing info
        if was_smoothed:
            title += f"\nGaussian Smoothed (σ={gaussian_sigma})"

        ax.set_title(title, fontsize=12)

        # Set extent from data
        try:
            ax.set_extent([lon_coords.min(), lon_coords.max(), lat_coords.min(), lat_coords.max()], crs=ccrs.PlateCarree())
        except ValueError as e:
             print(f"Warning: Could not automatically set extent: {e}")

        if save_plot_path is not None:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        return ax


    def plot_std_time(self, 
                      variable='air', 
                      latitude=None, 
                      longitude=None, 
                      level=None,
                      time_range=None, 
                      season='annual', 
                      gaussian_sigma=None,
                      figsize=(16,10), 
                      cmap='viridis', 
                      land_only = False,
                      levels=30,
                      save_plot_path = None):
        """
        Plot temporal standard deviation of a climate variable.
        
        Parameters
        ----------
        variable : str, default 'air'
            Name of the climate variable to plot
        latitude, longitude : optional
            Coordinate constraints
        level : optional
            Vertical level(s) to select
        time_range : optional
            Time range for calculation
        season : str, default 'annual'
            Options: 'annual', 'djf', 'mam', 'jja', 'jjas', 'son'
        gaussian_sigma : float or None, default None
            Smoothing parameter for spatial filtering
        figsize : tuple, default (16, 10)
            Figure dimensions
        cmap : str or matplotlib colormap, default 'viridis'
            Colormap for the plot
        land_only : bool, default False
            Mask out ocean areas if True
        levels : int or array-like, default 30
            Number of contour levels or explicit boundaries
        save_plot_path : str, optional
            Path to save the plot
            
        Returns
        -------
        matplotlib.axes.Axes
            The plot axes object
            
        Examples
        --------
        >>> ds.climate_plots.plot_std_time(variable='temperature')
        >>> ds.climate_plots.plot_std_time(variable='prate', season='jjas',
        ...     latitude=slice(40, 6), longitude=slice(60, 100))
        >>> ds.climate_plots.plot_std_time(variable='sst', gaussian_sigma=1.0)
        """
        selected_data, level_dim_name, level_op = self._select_data(
            variable, latitude, longitude, level, time_range
        )

        # Validate time dimension
        if 'time' not in selected_data.dims:
             raise ValueError("Standard deviation requires 'time' dimension.")
        data_season = self._filter_by_season(selected_data, season)

        if data_season.size == 0:
            raise ValueError(f"No data after selections and season filter ('{season}').")
        if data_season.sizes['time'] < 2:
             raise ValueError(f"Std dev requires > 1 time point (found {data_season.sizes['time']}).")

        # Compute standard deviation
        if data_season.chunks:
            print("Computing standard deviation over time...")
            with ProgressBar(): std_data = data_season.std(dim='time').compute()
        else:
            std_data = data_season.std(dim='time')

        # Average across levels if needed
        if level_op == 'range_selected' and level_dim_name in std_data.dims:
             std_data = std_data.mean(dim=level_dim_name)
             print(f"Averaging standard deviation map across selected levels.")

        # Apply smoothing
        smoothed_data, was_smoothed = self._apply_gaussian_filter(std_data, gaussian_sigma)

        # Create the plot
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add geographic features
        if land_only:
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, zorder=3)
            ax.coastlines(zorder=3)
        else:
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1)
            ax.coastlines()
            
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False; gl.right_labels = False

       # Get coordinate names
        lat_name = self._get_coord_name(['lat', 'latitude', 'LAT', 'LATITUDE', 'y', 'rlat', 'nav_lat'])
        lon_name = self._get_coord_name(['lon', 'longitude', 'LON', 'LONGITUDE', 'x', 'rlon', 'nav_lon'])
        time_name = self._get_coord_name(['time', 't'])

        if not lat_name or not lon_name:
            raise ValueError("Could not find latitude/longitude coordinates in the data")

        lon_coords = smoothed_data[lon_name].values
        lat_coords = smoothed_data[lat_name].values
        plot_data_np = smoothed_data.values
        
        if np.all(np.isnan(plot_data_np)):
            print("Warning: Data is all NaN after calculations. Cannot plot contours.")
            ax.set_title(f'{variable.capitalize()} Std Dev - Data is all NaN')
            return ax

        im = ax.contourf(lon_coords, lat_coords, plot_data_np,
                         levels=levels, cmap=cmap,
                         transform=ccrs.PlateCarree(), extend='both')
                         
        # Add ocean mask if requested
        if land_only:
            ax.add_feature(cfeature.OCEAN, zorder=2, facecolor='white')

        # Colorbar
        unit_label = smoothed_data.attrs.get('units', '')
        cbar_label = f"Std. Dev. ({unit_label})"
        plt.colorbar(im, label=cbar_label, orientation='vertical', pad=0.05, shrink=0.8)

        # Build title
        season_map = {
            'annual': "Annual", 'djf': "Winter (DJF)", 'mam': "Spring (MAM)",
            'jja': "Summer (JJA)", 'jjas': "Summer Monsoon (JJAS)", 'son': "Autumn (SON)"
        }
        season_str = season_map.get(season.lower(), season.upper())
        var_name = variable.replace('_', ' ').capitalize()
        title = f"{season_str} Standard Deviation of {var_name}"

        # Add level info to title
        if level_op == 'single_selected' and level_dim_name:
            actual_level = smoothed_data[level_dim_name].values.item()
            level_unit = smoothed_data[level_dim_name].attrs.get('units', '')
            title += f"\nLevel={actual_level} {level_unit}"
        elif level_op == 'range_selected':
            title += " (Level Mean)"

        # Add time period to title
        try:
            start_time = data_season[time_name].min().dt.strftime('%Y').item()
            end_time = data_season[time_name].max().dt.strftime('%Y').item()
            title += f"\n({start_time}-{end_time})"
        except Exception:
            if time_range is not None and hasattr(time_range, 'start') and hasattr(time_range, 'stop'):
                try:
                    title += f"\n({time_range.start.strftime('%Y')}-{time_range.stop.strftime('%Y')})"
                except (AttributeError, TypeError):
                    pass

        # Add smoothing info
        if was_smoothed:
            title += f"\nGaussian Smoothed (σ={gaussian_sigma})"

        ax.set_title(title, fontsize=12)

        try:
            ax.set_extent([lon_coords.min(), lon_coords.max(), lat_coords.min(), lat_coords.max()], crs=ccrs.PlateCarree())
        except ValueError as e:
             print(f"Warning: Could not automatically set extent: {e}")

        if save_plot_path is not None:
            plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_plot_path}")
        return ax
    
__all__ = ['PlotsAccessor']
