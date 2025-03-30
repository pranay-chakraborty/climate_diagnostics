import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dask.diagnostics import ProgressBar
from sklearn.linear_model import LinearRegression
import pandas as pd
from statsmodels.tsa.seasonal import STL
import warnings

class Trends:
    def __init__(self, filepath=None):
        """Initialize the Trends class for analyzing climate data trends."""
        self.filepath = filepath
        self.dataset = None
        self._load_data()

    def _load_data(self):
        """Load dataset with automatic chunking."""
        try:
            if self.filepath:
                self.dataset = xr.open_dataset(self.filepath, chunks={'time': 'auto'})
                print(f"Dataset loaded from {self.filepath} with auto-chunking")
            else:
                print("No filepath provided. Initialize dataset manually or provide filepath.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def _get_coord_name(self, possible_names):
        """Find the actual coordinate name in the dataset."""
        if self.dataset is None: return None
        for name in possible_names:
            if name in self.dataset.coords:
                return name
        return None

    def _filter_by_season(self, data_array, season='annual', time_coord_name='time'):
        """Filter an xarray DataArray by meteorological season."""
        if season.lower() == 'annual':
            return data_array

        if time_coord_name not in data_array.dims:
            print(f"Warning: Cannot filter by season - no '{time_coord_name}' dimension found.")
            return data_array

        if 'month' not in data_array.coords:
            if np.issubdtype(data_array[time_coord_name].dtype, np.datetime64):
                data_array = data_array.assign_coords(month=(data_array[time_coord_name].dt.month))
            else:
                print(f"Warning: Cannot create 'month' coordinate - '{time_coord_name}' is not datetime type.")
                return data_array

        month_coord = data_array['month']
        if season.lower() == 'jjas':
            mask = month_coord.isin([6, 7, 8, 9])
        elif season.lower() == 'djf':
            mask = month_coord.isin([12, 1, 2])
        elif season.lower() == 'mam':
            mask = month_coord.isin([3, 4, 5])
        else:
            print(f"Warning: Unknown season '{season}'. Using annual data.")
            return data_array

        filtered_data = data_array.where(mask, drop=True)
        if time_coord_name in filtered_data.dims and len(filtered_data[time_coord_name]) == 0:
            warnings.warn(f"Filtering by season '{season}' resulted in no data points.", UserWarning)

        return filtered_data

    def calculate_trend(self,
                        variable='air',
                        latitude=None,
                        longitude=None,
                        level=None,
                        season='annual',
                        area_weighted=True,
                        period=12,
                        plot=True,
                        return_results=False):
        """
        Calculate trends from time series using STL decomposition and linear regression.
        
        Parameters:
        -----------
        variable: Variable name (default: 'air')
        latitude, longitude: Selection by point (float) or region (slice)
        level: Pressure level selection 
        season: 'annual', 'jjas', 'djf', or 'mam'
        area_weighted: Apply cosine latitude weighting
        period: Period for STL decomposition (default: 12 for monthly)
        plot: Generate visualization
        return_results: Return calculation results dictionary
        """
        if self.dataset is None: raise ValueError("Dataset not loaded.")
        if variable not in self.dataset.variables: raise ValueError(f"Variable '{variable}' not found.")

        # Get coordinate names
        lat_coord = self._get_coord_name(['lat', 'latitude'])
        lon_coord = self._get_coord_name(['lon', 'longitude'])
        level_coord = self._get_coord_name(['level', 'lev', 'plev', 'zlev'])
        time_coord = self._get_coord_name(['time'])
        
        if not all([lat_coord, lon_coord, time_coord]):
            raise ValueError("Dataset must contain recognizable time, latitude, and longitude coordinates.")

        # Initial data selection
        data_var = self.dataset[variable]
        if time_coord not in data_var.dims:
            raise ValueError(f"Variable '{variable}' has no '{time_coord}' dimension.")

        # Determine calculation type
        is_global = latitude is None and longitude is None
        is_point_lat = isinstance(latitude, (int, float))
        is_point_lon = isinstance(longitude, (int, float))
        is_point = is_point_lat and is_point_lon
        
        calculation_type = 'global' if is_global else ('point' if is_point else 'region')

        # Set default area weighting
        if area_weighted is None:
            area_weighted = calculation_type == 'global'
        if calculation_type == 'point':
            area_weighted = False

        print(f"Starting trend calculation: type='{calculation_type}', variable='{variable}', season='{season}', area_weighted={area_weighted}")

        # Filter by season
        data_var = self._filter_by_season(data_var, season=season, time_coord_name=time_coord)
        if len(data_var[time_coord]) == 0:
            raise ValueError(f"No data remains for variable '{variable}' after filtering for season '{season}'.")

        # Separate slice and point selectors for proper application
        sel_slices = {}
        sel_points = {}
        level_selection_info = ""
        level_dim_exists = level_coord and level_coord in data_var.dims

        # Set up latitude/longitude selectors
        if latitude is not None:
            if isinstance(latitude, slice): sel_slices[lat_coord] = latitude
            else: sel_points[lat_coord] = latitude
        if longitude is not None:
            if isinstance(longitude, slice): sel_slices[lon_coord] = longitude
            else: sel_points[lon_coord] = longitude

        # Handle level selection
        if level is not None:
            if level_dim_exists:
                level_selection_info = f"level(s)={level}"
                if isinstance(level, slice): sel_slices[level_coord] = level
                else: sel_points[level_coord] = level
            else:
                print(f"Warning: Level coordinate '{level_coord}' not found. Level selection ignored.")
        elif level_dim_exists and calculation_type != 'point':
            default_level_val = data_var[level_coord][0].item()
            sel_points[level_coord] = default_level_val
            level_selection_info = f"level={default_level_val} (defaulted)"
            print(f"Warning: Defaulting to first level: {default_level_val}")

        # Apply spatial selections in correct order
        try:
            if sel_slices:
                print(f"Applying slice selection: {sel_slices}")
                data_var = data_var.sel(sel_slices)

            if sel_points:
                print(f"Applying point selection with method='nearest': {sel_points}")
                data_var = data_var.sel(sel_points, method='nearest')
        except Exception as e:
            current_selectors = {**sel_slices, **sel_points}
            raise ValueError(f"Error during .sel() operation with selectors {current_selectors}: {e}")

        # Validate selection results
        selected_sizes = data_var.sizes
        print(f"Dimensions after selection: {selected_sizes}")
        spatial_dims_selected = [d for d in [lat_coord, lon_coord, level_coord] if d in selected_sizes]
        if any(selected_sizes[d] == 0 for d in spatial_dims_selected) or \
           time_coord not in selected_sizes or selected_sizes[time_coord] == 0:
            raise ValueError(f"Selection resulted in zero data points ({selected_sizes}). Check slice ranges.")

        # Spatial averaging if needed
        dims_to_average = [d for d in [lat_coord, lon_coord] if d in selected_sizes and selected_sizes[d] > 1]
        processed_ts_da = data_var
        region_coords = {d: data_var[d].values for d in data_var.coords if d != time_coord and d in data_var.coords}

        if dims_to_average:
            print(f"Averaging over dimensions: {dims_to_average}")
            if area_weighted:
                if lat_coord not in data_var.coords:
                    raise ValueError("Latitude coordinate needed for area weighting.")
                weights = np.cos(np.deg2rad(data_var[lat_coord]))
                weights.name = "weights"
                weights = weights.broadcast_like(data_var)
                with ProgressBar(dt=1.0):
                    processed_ts_da = data_var.weighted(weights).mean(dim=dims_to_average).compute()
            else:
                with ProgressBar(dt=1.0):
                    processed_ts_da = data_var.mean(dim=dims_to_average).compute()
        else:
            if calculation_type != 'point' and not dims_to_average:
                print("Selection resulted in a single spatial point. No averaging needed.")
                calculation_type = 'point'
            else:
                print("Point selection complete. No averaging needed.")
            
            with ProgressBar(dt=1.0):
                processed_ts_da = data_var.compute()
            region_coords = {d: processed_ts_da[d].values for d in processed_ts_da.coords 
                            if d != time_coord and d in processed_ts_da.coords}

        # Convert to pandas series
        if processed_ts_da is None:
            raise RuntimeError("Time series data not processed.")
        if time_coord not in processed_ts_da.dims:
            if not processed_ts_da.dims:
                raise ValueError("Processed data became a scalar value, cannot create time series.")
            raise ValueError(f"Processed data lost time dimension. Final dims: {processed_ts_da.dims}")

        print("Converting to Pandas Series...")
        ts_pd = processed_ts_da.to_pandas()

        # Handle different pandas output types
        if not isinstance(ts_pd, pd.Series):
            if isinstance(ts_pd, pd.DataFrame) and len(ts_pd.columns)==1:
                warnings.warn(f"Conversion resulted in DataFrame; extracting single column '{ts_pd.columns[0]}'.")
                ts_pd = ts_pd.iloc[:, 0]
            else:
                if np.isscalar(ts_pd):
                    raise TypeError(f"Expected pandas Series, but got a scalar ({ts_pd}).")
                raise TypeError(f"Expected pandas Series, but got {type(ts_pd)}")

        if ts_pd.isnull().all():
            raise ValueError(f"Time series is all NaNs after selection/averaging.")

        # Clean time series and check length for STL
        original_index = ts_pd.index
        ts_pd_clean = ts_pd.dropna()

        if ts_pd_clean.empty:
            raise ValueError(f"Time series is all NaNs after dropping NaN values.")
            
        min_stl_len = 2 * period
        if len(ts_pd_clean) < min_stl_len:
            raise ValueError(f"Time series length ({len(ts_pd_clean)}) is less than required minimum ({min_stl_len}).")

        # Apply STL decomposition
        print("Applying STL decomposition...")
        try:
            stl_result = STL(ts_pd_clean, period=period, robust=True).fit()
            trend_component = stl_result.trend.reindex(original_index)
        except Exception as e:
            print(f"Error during STL decomposition: {e}")
            raise

        # Linear regression on trend component
        print("Performing linear regression...")
        trend_component_clean = trend_component.dropna()
        if trend_component_clean.empty:
            raise ValueError("Trend component is all NaNs after STL.")

        try:
            # Convert index to numerical format
            if pd.api.types.is_numeric_dtype(trend_component_clean.index):
                dates_numeric = trend_component_clean.index.values.reshape(-1, 1)
            elif pd.api.types.is_datetime64_any_dtype(trend_component_clean.index):
                dates_numeric = pd.to_numeric(trend_component_clean.index).values.reshape(-1, 1)
            else:
                raise TypeError(f"Trend index type ({trend_component_clean.index.dtype}) not recognized.")

            y_train = trend_component_clean.values
            reg = LinearRegression()
            reg.fit(dates_numeric, y_train)
            y_pred_values = reg.predict(dates_numeric)
            y_pred_series = pd.Series(y_pred_values, index=trend_component_clean.index).reindex(original_index)
        except Exception as e:
            print(f"Error during linear regression: {e}")
            raise

        # Plotting
        if plot:
            print("Generating plot...")
            plt.figure(figsize=(16, 10), dpi=100)
            plt.scatter(trend_component.index, trend_component.values, color='blue', alpha=0.5, s=10, 
                       label='STL Trend Component')
            plt.plot(y_pred_series.index, y_pred_series.values, color='red', linewidth=2, 
                    label='Linear Trend Fit')

            # Dynamic title generation
            title = f"Trend: {variable.capitalize()}"
            units = processed_ts_da.attrs.get('units', '')
            ylabel = f'{variable.capitalize()} Trend' + (f' ({units})' if units else '')

            coord_strs = []
            
            def format_coord(coord_name, coords_dict):
                if coord_name in coords_dict:
                    vals = np.atleast_1d(coords_dict[coord_name])
                    if len(vals) == 0: return None
                    name_map = {'lat': 'Lat', 'latitude': 'Lat', 'lon': 'Lon', 'longitude': 'Lon', 
                               'level': 'Level', 'lev': 'Level', 'plev': 'Level'}
                    prefix = name_map.get(coord_name, coord_name.capitalize())
                    if len(vals) > 1:
                        return f"{prefix}=[{np.nanmin(vals):.2f}:{np.nanmax(vals):.2f}]"
                    else:
                        scalar_val = vals.item() if vals.ndim == 0 or vals.size == 1 else vals[0]
                        return f"{prefix}={scalar_val:.2f}"
                return None

            lat_str = format_coord(lat_coord, region_coords)
            lon_str = format_coord(lon_coord, region_coords)
            level_str = format_coord(level_coord, region_coords)

            # Title components based on calculation type
            if calculation_type == 'point':
                title += f" (Point Analysis)"
                if lat_str: coord_strs.append(lat_str)
                if lon_str: coord_strs.append(lon_str)
                if level_str: coord_strs.append(level_str)
            elif calculation_type == 'region' or calculation_type == 'global':
                avg_str = f"{'Weighted' if area_weighted else 'Unweighted'} Mean" if dims_to_average else "Selection"
                title += f" ({'Global' if is_global else 'Regional'} {avg_str})"
                if lat_str: coord_strs.append(lat_str)
                if lon_str: coord_strs.append(lon_str)
                if level_str: coord_strs.append(level_str)
            else:
                title += " (Unknown Type)"

            if season.lower() != 'annual': coord_strs.append(f"Season={season.upper()}")
            title += "\n" + ", ".join(filter(None, coord_strs))

            plt.title(title + "\n(STL Trend + Linear Regression)", fontsize=14)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(10))
            plt.tight_layout()
            plt.show()

        # Return results
        if return_results:
            results = {
                'calculation_type': calculation_type,
                'raw_timeseries': ts_pd,
                'trend_component': trend_component,
                'regression_model': reg,
                'predicted_trend': y_pred_series,
                'area_weighted': area_weighted,
                'region_details': {'variable': variable, 'season': season, 'level_info': level_selection_info},
                'actual_coords': region_coords,
                'stl_period': period
            }
            return results
        else:
            return None