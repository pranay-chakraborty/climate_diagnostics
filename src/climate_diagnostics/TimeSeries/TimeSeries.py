import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from statsmodels.tsa.seasonal import STL
from ..utils import get_coord_name, filter_by_season, get_or_create_dask_client, select_process_data, get_spatial_mean

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
        get_or_create_dask_client()
        # --- Step 1: Select and process the data ---
        data_selected = select_process_data(
            self._obj, variable, latitude, longitude, level, time_range, season, year
        )
        time_name = get_coord_name(data_selected, ['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found for time series plot.")
        if data_selected.size == 0:
            print("No data to plot after selections.")
            return None

        # --- Step 2: Calculate the spatial mean time series ---
        ts_data = get_spatial_mean(data_selected, area_weighted)

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
        weight_display = "Area-Weighted " if area_weighted and get_coord_name(data_selected, ['lat', 'latitude']) in data_selected.dims else ""
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
        get_or_create_dask_client()
        # --- Step 1: Select and process the data ---
        data_selected = select_process_data(
             self._obj, variable, latitude, longitude, level, time_range, season, year
        )
        time_name = get_coord_name(data_selected, ['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension not found for spatial standard deviation plot.")
        if data_selected.size == 0:
            print("No data to plot after selections.")
            return None

        # --- Step 2: Calculate the time series of spatial standard deviation ---
        lat_name = get_coord_name(data_selected, ['lat', 'latitude'])
        lon_name = get_coord_name(data_selected, ['lon', 'longitude'])
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
        get_or_create_dask_client()
        # --- Step 1: Select and process data for the time series ---
        data_selected = select_process_data(
             self._obj, variable, latitude, longitude, level, time_range, season, year
        )
        time_name = get_coord_name(data_selected, ['time', 't'])
        if not time_name or time_name not in data_selected.dims:
            raise ValueError("Time dimension required for decomposition.")
        if data_selected.size == 0: print("No data after selections."); return (None, None) if plot_results else None

        # --- Step 2: Compute the spatially-averaged time series ---
        ts_spatial_mean = get_spatial_mean(data_selected, area_weighted)
        if hasattr(ts_spatial_mean, 'chunks') and ts_spatial_mean.chunks:
            print("Computing mean time series for decomposition...")
            with ProgressBar(): ts_spatial_mean = ts_spatial_mean.compute()
        if ts_spatial_mean.size == 0 : print("Time series empty after spatial mean."); return (None, None) if plot_results else None
        
        # --- Step 3: Convert the xarray DataArray to a pandas Series for STL ---
        ts_spatial_mean = ts_spatial_mean.squeeze(drop=True)
        if ts_spatial_mean.ndim > 1:
             raise ValueError(f"Spatially averaged data for STL still has >1 dimension: {ts_spatial_mean.dims}")

        try:
            ts_pd = ts_spatial_mean.to_series()
        except Exception as e_pd:
            raise ValueError(f"Could not convert to pandas Series for STL: {e_pd}")

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
