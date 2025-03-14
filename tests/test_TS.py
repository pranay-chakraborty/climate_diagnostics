import unittest
from unittest.mock import patch, MagicMock, call, ANY
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import pytest

from climate_diagnostics.TimeSeries.TimeSeries import TimeSeries

class TestTimeSeries(unittest.TestCase):
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        self.temp_file.close()
        
        lat = np.linspace(-90, 90, 10)
        lon = np.linspace(-180, 180, 10)
        time = pd.date_range('2023-01-01', periods=5)
        level = [1000, 500]
        
        data = np.random.rand(5, 2, 10, 10)
        ds = xr.Dataset(
            data_vars=dict(
                air=(["time", "level", "lat", "lon"], data, {'units': 'K'}),
            ),
            coords=dict(
                lon=(["lon"], lon),
                lat=(["lat"], lat),
                time=time,
                level=level,
            ),
        )
        ds.to_netcdf(self.temp_file.name)
        self.valid_filepath = self.temp_file.name
        
    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
    
    def test_init_with_valid_filepath(self):
        ts = TimeSeries(self.valid_filepath)
        self.assertIsNotNone(ts.dataset)
        self.assertEqual(ts.filepath, self.valid_filepath)
    
    def test_init_with_invalid_filepath(self):
        with patch('builtins.print') as mock_print:
            ts = TimeSeries("invalid_file.nc")
            self.assertIsNone(ts.dataset)
            
            called = False
            for call_args, _ in mock_print.call_args_list:
                if any("Error loading data" in str(arg) for arg in call_args):
                    called = True
                    break
            self.assertTrue(called, "Expected error message about loading data wasn't printed")
    
    def test_init_with_no_filepath(self):
        with patch('builtins.print') as mock_print:
            ts = TimeSeries()
            self.assertIsNone(ts.dataset)
            self.assertIsNone(ts.filepath)
            mock_print.assert_called_once_with("Invalid filepath provided. Please specify a valid filepath.")
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('dask.diagnostics.ProgressBar')
    def test_plot_time_series_basic(self, mock_progress, mock_tight_layout, mock_grid, mock_ylabel, mock_xlabel, mock_title, mock_figure):
        small_ds = xr.Dataset(
            data_vars=dict(
                air=(["time", "lat", "lon"], np.random.rand(3, 2, 2), {'units': 'K'}),
            ),
            coords=dict(
                lon=[0, 1],
                lat=[0, 1],
                time=pd.date_range('2023-01-01', periods=3),
            ),
        )
        
        with patch.object(TimeSeries, '_load_data'):
            ts = TimeSeries()
            ts.dataset = small_ds
            
            with patch('xarray.DataArray.plot') as mock_plot:
                mock_ax = MagicMock()
                mock_plot.return_value = mock_ax
                
                result = ts.plot_time_series(variable='air')
                
                mock_figure.assert_called_once()
                mock_title.assert_called_once()
                mock_xlabel.assert_called_once()
                mock_ylabel.assert_called_once()
                mock_grid.assert_called_once()
                mock_plot.assert_called_once()
                self.assertEqual(result, mock_ax)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('dask.diagnostics.ProgressBar')
    def test_plot_time_series_with_selections(self, mock_progress, mock_tight_layout, mock_grid, mock_ylabel, mock_xlabel, mock_title, mock_figure):
        ds = xr.Dataset(
            data_vars=dict(
                air=(["time", "level", "lat", "lon"], 
                     np.random.rand(4, 2, 5, 5), 
                     {'units': 'K'}),
            ),
            coords=dict(
                lon=[-20, -10, 0, 10, 20],
                lat=[-20, -10, 0, 10, 20],
                time=pd.date_range('2023-01-01', periods=4),
                level=[1000, 500],
            ),
        )
        
        with patch.object(TimeSeries, '_load_data'):
            ts = TimeSeries()
            ts.dataset = ds
            
            with patch('xarray.DataArray.plot') as mock_plot:
                mock_ax = MagicMock()
                mock_plot.return_value = mock_ax
                
                result = ts.plot_time_series(
                    latitude=slice(-10, 10),
                    longitude=slice(-10, 10),
                    level=500,
                    time_range=slice('2023-01-01', '2023-01-04'),
                    variable='air'
                )
                
                mock_figure.assert_called_once()
                mock_title.assert_called_once()
                mock_xlabel.assert_called_once()
                mock_ylabel.assert_called_once()
                mock_grid.assert_called_once()
                mock_plot.assert_called_once()
                self.assertEqual(result, mock_ax)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('dask.diagnostics.ProgressBar')
    def test_plot_time_series_with_level_list(self, mock_progress, mock_tight_layout, mock_grid, mock_ylabel, mock_xlabel, mock_title, mock_figure):
        ds = xr.Dataset(
            data_vars=dict(
                air=(["time", "level", "lat", "lon"], 
                     np.random.rand(4, 3, 5, 5), 
                     {'units': 'K'}),
            ),
            coords=dict(
                lon=[-20, -10, 0, 10, 20],
                lat=[-20, -10, 0, 10, 20],
                time=pd.date_range('2023-01-01', periods=4),
                level=[1000, 850, 500],
            ),
        )
        
        with patch.object(TimeSeries, '_load_data'):
            ts = TimeSeries()
            ts.dataset = ds
            
            with patch('xarray.DataArray.plot') as mock_plot:
                mock_ax = MagicMock()
                mock_plot.return_value = mock_ax
                
                result = ts.plot_time_series(
                    level=[1000, 850],
                    variable='air'
                )
                
                mock_figure.assert_called_once()
                mock_title.assert_called_once()
                mock_xlabel.assert_called_once()
                mock_ylabel.assert_called_once()
                mock_grid.assert_called_once()
                mock_plot.assert_called_once()
                self.assertEqual(result, mock_ax)
    
    def test_plot_time_series_invalid_variable(self):
        ts = TimeSeries(self.valid_filepath)
        
        with self.assertRaises(ValueError) as context:
            ts.plot_time_series(variable='nonexistent_var')
        
        self.assertIn("Variable 'nonexistent_var' not found in dataset", str(context.exception))
    
    def test_plot_time_series_no_dataset(self):
        ts = TimeSeries()
        
        with self.assertRaises(ValueError) as context:
            ts.plot_time_series()
        
        self.assertIn("No dataset available for plotting", str(context.exception))
    
    def test_plot_time_series_no_time_dim(self):
        ds = xr.Dataset(
            data_vars=dict(
                air=(["lat", "lon"], np.random.rand(5, 5), {'units': 'K'}),
            ),
            coords=dict(
                lon=np.linspace(-180, 180, 5),
                lat=np.linspace(-90, 90, 5),
            ),
        )
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            ds.to_netcdf(tmp.name)
            
            try:
                ts = TimeSeries(tmp.name)
                
                with self.assertRaises(ValueError) as context:
                    ts.plot_time_series()
                
                self.assertIn("Time dimension not found in dataset", str(context.exception))
            finally:
                os.unlink(tmp.name)
                
    def test_plot_time_series_with_single_point(self):
        ds = xr.Dataset(
            data_vars=dict(
                air=(["time", "lat", "lon"], 
                     np.random.rand(3, 5, 5), 
                     {'units': 'K'}),
            ),
            coords=dict(
                lon=[-20, -10, 0, 10, 20],
                lat=[-20, -10, 0, 10, 20],
                time=pd.date_range('2023-01-01', periods=3),
            ),
        )
        
        with patch.object(TimeSeries, '_load_data'):
            ts = TimeSeries()
            ts.dataset = ds
            
            with patch('xarray.DataArray.plot') as mock_plot:
                mock_ax = MagicMock()
                mock_plot.return_value = mock_ax
                
                result = ts.plot_time_series(
                    latitude=6,
                    longitude=70,
                    variable='air'
                )
                
                mock_plot.assert_called_once()
                self.assertEqual(result, mock_ax)

if __name__ == "__main__":
    unittest.main()