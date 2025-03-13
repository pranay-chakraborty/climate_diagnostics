import unittest
from unittest.mock import patch, MagicMock, call, ANY
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import pytest

from plots import Plots

class TestPlots(unittest.TestCase):
    
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
        plots = Plots(self.valid_filepath)
        self.assertIsNotNone(plots.dataset)
        self.assertEqual(plots.filepath, self.valid_filepath)
    
    def test_init_with_invalid_filepath(self):
        with patch('builtins.print') as mock_print:
            plots = Plots("invalid_file.nc")
            self.assertIsNone(plots.dataset)
            
            called = False
            for call_args, _ in mock_print.call_args_list:
                if any("Error loading data" in str(arg) for arg in call_args):
                    called = True
                    break
            self.assertTrue(called, "Expected error message about loading data wasn't printed")
    
    def test_init_with_no_filepath(self):
        with patch('builtins.print') as mock_print:
            plots = Plots()
            self.assertIsNone(plots.dataset)
            self.assertIsNone(plots.filepath)
            mock_print.assert_called_once_with("Invalid filepath provided. Please specify a valid filepath.")
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axes')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.title')
    def test_plot_mean_basic(self, mock_title, mock_colorbar, mock_axes, mock_figure):
        mock_ax = MagicMock()
        mock_axes.return_value = mock_ax
        
        small_ds = xr.Dataset(
            data_vars=dict(
                air=(["lat", "lon"], np.random.rand(2, 2), {'units': 'K'}),
            ),
            coords=dict(
                lon=[0, 1],
                lat=[0, 1],
            ),
        )
        
        with patch.object(Plots, '_load_data'):
            plots = Plots()
            plots.dataset = small_ds
            
            with patch('xarray.DataArray.plot') as mock_plot:
                mock_plot.return_value = MagicMock()
                
                result = plots.plot_mean(variable='air')
                
                mock_figure.assert_called_once()
                mock_axes.assert_called_once()
                mock_plot.assert_called_once()
                self.assertEqual(result, mock_ax)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axes')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.title')
    def test_plot_mean_with_selections(self, mock_title, mock_colorbar, mock_axes, mock_figure):
        mock_ax = MagicMock()
        mock_axes.return_value = mock_ax
        
        ds = xr.Dataset(
            data_vars=dict(
                air=(["time", "level", "lat", "lon"], 
                     np.random.rand(2, 2, 5, 5), 
                     {'units': 'K'}),
            ),
            coords=dict(
                lon=[-20, -10, 0, 10, 20],
                lat=[-20, -10, 0, 10, 20],
                time=pd.date_range('2023-01-01', periods=2),
                level=[1000, 500],
            ),
        )
        
        with patch.object(Plots, '_load_data'):
            plots = Plots()
            plots.dataset = ds
            
            with patch('xarray.DataArray.plot') as mock_plot:
                mock_plot.return_value = MagicMock()
                
                result = plots.plot_mean(
                    latitude=slice(-10, 10),
                    longitude=slice(-10, 10),
                    level=500,
                    time_range=slice('2023-01-01', '2023-01-02'),
                    variable='air'
                )
                
                self.assertEqual(result, mock_ax)
                mock_figure.assert_called_once()
                mock_axes.assert_called_once()
                mock_plot.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axes')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.title')
    def test_plot_std_time(self, mock_title, mock_colorbar, mock_axes, mock_figure):
        mock_ax = MagicMock()
        mock_axes.return_value = mock_ax
        
        small_ds = xr.Dataset(
            data_vars=dict(
                air=(["time", "lat", "lon"], np.random.rand(2, 2, 2), {'units': 'K'}),
            ),
            coords=dict(
                lon=[0, 1],
                lat=[0, 1],
                time=pd.date_range('2023-01-01', periods=2),
            ),
        )
        
        with patch.object(Plots, '_load_data'):
            plots = Plots()
            plots.dataset = small_ds
            
            with patch('xarray.DataArray.plot') as mock_plot:
                mock_plot.return_value = MagicMock()
                
                result = plots.plot_std_time(variable='air')
                
                mock_figure.assert_called_once()
                mock_axes.assert_called_once()
                mock_plot.assert_called_once()
                self.assertEqual(result, mock_ax)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axes')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.title')
    def test_plot_std_time_with_selections(self, mock_title, mock_colorbar, mock_axes, mock_figure):
        mock_ax = MagicMock()
        mock_axes.return_value = mock_ax
        
        ds = xr.Dataset(
            data_vars=dict(
                air=(["time", "lat", "lon"], 
                     np.random.rand(2, 5, 5), 
                     {'units': 'K'}),
            ),
            coords=dict(
                lon=[-20, -10, 0, 10, 20],
                lat=[-20, -10, 0, 10, 20],
                time=pd.date_range('2023-01-01', periods=2),
            ),
        )
        
        with patch.object(Plots, '_load_data'):
            plots = Plots()
            plots.dataset = ds
            
            with patch('xarray.DataArray.plot') as mock_plot:
                mock_plot.return_value = MagicMock()
                
                result = plots.plot_std_time(
                    variable='air',
                    latitude=slice(-10, 10),
                    longitude=slice(-10, 10)
                )
                
                self.assertEqual(result, mock_ax)
                mock_figure.assert_called_once()
                mock_axes.assert_called_once()
                mock_plot.assert_called_once()
    
    def test_plot_mean_invalid_variable(self):
        plots = Plots(self.valid_filepath)
        
        with self.assertRaises(ValueError) as context:
            plots.plot_mean(variable='nonexistent_var')
        
        self.assertIn("Variable 'nonexistent_var' not found in dataset", str(context.exception))
    
    def test_plot_mean_no_dataset(self):
        plots = Plots()
        
        with self.assertRaises(ValueError) as context:
            plots.plot_mean()
        
        self.assertIn("No dataset available for plotting", str(context.exception))
    
    def test_plot_std_time_no_time_dim(self):
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
                plots = Plots(tmp.name)
                
                with self.assertRaises(ValueError) as context:
                    plots.plot_std_time()
                
                self.assertIn("Time dimension not found in dataset", str(context.exception))
            finally:
                os.unlink(tmp.name)

if __name__ == "__main__":
    unittest.main()