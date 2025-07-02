"""
Test module for climate_diagnostics.plots.plot module.

This module contains unit tests for the plotting functionality,
including the PlotsAccessor class and its methods.
"""

import unittest
import pytest
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import climate_diagnostics to register the accessors
import climate_diagnostics


class TestPlotsAccessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.ds = self.create_mock_dataset()
        plt.ioff()  # Turn off interactive plotting
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
    
    def create_mock_dataset(self):
        """Create a mock dataset for testing."""
        # Create coordinate arrays
        time = pd.date_range('2020-01-01', '2021-12-31', freq='MS')
        lat = np.linspace(-90, 90, 36)
        lon = np.linspace(-180, 179, 72)
        level = [1000, 850, 500, 200]
        
        # Create realistic temperature data
        np.random.seed(42)
        base_temp = 273.15 + 15  # 15°C in Kelvin
        
        # Create temperature data with spatial and temporal patterns
        temp_data = np.random.randn(len(time), len(level), len(lat), len(lon)) * 10 + base_temp
        
        # Add latitude gradient (warmer at equator)
        lat_gradient = 20 * np.cos(np.radians(lat))
        temp_data += lat_gradient[np.newaxis, np.newaxis, :, np.newaxis]
        
        # Create precipitation data
        precip_data = np.random.exponential(2.0, temp_data.shape)
        
        # Create dataset
        ds = xr.Dataset({
            'air': (['time', 'level', 'lat', 'lon'], temp_data),
            'prate': (['time', 'level', 'lat', 'lon'], precip_data)
        }, coords={
            'time': time,
            'level': level,
            'lat': lat,
            'lon': lon
        })
        
        # Add attributes
        ds['air'].attrs = {'units': 'K', 'long_name': 'Air Temperature'}
        ds['prate'].attrs = {'units': 'mm/day', 'long_name': 'Precipitation Rate'}
        
        return ds
    
    def test_accessor_availability(self):
        """Test that the climate_plots accessor is available."""
        self.assertTrue(hasattr(self.ds, 'climate_plots'))
        self.assertIsNotNone(self.ds.climate_plots)
    
    def test_dataset_loaded(self):
        """Test that the dataset was loaded successfully."""
        self.assertIsInstance(self.ds, xr.Dataset)
        self.assertIn('air', self.ds.data_vars)
        self.assertIn('prate', self.ds.data_vars)
        self.assertIn('time', self.ds.coords)
        self.assertIn('lat', self.ds.coords)
        self.assertIn('lon', self.ds.coords)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_plot_mean(self, mock_figure, mock_show):
        """Test plot_mean method."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Test basic functionality
        result = self.ds.climate_plots.plot_mean(variable='air')
        self.assertIsNotNone(result)
        mock_figure.assert_called()
        
        # Test with parameters
        result = self.ds.climate_plots.plot_mean(
            variable='air',
            latitude=slice(-30, 30),
            longitude=slice(0, 180),
            level=850,
            season='jjas',
            figsize=(12, 8)
        )
        self.assertIsNotNone(result)
        
        # Test with invalid variable
        with self.assertRaises(ValueError):
            self.ds.climate_plots.plot_mean(variable='nonexistent_var')
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_plot_std_time(self, mock_figure, mock_show):
        """Test plot_std_time method."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Test basic functionality
        result = self.ds.climate_plots.plot_std_time(variable='air')
        self.assertIsNotNone(result)
        mock_figure.assert_called()
        
        # Test with parameters
        result = self.ds.climate_plots.plot_std_time(
            variable='air',
            latitude=slice(-30, 30),
            longitude=slice(0, 180),
            level=850,
            season='annual'
        )
        self.assertIsNotNone(result)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_plot_percentile_spatial(self, mock_figure, mock_show):
        """Test plot_percentile_spatial method."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Test basic functionality
        result = self.ds.climate_plots.plot_percentile_spatial(
            variable='prate', 
            percentile=95
        )
        self.assertIsNotNone(result)
        mock_figure.assert_called()
        
        # Test with different percentile
        result = self.ds.climate_plots.plot_percentile_spatial(
            variable='prate',
            percentile=5,
            latitude=slice(-60, 60)
        )
        self.assertIsNotNone(result)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_save_plot_functionality(self, mock_figure, mock_show):
        """Test that save_plot_path parameter works."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'test_plot.png')
            
            # Mock the savefig method
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                result = self.ds.climate_plots.plot_mean(
                    variable='air',
                    save_plot_path=save_path
                )
                mock_savefig.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with invalid variable
        with self.assertRaises(ValueError):
            self.ds.climate_plots.plot_mean(variable='invalid_variable')
        
        # Test with invalid season
        with self.assertRaises(ValueError):
            self.ds.climate_plots.plot_mean(variable='air', season='invalid_season')


# Additional pytest-style tests for specific plotting functionality
@pytest.fixture
def sample_dataset():
    """Create a sample dataset for pytest tests."""
    time = pd.date_range('2020-01-01', '2020-12-31', freq='MS')
    lat = np.linspace(-60, 60, 25)
    lon = np.linspace(-120, 120, 49)
    
    np.random.seed(42)
    temp_data = 273.15 + 15 + np.random.randn(len(time), len(lat), len(lon)) * 5
    precip_data = np.random.exponential(3.0, temp_data.shape)
    
    ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], temp_data),
        'precipitation': (['time', 'lat', 'lon'], precip_data)
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon
    })
    
    ds['temperature'].attrs = {'units': 'K', 'long_name': 'Temperature'}
    ds['precipitation'].attrs = {'units': 'mm/day', 'long_name': 'Precipitation'}
    
    return ds


def test_accessor_registration(sample_dataset):
    """Test that the plots accessor is properly registered."""
    assert hasattr(sample_dataset, 'climate_plots')
    assert sample_dataset.climate_plots is not None


@pytest.mark.parametrize("variable", ['temperature', 'precipitation'])
def test_plot_methods_with_variables(sample_dataset, variable):
    """Test plot methods with different variables."""
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.figure'):
        result = sample_dataset.climate_plots.plot_mean(variable=variable)
        assert result is not None


@pytest.mark.parametrize("season", ['annual', 'jjas', 'djf', 'mam'])
def test_seasonal_plotting(sample_dataset, season):
    """Test plotting with different seasons."""
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.figure'):
        result = sample_dataset.climate_plots.plot_mean(
            variable='temperature', 
            season=season
        )
        assert result is not None


def test_plot_with_regional_selection(sample_dataset):
    """Test plotting with regional selections."""
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.figure'):
        result = sample_dataset.climate_plots.plot_mean(
            variable='temperature',
            latitude=slice(-30, 30),
            longitude=slice(-60, 60)
        )
        assert result is not None


if __name__ == '__main__':
    unittest.main()
