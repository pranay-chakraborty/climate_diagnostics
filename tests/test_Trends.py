"""
Test module for climate_diagnostics.TimeSeries.Trends module.

This module contains unit tests for the trend analysis functionality,
including the TrendsAccessor class and its methods.
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


class TestTrendsAccessor(unittest.TestCase):
    
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
        time = pd.date_range('2015-01-01', '2020-12-31', freq='MS')  # 6 years of monthly data
        lat = np.linspace(-60, 60, 25)
        lon = np.linspace(-120, 120, 49)
        level = [1000, 850, 500]
        
        # Create realistic temperature data with a trend
        np.random.seed(42)
        base_temp = 273.15 + 15  # 15°C in Kelvin
        
        # Add a linear trend over time
        trend = np.linspace(0, 2, len(time))  # 2K warming over the period
        
        temp_data = np.zeros((len(time), len(level), len(lat), len(lon)))
        for i, t in enumerate(time):
            # Add seasonal cycle
            day_of_year = t.dayofyear
            seasonal = 5 * np.sin(2 * np.pi * day_of_year / 365.25)
            
            # Add spatial patterns
            lat_gradient = 20 * np.cos(np.radians(lat))
            
            # Combine base temperature, trend, seasonal cycle, and random noise
            temp_data[i, :, :, :] = (base_temp + trend[i] + seasonal + 
                                   lat_gradient[np.newaxis, :, np.newaxis] + 
                                   np.random.randn(len(level), len(lat), len(lon)) * 2)
        
        # Create dataset
        ds = xr.Dataset({
            'air': (['time', 'level', 'lat', 'lon'], temp_data)
        }, coords={
            'time': time,
            'level': level,
            'lat': lat,
            'lon': lon
        })
        
        # Add attributes
        ds['air'].attrs = {'units': 'K', 'long_name': 'Air Temperature'}
        
        return ds
    
    def test_accessor_availability(self):
        """Test that the climate_trends accessor is available."""
        self.assertTrue(hasattr(self.ds, 'climate_trends'))
        self.assertIsNotNone(self.ds.climate_trends)
    
    def test_dataset_loaded(self):
        """Test that the dataset was loaded successfully."""
        self.assertIsInstance(self.ds, xr.Dataset)
        self.assertIn('air', self.ds.data_vars)
        self.assertIn('time', self.ds.coords)
        self.assertIn('lat', self.ds.coords)
        self.assertIn('lon', self.ds.coords)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_calculate_trend_basic(self, mock_figure, mock_show):
        """Test basic trend calculation functionality."""
        mock_figure.return_value = MagicMock()
        
        # Test with return_results=True to get the results
        result = self.ds.climate_trends.calculate_trend(
            variable='air',
            latitude=slice(-30, 30),
            longitude=slice(-60, 60),
            plot=False,
            return_results=True
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('trend_component', result)
        self.assertIn('trend_statistics', result)
        self.assertIn('predicted_trend_line', result)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_calculate_trend_with_plot(self, mock_figure, mock_show):
        """Test trend calculation with plotting."""
        mock_figure.return_value = MagicMock()
        
        # Test with plotting enabled
        result = self.ds.climate_trends.calculate_trend(
            variable='air',
            latitude=0,
            longitude=0,
            plot=True,
            return_results=False
        )
        
        # Should return None when return_results=False
        self.assertIsNone(result)
        mock_figure.assert_called()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure') 
    def test_calculate_trend_seasonal(self, mock_figure, mock_show):
        """Test trend calculation with seasonal filtering."""
        mock_figure.return_value = MagicMock()
        
        result = self.ds.climate_trends.calculate_trend(
            variable='air',
            latitude=slice(-10, 10),
            longitude=slice(-10, 10),
            season='jjas',
            plot=False,
            return_results=True
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('region_details', result)
        self.assertEqual(result['region_details']['season'], 'jjas')
    
    def test_calculate_trend_error_cases(self):
        """Test error handling in trend calculation."""
        # Test with invalid variable
        with self.assertRaises(ValueError):
            self.ds.climate_trends.calculate_trend(variable='nonexistent_var')
        
        # Test with invalid frequency
        with self.assertRaises(ValueError):
            self.ds.climate_trends.calculate_trend(variable='air', frequency='X')
        
        # Test with invalid period
        with self.assertRaises(ValueError):
            self.ds.climate_trends.calculate_trend(variable='air', period=-1)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_calculate_spatial_trends_basic(self, mock_figure, mock_show):
        """Test basic spatial trends calculation."""
        mock_figure.return_value = MagicMock()
        
        # Test with a smaller spatial domain for faster computation
        result = self.ds.climate_trends.calculate_spatial_trends(
            variable='air',
            latitude=slice(-20, 20),
            longitude=slice(-30, 30),
            num_years=1,
            plot_map=False,
            n_workers=1  # Use single worker for testing
        )
        
        self.assertIsInstance(result, xr.DataArray)
        self.assertTrue('trend' in result.name)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_calculate_spatial_trends_with_plot(self, mock_figure, mock_show):
        """Test spatial trends calculation with plotting."""
        mock_figure.return_value = MagicMock()
        
        result = self.ds.climate_trends.calculate_spatial_trends(
            variable='air',
            latitude=slice(-10, 10),
            longitude=slice(-20, 20),
            num_years=10,  # Decadal trends
            plot_map=True,
            n_workers=1
        )
        
        self.assertIsInstance(result, xr.DataArray)
        mock_figure.assert_called()
    
    def test_spatial_trends_error_cases(self):
        """Test error handling in spatial trends calculation."""
        # Test with invalid variable
        with self.assertRaises(ValueError):
            self.ds.climate_trends.calculate_spatial_trends(variable='nonexistent_var')
        
        # Test with invalid num_years
        with self.assertRaises(ValueError):
            self.ds.climate_trends.calculate_spatial_trends(variable='air', num_years=0)
        
        # Test with invalid n_workers
        with self.assertRaises(ValueError):
            self.ds.climate_trends.calculate_spatial_trends(variable='air', n_workers=0)
    
    def test_optimize_for_trends(self):
        """Test the optimize_for_trends method."""
        # This may return None if chunking utilities are not available
        result = self.ds.climate_trends.optimize_for_trends(variable='air')
        
        # Should return either an optimized dataset or None
        self.assertTrue(result is None or isinstance(result, xr.Dataset))
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_save_plot_functionality(self, mock_figure, mock_show, mock_savefig):
        """Test that save_plot_path parameter works."""
        mock_figure.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'test_trend.png')
            
            self.ds.climate_trends.calculate_trend(
                variable='air',
                latitude=0,
                longitude=0,
                plot=True,
                save_plot_path=save_path,
                return_results=False
            )
            
            mock_savefig.assert_called()


# Additional pytest-style tests for specific trend functionality
@pytest.fixture
def trend_dataset():
    """Create a dataset with known trend for testing."""
    time = pd.date_range('2010-01-01', '2020-12-31', freq='MS')
    lat = np.linspace(-30, 30, 13)
    lon = np.linspace(-60, 60, 25)
    
    # Create data with known linear trend
    np.random.seed(123)
    trend_per_year = 0.1  # 0.1 K/year warming
    
    temp_data = np.zeros((len(time), len(lat), len(lon)))
    for i, t in enumerate(time):
        years_from_start = (t.year - 2010) + (t.month - 1) / 12
        trend_value = trend_per_year * years_from_start
        
        # Add some noise but preserve the trend
        temp_data[i, :, :] = (273.15 + 15 + trend_value + 
                              np.random.randn(len(lat), len(lon)) * 0.5)
    
    ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], temp_data)
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon
    })
    
    ds['temperature'].attrs = {'units': 'K', 'long_name': 'Temperature'}
    return ds


def test_accessor_registration(trend_dataset):
    """Test that the trends accessor is properly registered."""
    assert hasattr(trend_dataset, 'climate_trends')
    assert trend_dataset.climate_trends is not None


@pytest.mark.parametrize("season", ['annual', 'jjas', 'djf'])
def test_trend_calculation_seasons(trend_dataset, season):
    """Test trend calculation with different seasons."""
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.figure'):
        result = trend_dataset.climate_trends.calculate_trend(
            variable='temperature',
            season=season,
            plot=False,
            return_results=True
        )
        assert isinstance(result, dict)
        assert 'trend_statistics' in result


@pytest.mark.parametrize("num_years", [1, 5, 10])
def test_spatial_trends_different_periods(trend_dataset, num_years):
    """Test spatial trends with different time periods."""
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.figure'):
        result = trend_dataset.climate_trends.calculate_spatial_trends(
            variable='temperature',
            latitude=slice(-15, 15),
            longitude=slice(-30, 30),
            num_years=num_years,
            plot_map=False,
            n_workers=1
        )
        assert isinstance(result, xr.DataArray)


def test_trend_with_point_selection(trend_dataset):
    """Test trend calculation with point selection."""
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.figure'):
        result = trend_dataset.climate_trends.calculate_trend(
            variable='temperature',
            latitude=0.0,
            longitude=0.0,
            plot=False,
            return_results=True
        )
        assert isinstance(result, dict)
        assert result['calculation_type'] == 'Point'


def test_trend_with_regional_selection(trend_dataset):
    """Test trend calculation with regional selection."""
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.figure'):
        result = trend_dataset.climate_trends.calculate_trend(
            variable='temperature',
            latitude=slice(-10, 10),
            longitude=slice(-20, 20),
            plot=False,
            return_results=True
        )
        assert isinstance(result, dict)
        assert result['calculation_type'] == 'Regional'


if __name__ == '__main__':
    unittest.main()
