import unittest
import pytest
import numpy as np
import xarray as xr
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Import climate_diagnostics to register the accessors
import climate_diagnostics

class TestTimeSeriesAccessor(unittest.TestCase):
    
    def setUp(self):
        self.create_mock_dataset()
        
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        self.mock_ds.to_netcdf(self.temp_file.name)
        self.temp_file.close()
        
        # Load dataset and use accessor approach
        self.ds = xr.open_dataset(self.temp_file.name)
    
    def tearDown(self):
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        if hasattr(self, 'ds'):
            self.ds.close()
    
    def create_mock_dataset(self):
        lat = np.linspace(-90, 90, 73)
        lon = np.linspace(0, 357.5, 144)
        level = np.array([1000, 850, 500, 200])
        time = pd.date_range('2020-01-01', periods=24, freq='MS')
        
        air_data = np.random.rand(len(time), len(level), len(lat), len(lon)) * 10 + 273.15
        precip_data = np.random.rand(len(time), len(lat), len(lon)) * 5
        
        self.mock_ds = xr.Dataset(
            data_vars={
                'air': xr.DataArray(
                    data=air_data,
                    dims=['time', 'level', 'lat', 'lon'],
                    coords={
                        'time': time,
                        'level': level,
                        'lat': lat,
                        'lon': lon
                    },
                    attrs={'units': 'K'}
                ),
                'precip': xr.DataArray(
                    data=precip_data,
                    dims=['time', 'lat', 'lon'],
                    coords={
                        'time': time,
                        'lat': lat,
                        'lon': lon
                    },
                    attrs={'units': 'mm/day'}
                )
            }
        )
    
    def test_accessor_availability(self):
        """Test that the climate_timeseries accessor is available"""
        self.assertTrue(hasattr(self.ds, 'climate_timeseries'))
        
    def test_dataset_loaded(self):
        """Test that the dataset was loaded successfully"""
        self.assertIsNotNone(self.ds)
        self.assertIn('air', self.ds.data_vars)
        self.assertIn('precip', self.ds.data_vars)
    
    def test_filter_by_season(self):
        from climate_diagnostics.utils.coord_utils import filter_by_season
        
        annual_data = filter_by_season(self.ds, 'annual')
        self.assertEqual(len(annual_data.time), 24)
        
        jjas_data = filter_by_season(self.ds, 'jjas')
        self.assertEqual(len(jjas_data.time), 8)
        for month in jjas_data.time.dt.month.values:
            self.assertIn(month, [6, 7, 8, 9])
        
        djf_data = filter_by_season(self.ds, 'djf')
        self.assertEqual(len(djf_data.time), 6)
        for month in djf_data.time.dt.month.values:
            self.assertIn(month, [12, 1, 2])
        
        mam_data = filter_by_season(self.ds, 'mam')
        self.assertEqual(len(mam_data.time), 6)
        for month in mam_data.time.dt.month.values:
            self.assertIn(month, [3, 4, 5])
        
        with patch('builtins.print') as mock_print:
            unknown_data = filter_by_season(self.ds, 'unknown')
            self.assertEqual(len(unknown_data.time), 24)
            mock_print.assert_called_with("Warning: Unknown season 'unknown'. Supported: ['jjas', 'djf', 'mam', 'son', 'jja']. Returning unfiltered data.")
    
    @patch('matplotlib.pyplot.figure')
    @patch('xarray.DataArray.plot')
    def test_plot_time_series(self, mock_plot, mock_figure):
        mock_plot.return_value = MagicMock()
        
        ax = self.ds.climate_timeseries.plot_time_series(variable='air')
        self.assertIsNotNone(ax)
        mock_plot.assert_called_once()
        
        mock_plot.reset_mock()
        
        ax = self.ds.climate_timeseries.plot_time_series(
            latitude=0, 
            longitude=180, 
            level=850, 
            time_range=slice('2020-01', '2020-12'),
            variable='air',
            figsize=(15, 8),
            season='jjas'
        )
        self.assertIsNotNone(ax)
        mock_plot.assert_called_once()
        
        with self.assertRaises(ValueError):
            self.ds.climate_timeseries.plot_time_series(variable='nonexistent_var')
            
        mock_plot.reset_mock()
        ax = self.ds.climate_timeseries.plot_time_series(year=2020, variable='air')
        self.assertIsNotNone(ax)
        mock_plot.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('xarray.DataArray.plot')
    def test_plot_std_space(self, mock_plot, mock_figure):
        mock_plot.return_value = MagicMock()
        
        ax = self.ds.climate_timeseries.plot_std_space(variable='air')
        self.assertIsNotNone(ax)
        mock_plot.assert_called_once()
        
        mock_plot.reset_mock()
        
        ax = self.ds.climate_timeseries.plot_std_space(
            latitude=slice(-30, 30), 
            longitude=slice(0, 180), 
            level=500, 
            time_range=slice('2020-01', '2020-12'),
            variable='air',
            figsize=(15, 8),
            season='djf'
        )
        self.assertIsNotNone(ax)
        mock_plot.assert_called_once()
        
        with self.assertRaises(ValueError):
            self.ds.climate_timeseries.plot_std_space(variable='nonexistent_var')
    
    @patch('matplotlib.pyplot.subplots')
    def test_decompose_time_series(self, mock_subplots):
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        results = self.ds.climate_timeseries.decompose_time_series(variable='air', plot_results=False)
        self.assertIsInstance(results, dict)
        self.assertIn('original', results)
        self.assertIn('trend', results)
        self.assertIn('seasonal', results)
        self.assertIn('residual', results)
        
        results, fig = self.ds.climate_timeseries.decompose_time_series(
            variable='air',
            level=850,
            latitude=slice(-30, 30),
            longitude=slice(0, 180),
            time_range=slice('2020-01', '2020-12'),
            season='annual',
            stl_seasonal=13,
            stl_period=12,
            area_weighted=True,
            plot_results=True,
            figsize=(14, 12)
        )
        self.assertIsInstance(results, dict)
        self.assertEqual(fig, mock_fig)
        
        with self.assertRaises(ValueError):
            self.ds.climate_timeseries.decompose_time_series(variable='nonexistent_var')

@pytest.fixture
def mock_dataset():
    lat = np.linspace(-90, 90, 73)
    lon = np.linspace(0, 357.5, 144)
    level = np.array([1000, 850, 500, 200])
    time = pd.date_range('2020-01-01', periods=24, freq='MS')
    
    air_data = np.random.rand(len(time), len(level), len(lat), len(lon)) * 10 + 273.15
    precip_data = np.random.rand(len(time), len(lat), len(lon)) * 5
    
    ds = xr.Dataset(
        data_vars={
            'air': xr.DataArray(
                data=air_data,
                dims=['time', 'level', 'lat', 'lon'],
                coords={
                    'time': time,
                    'level': level,
                    'lat': lat,
                    'lon': lon
                },
                attrs={'units': 'K'}
            ),
            'precip': xr.DataArray(
                data=precip_data,
                dims=['time', 'lat', 'lon'],
                coords={
                    'time': time,
                    'lat': lat,
                    'lon': lon
                },
                attrs={'units': 'mm/day'}
            )
        }
    )
    return ds

# Additional pytest-style tests for accessor functionality
@pytest.fixture  
def sample_dataset():
    """Create a sample dataset for pytest tests"""
    lat = np.linspace(-90, 90, 73)
    lon = np.linspace(0, 357.5, 144) 
    level = np.array([1000, 850, 500, 200])
    time = pd.date_range('2020-01-01', periods=24, freq='MS')
    
    air_data = np.random.rand(len(time), len(level), len(lat), len(lon)) * 10 + 273.15
    precip_data = np.random.rand(len(time), len(lat), len(lon)) * 5
    
    ds = xr.Dataset(
        data_vars={
            'air': xr.DataArray(
                data=air_data,
                dims=['time', 'level', 'lat', 'lon'],
                coords={'time': time, 'level': level, 'lat': lat, 'lon': lon},
                attrs={'units': 'K'}
            ),
            'precip': xr.DataArray(
                data=precip_data,
                dims=['time', 'lat', 'lon'], 
                coords={'time': time, 'lat': lat, 'lon': lon},
                attrs={'units': 'mm/day'}
            )
        }
    )
    return ds

def test_accessor_registration(sample_dataset):
    """Test that accessors are properly registered"""
    assert hasattr(sample_dataset, 'climate_timeseries')
    assert hasattr(sample_dataset, 'climate_trends') 
    assert hasattr(sample_dataset, 'climate_plots')

@pytest.mark.parametrize("season, expected_months", [
    ('annual', list(range(1, 13))),
    ('jjas', [6, 7, 8, 9]),
    ('djf', [12, 1, 2]),
    ('mam', [3, 4, 5]),
])  
def test_season_filtering_utils(sample_dataset, season, expected_months):
    """Test season filtering utility function"""
    from climate_diagnostics.utils.coord_utils import filter_by_season
    
    filtered_data = filter_by_season(sample_dataset, season)
    if season == 'annual':
        assert len(filtered_data.time) == 24
    else:
        months = filtered_data.time.dt.month.values
        assert all(m in expected_months for m in months)

@pytest.mark.parametrize("variable", ['air', 'precip'])
def test_accessor_methods_exist(sample_dataset, variable):
    """Test that accessor methods exist and can be called"""
    # Test method existence
    assert hasattr(sample_dataset.climate_timeseries, 'plot_time_series')
    assert hasattr(sample_dataset.climate_timeseries, 'decompose_time_series')
    assert hasattr(sample_dataset.climate_timeseries, 'plot_std_space')

if __name__ == '__main__':
    unittest.main()