import unittest
import pytest
import numpy as np
import xarray as xr
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from climate_diagnostics import TimeSeries

class TestTimeSeries(unittest.TestCase):
    
    def setUp(self):
        self.create_mock_dataset()
        
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        self.mock_ds.to_netcdf(self.temp_file.name)
        self.temp_file.close()
        
        self.ts = TimeSeries(self.temp_file.name)
    
    def tearDown(self):
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
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
    
    def test_init_and_load_data(self):
        self.assertIsNotNone(self.ts.dataset)
        self.assertEqual(self.ts.filepath, self.temp_file.name)
        
        with patch('builtins.print') as mock_print:
            ts_invalid = TimeSeries("nonexistent_file.nc")
            self.assertTrue(mock_print.called)
            error_msg = mock_print.call_args[0][0]
            self.assertIn("Error loading data", error_msg)
            self.assertIn("No such file or directory", error_msg)
            self.assertIn("nonexistent_file.nc", error_msg)
    
    def test_filter_by_season(self):
        annual_data = self.ts._filter_by_season('annual')
        self.assertEqual(len(annual_data.time), 24)
        
        jjas_data = self.ts._filter_by_season('jjas')
        self.assertEqual(len(jjas_data.time), 8)
        for month in jjas_data.time.dt.month.values:
            self.assertIn(month, [6, 7, 8, 9])
        
        djf_data = self.ts._filter_by_season('djf')
        self.assertEqual(len(djf_data.time), 6)
        for month in djf_data.time.dt.month.values:
            self.assertIn(month, [12, 1, 2])
        
        mam_data = self.ts._filter_by_season('mam')
        self.assertEqual(len(mam_data.time), 6)
        for month in mam_data.time.dt.month.values:
            self.assertIn(month, [3, 4, 5])
        
        with patch('builtins.print') as mock_print:
            unknown_data = self.ts._filter_by_season('unknown')
            self.assertEqual(len(unknown_data.time), 24)
            mock_print.assert_called_with("Warning: Unknown season 'unknown'. Using annual data.")
    
    @patch('matplotlib.pyplot.figure')
    @patch('xarray.DataArray.plot')
    def test_plot_time_series(self, mock_plot, mock_figure):
        mock_plot.return_value = MagicMock()
        
        ax = self.ts.plot_time_series(variable='air')
        self.assertIsNotNone(ax)
        mock_plot.assert_called_once()
        
        mock_plot.reset_mock()
        
        ax = self.ts.plot_time_series(
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
            self.ts.plot_time_series(variable='nonexistent_var')
            
        mock_plot.reset_mock()
        ax = self.ts.plot_time_series(year=2020, variable='air')
        self.assertIsNotNone(ax)
        mock_plot.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('xarray.DataArray.plot')
    def test_plot_std_space(self, mock_plot, mock_figure):
        mock_plot.return_value = MagicMock()
        
        ax = self.ts.plot_std_space(variable='air')
        self.assertIsNotNone(ax)
        mock_plot.assert_called_once()
        
        mock_plot.reset_mock()
        
        ax = self.ts.plot_std_space(
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
            self.ts.plot_std_space(variable='nonexistent_var')
    
    @patch('matplotlib.pyplot.subplots')
    def test_decompose_time_series(self, mock_subplots):
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        results = self.ts.decompose_time_series(variable='air', plot_results=False)
        self.assertIsInstance(results, dict)
        self.assertIn('original', results)
        self.assertIn('trend', results)
        self.assertIn('seasonal', results)
        self.assertIn('residual', results)
        
        results, fig = self.ts.decompose_time_series(
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
            self.ts.decompose_time_series(variable='nonexistent_var')

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

@pytest.fixture
def time_series_instance(mock_dataset):
    temp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    mock_dataset.to_netcdf(temp_path)
    
    ts = TimeSeries(temp_path)
    
    yield ts
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)

def test_initialization(time_series_instance):
    assert time_series_instance.dataset is not None
    assert hasattr(time_series_instance, 'filepath')

def test_seasons(time_series_instance):
    jjas_data = time_series_instance._filter_by_season('jjas')
    assert len(jjas_data.time) == 8
    assert all(m in [6, 7, 8, 9] for m in jjas_data.time.dt.month.values)
    
    djf_data = time_series_instance._filter_by_season('djf')
    assert len(djf_data.time) == 6
    assert all(m in [12, 1, 2] for m in djf_data.time.dt.month.values)

def test_dataset_variables(time_series_instance):
    assert 'air' in time_series_instance.dataset.data_vars
    assert 'precip' in time_series_instance.dataset.data_vars

@pytest.mark.parametrize("season, expected_months", [
    ('annual', list(range(1, 13))),
    ('jjas', [6, 7, 8, 9]),
    ('djf', [12, 1, 2]),
    ('mam', [3, 4, 5]),
])
def test_season_filtering(time_series_instance, season, expected_months):
    filtered_data = time_series_instance._filter_by_season(season)
    months = filtered_data.time.dt.month.values
    assert all(m in expected_months for m in months)

@pytest.mark.parametrize("variable", ['air', 'precip'])
def test_time_series_plot_variables(time_series_instance, variable, monkeypatch):
    monkeypatch.setattr(plt, 'figure', lambda **kwargs: MagicMock())
    monkeypatch.setattr(xr.DataArray, 'plot', lambda *args, **kwargs: MagicMock())
    
    result = time_series_instance.plot_time_series(variable=variable)
    assert result is not None

def test_plot_errors(time_series_instance):
    with pytest.raises(ValueError, match="Variable 'nonexistent' not found in dataset"):
        time_series_instance.plot_time_series(variable='nonexistent')
    
    with pytest.raises(TypeError, match="No numeric data to plot"):
        time_series_instance.plot_time_series(
            time_range=slice('2030-01-01', '2030-12-31'),
            variable='air'
        )

def test_decompose_time_series(time_series_instance, monkeypatch):
    class MockSTLResult:
        def __init__(self):
            self.trend = pd.Series([1, 2, 3])
            self.seasonal = pd.Series([0.1, 0.2, 0.3])
            self.resid = pd.Series([0.01, 0.02, 0.03])
    
    class MockSTL:
        def __init__(self, *args, **kwargs):
            pass
        
        def fit(self):
            return MockSTLResult()
    
    monkeypatch.setattr('statsmodels.tsa.seasonal.STL', MockSTL)
    monkeypatch.setattr(plt, 'subplots', lambda *args, **kwargs: (MagicMock(), [MagicMock() for _ in range(4)]))
    
    results = time_series_instance.decompose_time_series(
        variable='air', 
        plot_results=False
    )
    
    assert isinstance(results, dict)
    assert 'original' in results
    assert 'trend' in results
    assert 'seasonal' in results
    assert 'residual' in results

if __name__ == '__main__':
    unittest.main()