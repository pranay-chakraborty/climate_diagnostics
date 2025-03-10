import unittest
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from plots.urls import uwnd_mon_mean, vwnd_mon_mean, air_mon_mean
from plots import plots


class TestPlots(unittest.TestCase):

    def setUp(self):
        self.mock_air_data = MagicMock()
        self.mock_air_data.__getitem__.return_value = MagicMock()

    def test_init_default(self):
        p = plots()
        self.assertEqual(p.uwnd_url, uwnd_mon_mean)
        self.assertEqual(p.vwnd_url, vwnd_mon_mean)
        self.assertEqual(p.air_temp_url, air_mon_mean)
        self.assertIsNone(p.uwnd_data)
        self.assertIsNone(p.vwnd_data)
        self.assertIsNone(p.air_temp_data)

    def test_init_custom(self):
        custom_uwnd = "custom_uwnd_url"
        custom_vwnd = "custom_vwnd_url"
        custom_air = "custom_air_url"

        p = plots(custom_uwnd, custom_vwnd, custom_air)

        self.assertEqual(p.uwnd_url, custom_uwnd)
        self.assertEqual(p.vwnd_url, custom_vwnd)
        self.assertEqual(p.air_temp_url, custom_air)

    @patch('xarray.open_dataset')
    def test_load_air_temperature_success(self, mock_open_dataset):
        mock_open_dataset.return_value = self.mock_air_data

        p = plots()
        result = p.load_air_temperature()

        self.assertTrue(result)
        self.assertEqual(p.air_temp_data, self.mock_air_data)
        mock_open_dataset.assert_called_once_with(air_mon_mean)

    @patch('xarray.open_dataset')
    def test_load_air_temperature_failure(self, mock_open_dataset):
        mock_open_dataset.side_effect = Exception("Test error")

        p = plots()
        result = p.load_air_temperature()

        self.assertFalse(result)
        self.assertIsNone(p.air_temp_data)

    @patch('xarray.open_dataset')
    def test_plot_air_temperature_with_file_path(self, mock_open_dataset):
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = MagicMock()
        mock_dataset.variables = {'air': MagicMock()}
        mock_open_dataset.return_value = mock_dataset

        test_file_path = "test_file.nc"
        p = plots()

        with patch.object(plt, 'subplots', return_value=(MagicMock(), MagicMock())) as mock_subplots:
            ax = p.plot_air_temperature(file_path=test_file_path)

            mock_open_dataset.assert_called_once_with(test_file_path)
            mock_subplots.assert_called_once()
            ax.coastlines.assert_called_once()
            ax.gridlines.assert_called_once_with(draw_labels=True)

    def test_plot_air_temperature_with_loaded_data(self):
        p = plots()
        mock_dataset = MagicMock()
        mock_plot = MagicMock()
        mock_air_variable = MagicMock()
        mock_air_variable.plot.return_value = mock_plot
        mock_dataset.__getitem__.return_value = mock_air_variable
        mock_dataset.variables = {'air': MagicMock()}
        p.air_temp_data = mock_dataset

        with patch.object(plt, 'subplots', return_value=(MagicMock(), MagicMock())) as mock_subplots:
            ax = p.plot_air_temperature()

            mock_dataset.__getitem__.assert_called_once_with('air')
            mock_air_variable.plot.assert_called_once()
            mock_subplots.assert_called_once()
            ax.coastlines.assert_called_once()
            ax.gridlines.assert_called_once_with(draw_labels=True)

    def test_plot_air_temperature_no_data(self):
        p = plots()

        with self.assertRaises(ValueError):
            p.plot_air_temperature()

    @patch('xarray.open_dataset')
    def test_plot_air_temperature_no_air_variable(self, mock_open_dataset):
        mock_dataset = MagicMock()
        mock_dataset.variables = {}  # No 'air' variable
        mock_open_dataset.return_value = mock_dataset

        p = plots()
        test_file_path = "test_file.nc"

        with self.assertRaises(ValueError):
            p.plot_air_temperature(file_path=test_file_path)

    @patch('xarray.open_dataset')
    def test_plot_air_temperature_with_custom_kwargs(self, mock_open_dataset):
        mock_dataset = MagicMock()
        mock_air_variable = MagicMock()
        mock_dataset.__getitem__.return_value = mock_air_variable
        mock_dataset.variables = {'air': MagicMock()}
        mock_open_dataset.return_value = mock_dataset

        test_file_path = "test_file.nc"
        p = plots()
        custom_kwargs = {'cmap': 'viridis', 'vmin': -10, 'vmax': 40}

        with patch.object(plt, 'subplots', return_value=(MagicMock(), MagicMock())):
            p.plot_air_temperature(file_path=test_file_path, **custom_kwargs)

            mock_air_variable.plot.assert_called_once_with(ax=unittest.mock.ANY, **custom_kwargs)
