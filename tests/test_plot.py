import pytest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import os
import tempfile
from datetime import datetime
from plots import Plots

@pytest.fixture
def sample_dataset():
    lat = np.linspace(-90, 90, 73)
    lon = np.linspace(0, 357.5, 144)
    time = np.array([np.datetime64('2025-01-01'), np.datetime64('2025-01-02')])
    level = np.array([1000, 850, 500])

    air = np.random.rand(2, 3, 73, 144)

    ds = xr.Dataset(
        data_vars=dict(
            air=(["time", "level", "lat", "lon"], air, {"units": "degC"})
        ),
        coords=dict(
            lon=("lon", lon),
            lat=("lat", lat),
            time=("time", time),
            level=("level", level)
        ),
        attrs=dict(description="Test dataset for Plots class")
    )

    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        ds.to_netcdf(tmp.name)
        return tmp.name

@pytest.fixture(autouse=True)
def cleanup(request, sample_dataset):
    def remove_file():
        try:
            os.remove(sample_dataset)
        except:
            pass
    request.addfinalizer(remove_file)

def test_init_with_valid_filepath(sample_dataset):
    with patch('builtins.print') as mock_print:
        plot_obj = Plots(filepath=sample_dataset)
        mock_print.assert_called_with(f"Dataset loaded from {sample_dataset} with auto-chunking")
        assert plot_obj.dataset is not None
        assert plot_obj.filepath == sample_dataset

def test_init_with_invalid_filepath():
    with patch('builtins.print') as mock_print:
        plot_obj = Plots(filepath=None)
        mock_print.assert_called_with("Invalid filepath provided. Please specify a valid filepath.")
        assert plot_obj.dataset is None

def test_init_with_nonexistent_filepath():
    with patch('builtins.print') as mock_print:
        plot_obj = Plots(filepath="/path/that/does/not/exist.nc")
        mock_print.assert_called_with(mock_print.call_args[0][0])
        assert plot_obj.dataset is None

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.axes')
def test_plot_mean_no_dataset(mock_axes, mock_figure):
    plot_obj = Plots(filepath=None)
    with pytest.raises(ValueError) as excinfo:
        plot_obj.plot_mean()
    assert "No dataset available for plotting" in str(excinfo.value)

@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.axes')
@patch('matplotlib.pyplot.colorbar')
@patch('matplotlib.pyplot.title')
def test_plot_mean_temp_basic(mock_title, mock_colorbar, mock_axes, mock_figure, sample_dataset):
    mock_ax = MagicMock()
    mock_axes.return_value = mock_ax

    plot_obj = Plots(filepath=sample_dataset)

    with patch.object(xr.DataArray, 'plot', return_value=MagicMock()) as mock_plot:
        result = plot_obj.plot_mean()

        assert mock_figure.called
        assert mock_axes.called
        assert mock_ax.coastlines.called
        assert mock_ax.gridlines.called
        assert mock_plot.called
        assert mock_colorbar.called
        assert mock_title.called

        assert result == mock_ax

def test_plot_mean_temp_with_selections(sample_dataset):
    plot_obj = Plots(filepath=sample_dataset)

    with patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.axes', return_value=MagicMock()), \
         patch.object(xr.DataArray, 'plot', return_value=MagicMock()):

        result = plot_obj.plot_mean(latitude=0, longitude=0)
        assert result is not None

        result = plot_obj.plot_mean(
            latitude=slice(-30, 30),
            longitude=slice(0, 180)
        )
        assert result is not None

        result = plot_obj.plot_mean(level=1000)
        assert result is not None

        result = plot_obj.plot_mean(time_range=slice('2025-01-01', '2025-01-02'))
        assert result is not None

        with pytest.raises(ValueError):
            plot_obj.plot_mean(variable="nonexistent_var")
