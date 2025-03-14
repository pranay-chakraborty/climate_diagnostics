import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from dask.diagnostics import ProgressBar

class TimeSeries:
    def __init__(self, filepath=None):
        """
        Initialize the TimeSeries class.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the netCDF or other compatible data file.
        """
        self.filepath = filepath
        self.dataset = None
        self._load_data()
    def _load_data(self):
        """Load dataset from the provided filepath with automatic chunking."""
        try:
            if self.filepath:
                self.dataset = xr.open_dataset(self.filepath, chunks='auto')
                print(f"Dataset loaded from {self.filepath} with auto-chunking")
            else:
                print("Invalid filepath provided. Please specify a valid filepath.")
        except Exception as e:
            print(f"Error loading data: {e}")