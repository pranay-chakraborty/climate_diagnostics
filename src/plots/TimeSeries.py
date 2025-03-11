import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

class TimeSeries:
    """
    A class for loading, manipulating, and analyzing time series data using xarray.
    """
    
    def __init__(self, file_path=None):
        """
        Initialize the TimeSeries object.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the data file. If None, no data will be loaded.
        """
        self.file_path = file_path
        self.dataset = None
        
        if self.file_path is not None:
            self._load_data()
    
    def _load_data(self):
        """
        Load data from the file path.
        """
        try:
            self.dataset = xr.open_dataset(self.file_path)
            print(f"Dataset loaded from {self.file_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    def load_data(self, file_path):
        """
        Load data from a specified file path.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file.
        """
        self.file_path = file_path
        self._load_data()
    
   