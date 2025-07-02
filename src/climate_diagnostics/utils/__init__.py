from .dask_utils import get_or_create_dask_client
from .coord_utils import get_coord_name, filter_by_season
from .data_utils import select_process_data, get_spatial_mean

# Conditionally import chunking utilities
try:
    from .chunking_utils import (
        choose_time_chunk,
        estimate_bytes_per_timestep, 
        get_optimal_chunks,
        rechunk_dataset,
        print_chunking_info,
        suggest_chunking_strategy
    )
    _chunking_available = True
except ImportError:
    _chunking_available = False

if _chunking_available:
    __all__ = [
        'get_or_create_dask_client', 'get_coord_name', 'filter_by_season', 
        'select_process_data', 'get_spatial_mean',
        'choose_time_chunk', 'estimate_bytes_per_timestep', 'get_optimal_chunks',
        'rechunk_dataset', 'print_chunking_info', 'suggest_chunking_strategy'
    ]
else:
    __all__ = [
        'get_or_create_dask_client', 'get_coord_name', 'filter_by_season', 
        'select_process_data', 'get_spatial_mean'
    ] 