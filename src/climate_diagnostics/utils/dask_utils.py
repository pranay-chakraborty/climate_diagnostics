from dask.distributed import Client

def get_or_create_dask_client():
    """
    Get the existing Dask client or create a new one if none exists.

    This function checks if a Dask client is currently running. If not, it
    initializes a new `dask.distributed.Client` with default settings,
    allowing Dask to manage the scheduling environment.

    Returns
    -------
    dask.distributed.Client
        The active Dask client.
    """
    try:
        # get_client() will raise ValueError if no client is running
        from dask.distributed import get_client
        client = get_client()
    except (ValueError, ImportError):
        # No client exists, or we are in a minimal Dask environment.
        # Create a new one.
        client = Client()
    return client 