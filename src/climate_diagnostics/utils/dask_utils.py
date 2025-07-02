from dask.distributed import Client

def get_or_create_dask_client():
    """
    Get the active Dask client or create a new one if none exists.

    This function provides a centralized way to manage Dask client connections
    for distributed computing. It follows the pattern of reusing existing
    clients when available to avoid resource conflicts and connection overhead.

    Returns
    -------
    dask.distributed.Client
        The active Dask client for distributed computing tasks.
        
    Notes
    -----
    This function will:
    1. Check for an existing active Dask client
    2. Return it if found (avoids duplicate connections)
    3. Create a new LocalCluster client if none exists
    4. Handle graceful fallback for minimal Dask environments
    
    Examples
    --------
    >>> client = get_or_create_dask_client()
    >>> print(f"Dask client dashboard: {client.dashboard_link}")
    """
    try:
        # Attempt to get the currently active client to avoid duplicates
        from dask.distributed import get_client
        client = get_client()
    except (ValueError, ImportError):
        # No client exists or minimal Dask environment detected
        # Create a new client with LocalCluster (default configuration)
        client = Client()
    return client 