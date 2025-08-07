import logging
from contextlib import contextmanager
from dask.distributed import Client, get_client

# Set up a basic logger for the module
logger = logging.getLogger(__name__)

@contextmanager
def managed_dask_client(**kwargs):
    """
    A context manager to get an existing Dask client or create/manage a new one.

    This function provides robust resource management for Dask clients in library code.
    If a client already exists, it yields that client without closing it on exit.
    If no client exists, it creates a new one, yields it, and then cleanly
    shuts it down upon exiting the context.

    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments for the `dask.distributed.Client` constructor,
        used only if a new client is created. Common options include:
        - n_workers: Number of worker processes
        - threads_per_worker: Number of threads per worker
        - memory_limit: Memory limit per worker (e.g., '4GB')
        - processes: Whether to use processes (True) or threads (False)

    Yields
    ------
    dask.distributed.Client
        The active Dask client.
        
    Notes
    -----
    This context manager ensures proper resource cleanup:
    1. If it creates a new client, it will close it when exiting the context
    2. If it finds an existing client, it leaves it running (caller's responsibility)
    3. Provides logging feedback about client creation/reuse
    4. Handles graceful fallback for minimal Dask environments
    
    Examples
    --------
    >>> # Create and manage a new client
    >>> with managed_dask_client(n_workers=2, memory_limit='2GB') as client:
    ...     print(f"Doing work with client: {client}")
    ...     result = client.compute(some_dask_array)
    ... # Client is automatically shut down here

    >>> # Use with existing client
    >>> existing_client = Client()
    >>> with managed_dask_client() as client:
    ...     assert client is existing_client
    ...     print("Using existing client")
    ... # Existing client is NOT shut down here
    """
    client_was_created = False
    try:
        # Attempt to get the currently active client
        client = get_client()
        logger.info("Using existing Dask client: %s", client)
    except (ValueError, ImportError):
        # No client exists or minimal Dask environment detected
        logger.info("Creating and managing a new Dask client with kwargs: %s", kwargs)
        try:
            client = Client(**kwargs)
            client_was_created = True
            logger.info("Successfully created new Dask client: %s", client)
        except Exception as e:
            logger.error("Failed to create Dask client: %s", e)
            raise

    try:
        yield client
    finally:
        if client_was_created:
            logger.info("Context manager closing the Dask client it created.")
            try:
                client.close()
            except Exception as e:
                logger.warning("Error closing Dask client: %s", e)
        else:
            logger.info("Context manager leaving existing Dask client running.")


def get_or_create_dask_client(**kwargs):
    """
    Get an active Dask client or create a new one with specified settings.

    This function provides a centralized way to manage Dask client connections.
    It reuses an existing client if available or creates a new one, passing
    any keyword arguments to the Client constructor.

    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments to be passed to the `dask.distributed.Client`
        constructor if a new client is created. For example, `n_workers`,
        `threads_per_worker`, `memory_limit`.

    Returns
    -------
    dask.distributed.Client
        The active Dask client for distributed computing tasks.
        
    Notes
    -----
    This function will:
    1. Check for an existing active Dask client.
    2. Return it if found, ignoring any provided kwargs.
    3. Create a new client using the provided kwargs if none exists.
    
    Examples
    --------
    >>> # Get or create a default client
    >>> client = get_or_create_dask_client()
    >>> print(f"Client dashboard: {client.dashboard_link}")

    >>> # Get or create a client with specific configuration
    >>> client_config = {'n_workers': 4, 'threads_per_worker': 2, 'memory_limit': '4GB'}
    >>> specific_client = get_or_create_dask_client(**client_config)
    >>> print(f"Specific client has {len(specific_client.scheduler_info()['workers'])} workers.")
    """
    try:
        # Attempt to get the currently active client
        client = get_client()
        logger.info("Found existing Dask client: %s", client)
    except (ValueError, ImportError):
        # No client exists or minimal Dask environment detected
        logger.info("No active Dask client found. Creating a new one with kwargs: %s", kwargs)
        try:
            client = Client(**kwargs)
            logger.info("Successfully created new Dask client: %s", client)
        except Exception as e:
            logger.error("Failed to create Dask client: %s", e)
            raise
    
    return client


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Using the context manager (creates and closes) ---")
    with managed_dask_client(n_workers=2) as c:
        print(f"Inside context 1: {c}")
        print(f"Dashboard: {c.dashboard_link}")
        # do work...
    print("Outside context 1. Client should be closed.")

    print("\n--- Using the context manager with a pre-existing client ---")
    main_client = get_or_create_dask_client(n_workers=4)
    print(f"Created a main client: {main_client}")
    with managed_dask_client() as c:
        print(f"Inside context 2: {c}")
        assert c is main_client
    print("Outside context 2. Main client should still be running.")
    print(f"Main client status: {main_client.status}")
    main_client.close() 