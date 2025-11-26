import logging
from contextlib import contextmanager
from typing import Iterator, Any, Dict, Optional # 27 Nov: Added type hints
from dask.distributed import Client, get_client

# Set up a basic logger for the module
logger = logging.getLogger(__name__)

def _display_dashboard_info(client: Client) -> None: # 27 Nov: Helper for dashboard visibility
    """
    Safely extract and display the Dask Dashboard link.
    """
    try:
        # dashboard_link can sometimes be None if the dashboard is not running
        link = getattr(client, 'dashboard_link', None)
        if link:
            logger.info(f"Dask Dashboard available at: {link}")
            # 27 Nov: Explicit print for interactive sessions (Jupyter/IPython)
            # Users often miss log messages, but they need this link immediately.
            print(f"Dask Dashboard: {link}")
        else:
            logger.info("Dask client active, but dashboard link is not available.")
    except Exception:
        # Fallback to prevent crashing just because printing a link failed
        logger.debug("Could not retrieve dashboard link.")


@contextmanager
def managed_dask_client(**kwargs: Any) -> Iterator[Client]: # 27 Nov: Added Return Type Hint
    """
    A context manager to get an existing Dask client or create/manage a new one.

    This function provides robust resource management for Dask clients in library code.
    If a client already exists, it yields that client without closing it on exit.
    If no client exists, it creates a new one, yields it, and then cleanly
    shuts it down upon exiting the context.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments for the `dask.distributed.Client` constructor.

    Yields
    ------
    dask.distributed.Client
        The active Dask client.
    """
    client_was_created = False
    client: Optional[Client] = None

    try:
        # Attempt to get the currently active client
        client = get_client()
        logger.info("Using existing Dask client: %s", client)
    except ValueError: 
        # 27 Nov: Removed ImportError (it won't happen here if import succeeds at top)
        # ValueError is raised by get_client() when no global client exists.
        logger.info("Creating and managing a new Dask client with kwargs: %s", kwargs)
        try:
            client = Client(**kwargs)
            client_was_created = True
            logger.info("Successfully created new Dask client: %s", client)
        except Exception as e:
            logger.error("Failed to create Dask client: %s", e)
            raise

    # 27 Nov: Display dashboard link immediately after client acquisition
    if client:
        _display_dashboard_info(client)

    try:
        yield client
    finally:
        if client_was_created and client:
            logger.info("Context manager closing the Dask client it created.")
            try:
                client.close()
            except Exception as e:
                logger.warning("Error closing Dask client: %s", e)
        else:
            logger.info("Context manager leaving existing Dask client running.")


def get_or_create_dask_client(**kwargs: Any) -> Client: # 27 Nov: Added Return Type Hint
    """
    Get an active Dask client or create a new one with specified settings.

    This function provides a centralized way to manage Dask client connections.
    It reuses an existing client if available or creates a new one.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments to be passed to the `dask.distributed.Client` constructor.

    Returns
    -------
    dask.distributed.Client
        The active Dask client.
    """
    try:
        # Attempt to get the currently active client
        client = get_client()
        logger.info("Found existing Dask client: %s", client)
    except ValueError:
        # 27 Nov: Cleaned up exception handling
        logger.info("No active Dask client found. Creating a new one with kwargs: %s", kwargs)
        try:
            client = Client(**kwargs)
            logger.info("Successfully created new Dask client: %s", client)
        except Exception as e:
            logger.error("Failed to create Dask client: %s", e)
            raise
    
    # 27 Nov: Ensure the user sees the dashboard link
    _display_dashboard_info(client)
    
    return client