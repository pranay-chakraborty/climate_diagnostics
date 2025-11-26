"""
Plotting-related utility functions for the climate_diagnostics package.
"""
import cartopy.crs as ccrs
import logging
from typing import Optional, Union, Any, Dict, Type, List

# Set up logger
logger = logging.getLogger(__name__)

# 27 Nov: Moved projection mapping to module level to avoid re-defining it on every call.
# 27 Nov: Store Classes (Type[ccrs.Projection]), not instances.
# This prevents instantiating 8 unused projection objects every time the function is called.
PROJECTION_MAP: Dict[str, Type[ccrs.Projection]] = {
    'platecarree': ccrs.PlateCarree,
    'robinson': ccrs.Robinson,
    'mercator': ccrs.Mercator,
    'orthographic': ccrs.Orthographic,
    'mollweide': ccrs.Mollweide,
    'lambertcylindrical': ccrs.LambertCylindrical,
    'northpolarstereo': ccrs.NorthPolarStereo,
    'southpolarstereo': ccrs.SouthPolarStereo,
}

def get_projection(
    projection_name: Union[str, ccrs.Projection] = 'PlateCarree', 
    **kwargs: Any
) -> ccrs.Projection:
    """
    Get a cartopy projection instance from its name, with optional configuration.

    Parameters
    ----------
    projection_name : Union[str, ccrs.Projection], optional
        The name of the desired projection (e.g., 'Robinson') or an existing 
        Cartopy projection object. Defaults to 'PlateCarree'.
    **kwargs : Any
        Keyword arguments passed to the projection constructor.
        Commonly used: `central_longitude` (e.g., 180).

    Returns
    -------
    cartopy.crs.Projection
        An instance of the specified cartopy projection.

    Raises
    ------
    ValueError
        If the projection name is unknown.

    Examples
    --------
    >>> # Get a Robinson projection centered on the Pacific
    >>> proj = get_projection('Robinson', central_longitude=180)
    """
    # 1. Passthrough if already a projection object
    if not isinstance(projection_name, str):
        if kwargs:
            logger.warning(
                "Arguments %s provided to get_projection, but 'projection_name' "
                "is already an object. Arguments will be ignored.", kwargs
            )
        return projection_name

    norm_name = projection_name.lower()

    # 2. Look up the class
    proj_class = PROJECTION_MAP.get(norm_name)

    # 3. Fail Fast if unknown (27 Nov: Fixed silent fallback to PlateCarree)
    if proj_class is None:
        valid_options = list(PROJECTION_MAP.keys())
        raise ValueError(
            f"Unknown projection '{projection_name}'. "
            f"Available options: {valid_options}"
        )

    # 4. Instantiate with arguments (27 Nov: Added kwargs support)
    try:
        return proj_class(**kwargs)
    except TypeError as e:
        # Catch cases where the specific projection doesn't accept the provided arg
        raise TypeError(
            f"Failed to instantiate projection '{projection_name}' with args {kwargs}. "
            f"Error: {e}"
        )