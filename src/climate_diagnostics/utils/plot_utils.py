"""
Plotting-related utility functions for the climate_diagnostics package.
"""
import cartopy.crs as ccrs

def get_projection(projection_name='PlateCarree'):
    """
    Get a cartopy projection instance from its name.

    This helper function facilitates the use of different map projections
    by allowing users to specify a projection by a simple string name.
    It maps string names to `cartopy.crs` projection objects.

    If the requested projection name is not found, it defaults to
    `PlateCarree`. The comparison is case-insensitive.

    Parameters
    ----------
    projection_name : str, optional
        The name of the desired projection. Defaults to 'PlateCarree'.
        Supported projections include: 'PlateCarree', 'Robinson', 'Mercator',
        'Orthographic', 'Mollweide', 'LambertCylindrical', 'NorthPolarStereo',
        and 'SouthPolarStereo'.

    Returns
    -------
    cartopy.crs.Projection
        An instance of the specified cartopy projection.

    Examples
    --------
    >>> from climate_diagnostics.utils.plot_utils import get_projection
    >>> robinson_proj = get_projection('Robinson')
    >>> type(robinson_proj)
    <class 'cartopy.crs.Robinson'>
    """
    if not isinstance(projection_name, str):
        # Assuming it's already a projection object
        return projection_name

    projections = {
        'platecarree': ccrs.PlateCarree(),
        'robinson': ccrs.Robinson(),
        'mercator': ccrs.Mercator(),
        'orthographic': ccrs.Orthographic(),
        'mollweide': ccrs.Mollweide(),
        'lambertcylindrical': ccrs.LambertCylindrical(),
        'northpolarstereo': ccrs.NorthPolarStereo(),
        'southpolarstereo': ccrs.SouthPolarStereo(),
    }
    return projections.get(projection_name.lower(), ccrs.PlateCarree())
