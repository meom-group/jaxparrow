import jax
import jax.numpy as jnp
from jaxtyping import Float

from .geometry import compute_coriolis_factor, compute_spatial_step, compute_uv_grids
from .operators import derivative, interpolation
from .sanitize import init_land_mask, sanitize_data


def magnitude(
    u: Float[jax.Array, "lat lon"],
    v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"] = None,
    vel_on_uv: bool = True
) -> Float[jax.Array, "lat lon"]:
    """
    Computes the magnitude (azimuthal velocity) of a 2d velocity field, on a C-grid, following NEMO convention.
    The velocity field can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    u : Float[jax.Array, "lat lon"]
        U component of the velocity field (on the U or T grid)
    v : Float[jax.Array, "lat lon"]
        V component of the velocity field (on the V or T grid)
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.
        
        Defaults to `True`

    Returns
    -------
    magn_t : Float[jax.Array, "lat lon"]
        Magnitude of the velocity field, on the T grid
    """
    # Make sure the mask is initialized
    mask = init_land_mask(u, mask)

    if vel_on_uv:
        # interpolate to the T point
        u_t = interpolation(u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
        v_t = interpolation(v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
    else:
        u_t, v_t = u, v

    magn_t = jnp.sqrt(u_t ** 2 + v_t ** 2)
    magn_t = sanitize_data(magn_t, jnp.nan, mask)

    return magn_t


def normalized_relative_vorticity(
    u: Float[jax.Array, "lat lon"],
    v: Float[jax.Array, "lat lon"],
    lat_t: Float[jax.Array, "lat lon"] = None,
    lon_t: Float[jax.Array, "lat lon"] = None,
    lat_u: Float[jax.Array, "lat lon"] = None,
    lon_u: Float[jax.Array, "lat lon"] = None,
    lat_v: Float[jax.Array, "lat lon"] = None,
    lon_v: Float[jax.Array, "lat lon"] = None,
    mask: Float[jax.Array, "lat lon"] = None,
    vel_on_uv: bool = True,
    nrv_on_t: bool = True
) -> Float[jax.Array, "lat lon"]:
    """
    Computes the normalised relative vorticity of a velocity field, on a C-grid, following NEMO convention.
    The velocity field can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    If provided, the ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention.
    They can be left empty and will be automatically computed if ``lat_t`` and ``lon_t`` are provided.

    Parameters
    ----------
    u : Float[jax.Array, "lat lon"]
        U component of the velocity field
    v : Float[jax.Array, "lat lon"]
        V component of the velocity field
    lat_t : Float[jax.Array, "lat lon"], optional
        Latitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lon_t : Float[jax.Array, "lat lon"], optional
        Longitudes of the T grid.
       
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lat_u : Float[jax.Array, "lat lon"], optional
        Latitudes of the U grid.
        
        Defaults to `None`
    lon_u : Float[jax.Array, "lat lon"], optional
        Longitudes of the U grid.
        
        Defaults to `None`
    lat_v : Float[jax.Array, "lat lon"], optional
        Latitudes of the V grid.
        
        Defaults to `None`
    lon_v : Float[jax.Array, "lat lon"], optional
        Longitudes of the V grid.
        
        Defaults to `None`
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
    vel_on_uv : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.
        
        Defaults to `True`
    nrv_on_t : bool, optional
        If `True`, the normalised relative vorticity is returned on the T grid (following NEMO convention).
        If `False`, it is returned on the F grid.
        
        Defaults to `True`

    Returns
    -------
    nrv : Float[jax.Array, "lat lon"]
        The normalised relative vorticity,
        on the F grid (if ``interpolate=False``), or the T grid (if ``interpolate=True``)
    """
    # Make sure the mask is initialized
    mask = init_land_mask(u, mask)

    if not vel_on_uv:
        # interpolate to the U and V points
        u_u = interpolation(u, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)
        v_v = interpolation(v, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)
    else:
        u_u, v_v = u, v

    if lat_u is None or lon_u is None or lat_v is None or lon_v is None:
        if lat_t is None or lon_t is None:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
        lat_u, lon_u, lat_v, lon_v = compute_uv_grids(lat_t, lon_t)

    # Compute spatial step and Coriolis factor
    _, dy_u = compute_spatial_step(lat_u, lon_u)
    dx_v, _ = compute_spatial_step(lat_v, lon_v)
    f_u = compute_coriolis_factor(lat_u)

    # Compute the normalized relative vorticity
    du_dy_f = derivative(u_u, dy_u, mask, axis=0, padding="right")  # (U(j), U(j+1)) -> F(j)
    dv_dx_f = derivative(v_v, dx_v, mask, axis=1, padding="right")  # (V(i), V(i+1)) -> F(i)
    f_f = interpolation(f_u, mask, axis=0, padding="right")  # (U(j), U(j+1)) -> F(j)
    nrv_f = (dv_dx_f - du_dy_f) / f_f  # F(j)

    if nrv_on_t:
        nrv_u = interpolation(nrv_f, mask, axis=0, padding="left")  # (F(j), F(j+1)) -> U(j+1)
        nrv = interpolation(nrv_u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
    else:
        nrv = nrv_f

    nrv = sanitize_data(nrv, jnp.nan, mask)

    return nrv


def kinetic_energy(
    u: Float[jax.Array, "lat lon"],
    v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"] = None,
    vel_on_uv: bool = True
) -> Float[jax.Array, "lat lon"]:
    """
    Computes the Kinetic Energy (KE) of a velocity field, on a C-grid, following NEMO convention.
    The velocity field can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    u : Float[jax.Array, "lat lon"]
        U component of the velocity field (on the U grid)
    v : Float[jax.Array, "lat lon"]
        V component of the velocity field (on the V grid)
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).

        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.

        Defaults to `True`

    Returns
    -------
    ke : Float[jax.Array, "lat lon"]
        The Kinetic Energy on the T grid
    """
    # Make sure the mask is initialized
    mask = init_land_mask(u, mask)

    if vel_on_uv:
        # interpolate to the T point
        u_t = interpolation(u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
        v_t = interpolation(v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
    else:
        u_t, v_t = u, v

    ke_t = (u_t ** 2 + v_t ** 2) / 2

    return ke_t
