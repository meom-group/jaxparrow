import jax.numpy as jnp
from jaxtyping import Array, Float

from .geometry import compute_coriolis_factor, compute_spatial_step, compute_uv_grids
from .operators import derivative, interpolation
from .sanitize import init_land_mask, sanitize_data


def radius_of_curvature(
    u: Float[Array, "lat lon"],
    v: Float[Array, "lat lon"],
    lat_t: Float[Array, "lat lon"] = None,
    lon_t: Float[Array, "lat lon"] = None,
    lat_u: Float[Array, "lat lon"] = None,
    lon_u: Float[Array, "lat lon"] = None,
    lat_v: Float[Array, "lat lon"] = None,
    lon_v: Float[Array, "lat lon"] = None,
    mask: Float[Array, "lat lon"] = None,
    vel_on_uv: bool = True
):
    """
    Computes the radius of curvature of a 2d velocity field, on a C-grid (following NEMO convention).
    The velocity field can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    u : Float[Array, "lat lon"]
        U component of the velocity field
    v : Float[Array, "lat lon"]
        V component of the velocity field
    lat_t : Float[Array, "lat lon"], optional
        Latitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lon_t : Float[Array, "lat lon"], optional
        Longitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lat_u : Float[Array, "lat lon"], optional
        Latitudes of the U grid.
        
        Defaults to `None`
    lon_u : Float[Array, "lat lon"], optional
        Longitudes of the U grid.
       
        Defaults to `None`
    lat_v : Float[Array, "lat lon"], optional
        Latitudes of the V grid.
        
        Defaults to `None`
    lon_v : Float[Array, "lat lon"], optional
        Longitudes of the V grid.
        
        Defaults to `None`
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).
        
        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, ``u`` and ``v`` are on the U and V grids.
        If `False`, they are on the T grid.
        
        Defaults to `True`
    
    Returns
    -------
    r : Float[Array, "lat lon"]
        The radius of curvature of the velocity field
    """
    if lat_u is None or lon_u is None or lat_v is None or lon_v is None:
        if lat_t is None or lon_t is None:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
        lat_u, lon_u, lat_v, lon_v = compute_uv_grids(lat_t, lon_t)

    dx_u, dy_u = compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = compute_spatial_step(lat_v, lon_v)

    return _radius_of_curvature(u, v, dx_u, dx_v, dy_u, dy_v, mask, vel_on_uv)


def _radius_of_curvature(
    u: Float[Array, "lat lon"],
    v: Float[Array, "lat lon"],
    dx_u: Float[Array, "lat lon"],
    dx_v: Float[Array, "lat lon"],
    dy_u: Float[Array, "lat lon"],
    dy_v: Float[Array, "lat lon"],
    mask: Float[Array, "lat lon"],
    vel_on_uv: bool
) -> Float[Array, "lat lon"]:
    if not vel_on_uv:
        u_t = u
        v_t = v
        u_u = interpolation(u, mask, axis=1, padding="right")
        v_v = interpolation(v, mask, axis=0, padding="right")
    else:
        u_t = interpolation(u, mask, axis=1, padding="left")
        v_t = interpolation(v, mask, axis=0, padding="left")
        u_u = u
        v_v = v
    
    V_t = magnitude(u_t, v_t, vel_on_uv=False)

    du_dx_t = derivative(u_u, dx_u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
    du_dy_f = derivative(u_u, dy_u, mask, axis=0, padding="right")  # (U(j), U(j+1)) -> F(j)

    dv_dx_f = derivative(v_v, dx_v, mask, axis=1, padding="right")  # (V(i), V(i+1)) -> F(i)
    dv_dy_t = derivative(v_v, dy_v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)

    du_dy_v = interpolation(du_dy_f, mask, axis=1, padding="left")  # (F(i), F(i+1)) -> V(i+1)
    du_dy_t = interpolation(du_dy_v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
    dv_dx_u = interpolation(dv_dx_f, mask, axis=0, padding="left")  # (F(j), F(j+1)) -> U(j+1)
    dv_dx_t = interpolation(dv_dx_u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)

    numerator = V_t ** 3
    denominator = u_t ** 2 * dv_dx_t - v_t ** 2 * du_dy_t - u_t * v_t * (du_dx_t - dv_dy_t)
    r = numerator / denominator

    return r


def _advection(
    u_u: Float[Array, "lat lon"],
    v_v: Float[Array, "lat lon"],
    dx_u: Float[Array, "lat lon"],
    dx_v: Float[Array, "lat lon"],
    dy_u: Float[Array, "lat lon"],
    dy_v: Float[Array, "lat lon"],
    mask: Float[Array, "lat lon"]
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Computes the advection terms of a 2d velocity field, on a C-grid, following NEMO convention.

    Parameters
    ----------
    u_u : Float[Array, "lat lon"]
        U component of the velocity field (on the U grid)
    v_v : Float[Array, "lat lon"]
        V component of the SSC velocity field (on the V grid)
    dx_u : Float[Array, "lat lon"]
        Spatial steps on the U grid along `x`, in meters
    dy_u : Float[Array, "lat lon"]
        Spatial steps on the U grid along `y`, in meters
    dx_v : Float[Array, "lat lon"]
        Spatial steps on the V grid along `x`, in meters
    dy_v : Float[Array, "lat lon"]
        Spatial steps on the V grid along `y`, in meters
    mask : Float[Array, "lat lon"]
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

    Returns
    -------
    u_adv_v : Float[Array, "lat lon"]
        U component of the advection term, on the V grid
    v_adv_u : Float[Array, "lat lon"]
        V component of the advection term, on the U grid
    """
    u_adv_v = _u_advection_v(u_u, v_v, dx_v, dy_v, mask)
    v_adv_u = _v_advection_u(u_u, v_v, dx_u, dy_u, mask)

    return u_adv_v, v_adv_u


def _u_advection_v(
    u_u: Float[Array, "lat lon"],
    v_v: Float[Array, "lat lon"],
    dx_u: Float[Array, "lat lon"],
    dy_u: Float[Array, "lat lon"],
    mask: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    dudx_t = derivative(u_u, dx_u, mask, axis=1, padding="left")   # (U(i), U(i+1)) -> T(i+1)
    dudx_v = interpolation(dudx_t, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)

    dudy_f = derivative(u_u, dy_u, mask, axis=0, padding="right")  # (U(j), U(j+1)) -> F(j)
    dudy_v = interpolation(dudy_f, mask, axis=1, padding="left")   # (F(i), F(i+1)) -> V(i+1)

    u_t = interpolation(u_u, mask, axis=1, padding="left")   # (U(i), U(i+1)) -> T(i+1)
    u_v = interpolation(u_t, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)

    u_adv_v = u_v * dudx_v + v_v * dudy_v  # V(j)

    return u_adv_v


def _v_advection_u(
    u_u: Float[Array, "lat lon"],
    v_v: Float[Array, "lat lon"],
    dx_v: Float[Array, "lat lon"],
    dy_v: Float[Array, "lat lon"],
    mask: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    dvdx_f = derivative(v_v, dx_v, mask, axis=1, padding="right")  # (V(i), V(i+1)) -> F(i)
    dvdx_u = interpolation(dvdx_f, mask, axis=0, padding="left")   # (F(j), F(j+1)) -> U(j+1)

    dvdy_t = derivative(v_v, dy_v, mask, axis=0, padding="left")   # (V(j), V(j+1)) -> T(j+1)
    dvdy_u = interpolation(dvdy_t, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)

    v_t = interpolation(v_v, mask, axis=0, padding="left")   # (V(j), V(j+1)) -> T(j+1)
    v_u = interpolation(v_t, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)

    v_adv_u = u_u * dvdx_u + v_u * dvdy_u  # U(i)

    return v_adv_u


def cyclogeostrophic_imbalance(
    u_geos: Float[Array, "lat lon"],
    v_geos: Float[Array, "lat lon"],
    u_cyclo: Float[Array, "lat lon"],
    v_cyclo: Float[Array, "lat lon"],
    lat_t: Float[Array, "lat lon"] = None,
    lon_t: Float[Array, "lat lon"] = None,
    lat_u: Float[Array, "lat lon"] = None,
    lon_u: Float[Array, "lat lon"] = None,
    lat_v: Float[Array, "lat lon"] = None,
    lon_v: Float[Array, "lat lon"] = None,
    mask: Float[Array, "lat lon"] = None,
    vel_on_uv: bool = True
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Computes the cyclogeostrophic imbalance of a 2d velocity field, on a C-grid (following NEMO convention).
    The velocity fields can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    u_geos : Float[Array, "lat lon"]
        U component of the geostrophic velocity field
    v_geos : Float[Array, "lat lon"]
        V component of the geostrophic velocity field
    u_cyclo : Float[Array, "lat lon"]
        U component of the cyclogeostrophic velocity field
    v_cyclo : Float[Array, "lat lon"]
        V component of the cyclogeostrophic velocity field
    lat_t : Float[Array, "lat lon"], optional
        Latitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lon_t : Float[Array, "lat lon"], optional
        Longitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lat_u : Float[Array, "lat lon"], optional
        Latitudes of the U grid.
        
        Defaults to `None`
    lon_u : Float[Array, "lat lon"], optional
        Longitudes of the U grid.
        
        Defaults to `None`
    lat_v : Float[Array, "lat lon"], optional
        Latitudes of the V grid.
        
        Defaults to `None`
    lon_v : Float[Array, "lat lon"], optional
        Longitudes of the V grid.
        
        Defaults to `None`
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).

        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.
        
        Defaults to `True`

    Returns
    -------
    u_imbalance_u : Float[Array, "lat lon"]
        U component of the cyclogeostrophic imbalance, on the U grid
    v_imbalance_v : Float[Array, "lat lon"]
        V component of the cyclogeostrophic imbalance, on the V grid
    """
    if not vel_on_uv:
        u_geos_u = interpolation(u_geos, mask, axis=1, padding="right")
        v_geos_v = interpolation(v_geos, mask, axis=0, padding="right")
        u_cyclo_u = interpolation(u_cyclo, mask, axis=1, padding="right")
        v_cyclo_v = interpolation(v_cyclo, mask, axis=0, padding="right")
    else:
        u_geos_u = u_geos
        v_geos_v = v_geos
        u_cyclo_u = u_cyclo
        v_cyclo_v = v_cyclo

    if lat_u is None or lon_u is None or lat_v is None or lon_v is None:
        if lat_t is None or lon_t is None:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
        lat_u, lon_u, lat_v, lon_v = compute_uv_grids(lat_t, lon_t)

    dx_u, dy_u = compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = compute_spatial_step(lat_v, lon_v)
    coriolis_factor_u = compute_coriolis_factor(lat_u)
    coriolis_factor_v = compute_coriolis_factor(lat_v)

    return _cyclogeostrophic_imbalance(
        u_geos_u, v_geos_v, u_cyclo_u, v_cyclo_v, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, mask
    )


def _cyclogeostrophic_imbalance(
    u_geos_u: Float[Array, "lat lon"],
    v_geos_v: Float[Array, "lat lon"],
    u_cyclo_u: Float[Array, "lat lon"],
    v_cyclo_v: Float[Array, "lat lon"],
    dx_u: Float[Array, "lat lon"],
    dx_v: Float[Array, "lat lon"],
    dy_u: Float[Array, "lat lon"],
    dy_v: Float[Array, "lat lon"],
    coriolis_factor_u: Float[Array, "lat lon"],
    coriolis_factor_v: Float[Array, "lat lon"],
    mask: Float[Array, "lat lon"]
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    u_adv_v, v_adv_u = _advection(u_cyclo_u, v_cyclo_v, dx_u, dx_v, dy_u, dy_v, mask)

    u_imbalance_u = u_cyclo_u + v_adv_u / coriolis_factor_u - u_geos_u
    v_imbalance_v = v_cyclo_v - u_adv_v / coriolis_factor_v - v_geos_v

    return u_imbalance_u, v_imbalance_v


def magnitude(
    u: Float[Array, "lat lon"],
    v: Float[Array, "lat lon"],
    mask: Float[Array, "lat lon"] = None,
    vel_on_uv: bool = True
) -> Float[Array, "lat lon"]:
    """
    Computes the magnitude (azimuthal velocity) of a 2d velocity field, on a C-grid, following NEMO convention.
    The velocity field can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    u : Float[Array, "lat lon"]
        U component of the velocity field (on the U or T grid)
    v : Float[Array, "lat lon"]
        V component of the velocity field (on the V or T grid)
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.
        
        Defaults to `True`

    Returns
    -------
    magn_t : Float[Array, "lat lon"]
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
    u: Float[Array, "lat lon"],
    v: Float[Array, "lat lon"],
    lat_t: Float[Array, "lat lon"] = None,
    lon_t: Float[Array, "lat lon"] = None,
    lat_u: Float[Array, "lat lon"] = None,
    lon_u: Float[Array, "lat lon"] = None,
    lat_v: Float[Array, "lat lon"] = None,
    lon_v: Float[Array, "lat lon"] = None,
    mask: Float[Array, "lat lon"] = None,
    vel_on_uv: bool = True,
    nrv_on_t: bool = True
) -> Float[Array, "lat lon"]:
    """
    Computes the normalised relative vorticity of a velocity field, on a C-grid, following NEMO convention.
    The velocity field can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    If provided, the ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention.
    They can be left empty and will be automatically computed if ``lat_t`` and ``lon_t`` are provided.

    Parameters
    ----------
    u : Float[Array, "lat lon"]
        U component of the velocity field
    v : Float[Array, "lat lon"]
        V component of the velocity field
    lat_t : Float[Array, "lat lon"], optional
        Latitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lon_t : Float[Array, "lat lon"], optional
        Longitudes of the T grid.
       
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lat_u : Float[Array, "lat lon"], optional
        Latitudes of the U grid.
        
        Defaults to `None`
    lon_u : Float[Array, "lat lon"], optional
        Longitudes of the U grid.
        
        Defaults to `None`
    lat_v : Float[Array, "lat lon"], optional
        Latitudes of the V grid.
        
        Defaults to `None`
    lon_v : Float[Array, "lat lon"], optional
        Longitudes of the V grid.
        
        Defaults to `None`
    mask : Float[Array, "lat lon"], optional
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
    w : Float[Array, "lat lon"]
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
    w_f = (dv_dx_f - du_dy_f) / f_f  # F(j)

    if nrv_on_t:
        w_u = interpolation(w_f, mask, axis=0, padding="left")  # (F(j), F(j+1)) -> U(j+1)
        w = interpolation(w_u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
    else:
        w = w_f

    w = sanitize_data(w, jnp.nan, mask)

    return w


def kinetic_energy(
    u: Float[Array, "lat lon"],
    v: Float[Array, "lat lon"],
    mask: Float[Array, "lat lon"] = None,
    vel_on_uv: bool = True
) -> Float[Array, "lat lon"]:
    """
    Computes the Kinetic Energy (KE) of a velocity field, on a C-grid, following NEMO convention.
    The velocity field can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    u : Float[Array, "lat lon"]
        U component of the velocity field (on the U grid)
    v : Float[Array, "lat lon"]
        V component of the velocity field (on the V grid)
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).

        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.

        Defaults to `True`

    Returns
    -------
    eke : Float[Array, "lat lon"]
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

    eke_t = (u_t ** 2 + v_t ** 2) / 2

    return eke_t
