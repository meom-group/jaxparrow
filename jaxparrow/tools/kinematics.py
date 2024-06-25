import jax.numpy as jnp
from jaxtyping import Array, Float

from .geometry import compute_spatial_step, compute_coriolis_factor
from .operators import derivative, interpolation
from .sanitize import init_land_mask, sanitize_data


def advection(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
        dx_u: Float[Array, "lat lon"],
        dy_u: Float[Array, "lat lon"],
        dx_v: Float[Array, "lat lon"],
        dy_v: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"]
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Computes the advection terms of a 2d velocity field, on a C-grid, following NEMO convention [1]_.

    Parameters
    ----------
    u : Float[Array, "lat lon"]
        U component of the velocity field (on the U grid)
    v : Float[Array, "lat lon"]
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
    u_adv_v = _u_advection_v(u, v, dx_v, dy_v, mask)
    v_adv_u = _v_advection_u(u, v, dx_u, dy_u, mask)

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


def magnitude(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"] = None,
        interpolate: bool = True
) -> Float[Array, "lat lon"]:
    """
    Computes the magnitude (azimuthal velocity) of a 2d velocity field,
    possibly on a C-grid (following NEMO convention [1]_) if ``interpolate=True``.

    Parameters
    ----------
    u : Float[Array, "lat lon"]
        U component of the velocity field (on the U or T grid)
    v : Float[Array, "lat lon"]
        V component of the velocity field (on the V or T grid)
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``u`` `nan` values
    interpolate : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention [1]_).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.

        Defaults to `True`

    Returns
    -------
    magn_t : Float[Array, "lat lon"]
        Magnitude of the velocity field, on the T grid
    """
    # Make sure the mask is initialized
    mask = init_land_mask(u, mask)

    if interpolate:
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
        lat_u: Float[Array, "lat lon"],
        lon_u: Float[Array, "lat lon"],
        lat_v: Float[Array, "lat lon"],
        lon_v: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"] = None,
        interpolate: bool = True
) -> Float[Array, "lat lon"]:
    """
    Computes the normalised relative vorticity of a velocity field, on a C-grid, following NEMO convention [1]_.

    The ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention [1]_.
    If not, the function will return inaccurate results.

    Parameters
    ----------
    u : Float[Array, "lat lon"]
        U component of the velocity field (on the U grid)
    v : Float[Array, "lat lon"]
        V component of the velocity field (on the V grid)
    lat_u : Float[Array, "lat lon"]
        Latitudes of the U grid
    lon_u : Float[Array, "lat lon"]
        Longitudes of the U grid
    lat_v : Float[Array, "lat lon"]
        Latitudes of the V grid
    lon_v : Float[Array, "lat lon"]
        Longitudes of the V grid
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
    interpolate : bool, optional
        If `True`, the relative normalized vorticity is interpolated from the F grid to the T grid.
        If `False`, it remains on the F grid.

        Defaults to `True`

    Returns
    -------
    w : Float[Array, "lat lon"]
        The normalised relative vorticity,
        on the F grid (if ``interpolate=False``), or the T grid (if ``interpolate=True``)
    """
    # Make sure the mask is initialized
    mask = init_land_mask(u, mask)

    # Compute spatial step and Coriolis factor
    _, dy_u = compute_spatial_step(lat_u, lon_u)
    dx_v, _ = compute_spatial_step(lat_v, lon_v)
    f_u = compute_coriolis_factor(lat_u)

    # Handle spurious data and apply mask
    # dy_u = sanitize_data(dy_u, jnp.nan, mask)
    # dx_v = sanitize_data(dx_v, jnp.nan, mask)
    # f_u = sanitize_data(f_u, jnp.nan, mask)

    # Compute the normalized relative vorticity
    du_dy_f = derivative(u, dy_u, mask, axis=0, padding="right")  # (U(j), U(j+1)) -> F(j)
    dv_dx_f = derivative(v, dx_v, mask, axis=1, padding="right")  # (V(i), V(i+1)) -> F(i)
    f_f = interpolation(f_u, mask, axis=0, padding="right")  # (U(j), U(j+1)) -> F(j)
    w_f = (dv_dx_f - du_dy_f) / f_f  # F(j)

    if interpolate:
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
        interpolate: bool = True
) -> Float[Array, "lat lon"]:
    """
    Computes the Kinetic Energy (KE) of a velocity field,
    possibly on a C-grid (following NEMO convention [1]_) if ``interpolate=True``.

    Parameters
    ----------
    u : Float[Array, "lat lon"]
        U component of the velocity field (on the U grid)
    v : Float[Array, "lat lon"]
        V component of the velocity field (on the V grid)
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``u`` `nan` values
    interpolate : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids,
        and are interpolated to the T one (following NEMO convention [1]_).
        If `False`, the velocity components are assumed to be located on the T grid, and interpolation is not needed.

        Defaults to `True`

    Returns
    -------
    eke : Float[Array, "lat lon"]
        The Kinetic Energy on the T grid
    """
    # Make sure the mask is initialized
    mask = init_land_mask(u, mask)

    if interpolate:
        # interpolate to the T point
        u_t = interpolation(u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
        v_t = interpolation(v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
    else:
        u_t, v_t = u, v

    eke_t = (u_t ** 2 + v_t ** 2) / 2

    return eke_t
