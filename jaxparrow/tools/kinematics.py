from jax import jit, lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .geometry import compute_coriolis_factor
from .operators import derivative, interpolation
from .sanitize import init_land_mask, sanitize_data
from .stencil.stencil import compute_stencil_weights, STENCIL_WIDTH


@jit
def advection(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
        stencil_weights_u: Float[Array, "2 2 lat lon stencil_width"],
        stencil_weights_v: Float[Array, "2 2 lat lon stencil_width"],
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
    stencil_weights_u : Float[Array, "2 2 lat lon stencil_width"]
        Tensor of stencil weights of every ``field`` U-point in both directions and for both padding
    stencil_weights_v : Float[Array, "2 2 lat lon stencil_width"]
        Tensor of stencil weights of every ``field`` V-point in both directions and for both padding
    mask : Float[Array, "lat lon"]
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

    Returns
    -------
    u_adv_v : Float[Array, "lat lon"]
        U component of the advection term, on the V grid
    v_adv_u : Float[Array, "lat lon"]
        V component of the advection term, on the U grid
    """
    u_adv_v = _u_advection_v(u, v, stencil_weights_v, mask)
    v_adv_u = _v_advection_u(u, v, stencil_weights_u, mask)

    return u_adv_v, v_adv_u


def _u_advection_v(
        u_u: Float[Array, "lat lon"],
        v_v: Float[Array, "lat lon"],
        stencil_weights_u: Float[Array, "2 2 lat lon stencil_width"],
        mask: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    dudx_t = derivative(u_u, stencil_weights_u, axis=1, pad_left=True)   # (U(i), U(i+1)) -> T(i+1)
    dudx_v = interpolation(dudx_t, axis=0, pad_left=False)  # (T(j), T(j+1)) -> V(j)

    dudy_f = derivative(u_u, stencil_weights_u, axis=0, pad_left=False)  # (U(j), U(j+1)) -> F(j)
    dudy_v = interpolation(dudy_f, axis=1, pad_left=True)   # (F(i), F(i+1)) -> V(i+1)

    u_t = interpolation(u_u, axis=1, pad_left=True)   # (U(i), U(i+1)) -> T(i+1)
    u_v = interpolation(u_t, axis=0, pad_left=False)  # (T(j), T(j+1)) -> V(j)

    u_adv_v = u_v * dudx_v + v_v * dudy_v  # V(j)
    u_adv_v = sanitize_data(u_adv_v, 0., mask)

    return u_adv_v


def _v_advection_u(
        u_u: Float[Array, "lat lon"],
        v_v: Float[Array, "lat lon"],
        stencil_weights_v: Float[Array, "2 2 lat lon stencil_width"],
        mask: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    dvdx_f = derivative(v_v, stencil_weights_v, axis=1, pad_left=False)  # (V(i), V(i+1)) -> F(i)
    dvdx_u = interpolation(dvdx_f, axis=0, pad_left=True)   # (F(j), F(j+1)) -> U(j+1)

    dvdy_t = derivative(v_v, stencil_weights_v, axis=0, pad_left=True)   # (V(j), V(j+1)) -> T(j+1)
    dvdy_u = interpolation(dvdy_t, axis=1, pad_left=False)  # (T(i), T(i+1)) -> U(i)

    v_t = interpolation(v_v, axis=0, pad_left=True)   # (V(j), V(j+1)) -> T(j+1)
    v_u = interpolation(v_t, axis=1, pad_left=False)  # (T(i), T(i+1)) -> U(i)

    v_adv_u = u_u * dvdx_u + v_u * dvdy_u  # U(i)
    v_adv_u = sanitize_data(v_adv_u, 0., mask)

    return v_adv_u


@jit
def magnitude(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
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
    u_t, v_t = lax.cond(
        interpolate,
        lambda: (interpolation(u, axis=1, pad_left=True),  # (U(i), U(i+1)) -> T(i+1)
                 interpolation(v, axis=0, pad_left=True)),  # (V(j), V(j+1)) -> T(j+1)
        lambda: (u, v)
    )

    magn_t = jnp.sqrt(u_t ** 2 + v_t ** 2)

    return magn_t


def normalized_relative_vorticity(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
        lat_u: Float[Array, "lat lon"],
        lon_u: Float[Array, "lat lon"],
        lat_v: Float[Array, "lat lon"],
        lon_v: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"] = None,
        interpolate: bool = True,
        stencil_width: int = STENCIL_WIDTH,
        stencil_weights_u: Float[Array, "2 2 lat lon stencil_width"] = None,
        stencil_weights_v: Float[Array, "2 2 lat lon stencil_width"] = None
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
    stencil_width : int, optional
        Width of the stencil used to compute derivatives. As we use C-grids, it should be an even integer.

        Defaults to ``STENCIL_WIDTH``
    stencil_weights_u : Float[Array, "2 2 lat lon stencil_width"], optional
        Pre-computed stencil weights. Computed if not provided.
        Computation is expensive, so pre-compute them is preferable in the case of repeated calls
        (see ``compute_stencil_weights``)
    stencil_weights_v : Float[Array, "2 2 lat lon stencil_width"], optional
        Pre-computed stencil weights. Computed if not provided.
        Computation is expensive, so pre-compute them is preferable in the case of repeated calls
        (see ``compute_stencil_weights``)

    Returns
    -------
    w : Float[Array, "lat lon"]
        The normalised relative vorticity,
        on the F grid (if ``interpolate=False``), or the T grid (if ``interpolate=True``)
    """
    # Make sure the mask is initialized
    mask = init_land_mask(u, mask)

    # Compute Coriolis factors and stencil weights
    f_u = compute_coriolis_factor(lat_u)
    if stencil_weights_u is None:
        stencil_weights_u = compute_stencil_weights(u, lat_u, lon_u, stencil_width=stencil_width)
    if stencil_weights_v is None:
        stencil_weights_v = compute_stencil_weights(v, lat_v, lon_v, stencil_width=stencil_width)

    # Handle spurious data and apply mask
    f_u = sanitize_data(f_u, jnp.nan, mask)

    # Compute the normalized relative vorticity
    du_dy_f = derivative(u, stencil_weights_u, axis=0, pad_left=False)  # (U(j), U(j+1)) -> F(j)
    dv_dx_f = derivative(v, stencil_weights_v, axis=1, pad_left=False)  # (V(i), V(i+1)) -> F(i)
    f_f = interpolation(f_u, axis=0, pad_left=False)  # (U(j), U(j+1)) -> F(j)
    w_f = (dv_dx_f - du_dy_f) / f_f  # F(j)

    if interpolate:
        w_u = interpolation(w_f, axis=0, pad_left=True)  # (F(j), F(j+1)) -> U(j+1)
        w = interpolation(w_u, axis=1, pad_left=True)  # (U(i), U(i+1)) -> T(i+1)
    else:
        w = w_f

    w = sanitize_data(w, jnp.nan, mask)

    return w


@jit
def kinetic_energy(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
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
    u_t, v_t = lax.cond(
        interpolate,
        lambda: (interpolation(u, axis=1, pad_left=True),  # (U(i), U(i+1)) -> T(i+1)
                 interpolation(v, axis=0, pad_left=True)),  # (V(j), V(j+1)) -> T(j+1)
        lambda: (u, v)
    )

    eke_t = (u_t ** 2 + v_t ** 2) / 2

    return eke_t
