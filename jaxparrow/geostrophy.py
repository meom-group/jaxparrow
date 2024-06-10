import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .tools import geometry, operators, sanitize


# =============================================================================
# Geostrophy
# =============================================================================

def geostrophy(
        ssh_t: Float[Array, "lat lon"],
        lat_t: Float[Array, "lat lon"],
        lon_t: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"] = None,
        return_grids: bool = True
) -> [Float[Array, "lat lon"], ...]:
    """
    Computes the geostrophic Sea Surface Current (SSC) velocity field from a Sea Surface Height (SSH) field.

    The geostrophic SSC velocity field is computed on a C-grid, following NEMO convention [1]_.

    Parameters
    ----------
    ssh_t : Float[Array, "lat lon"]
        SSH field (on the T grid)
    lat_t : Float[Array, "lat lon"]
        Latitudes of the T grid
    lon_t : Float[Array, "lat lon"]
        Longitudes of the T grid
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` `nan` values
    return_grids : bool, optional
        If `True`, returns the U and V grids.

        Defaults to `True`

    Returns
    -------
    u_geos_u : Float[Array, "lat lon"]
        U component of the geostrophic SSC velocity field (on the U grid)
    v_geos_v : Float[Array, "lat lon"]
        V component of the geostrophic SSC velocity field (on the V grid)
    lat_u : Float[Array, "lat lon"]
        Latitudes of the U grid, if ``return_grids=True``
    lon_u : Float[Array, "lat lon"]
        Longitudes of the U grid, if ``return_grids=True``
    lat_v : Float[Array, "lat lon"]
        Latitudes of the V grid, if ``return_grids=True``
    lon_v : Float[Array, "lat lon"]
        Longitudes of the V grid, if ``return_grids=True``
    """
    # Make sure the mask is initialized
    is_land = sanitize.init_land_mask(ssh_t, mask)

    # Compute spatial steps and Coriolis factors
    dx_t, dy_t = geometry.compute_spatial_step(lat_t, lon_t)
    coriolis_factor_t = geometry.compute_coriolis_factor(lat_t)

    # Handle spurious and masked data
    ssh_t = sanitize.sanitize_data(ssh_t, 0, is_land)

    u_geos_u, v_geos_v = _geostrophy(ssh_t, dx_t, dy_t, coriolis_factor_t, is_land)

    # Handle masked data
    u_geos_u = sanitize.sanitize_data(u_geos_u, jnp.nan, is_land)
    v_geos_v = sanitize.sanitize_data(v_geos_v, jnp.nan, is_land)

    # Compute U and V grids
    lat_u, lon_u, lat_v, lon_v = geometry.compute_uv_grids(lat_t, lon_t)

    res = (u_geos_u, v_geos_v)
    if return_grids:
        res = res + (lat_u, lon_u, lat_v, lon_v)

    return res


@jax.jit
def _geostrophy(
        ssh_t: Float[Array, "lat lon"],
        dx_t: Float[Array, "lat lon"],
        dy_t: Float[Array, "lat lon"],
        coriolis_factor_t: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"]
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    # Compute the gradient of the ssh
    ssh_dx_u = operators.derivative(ssh_t, dx_t, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)
    ssh_dy_v = operators.derivative(ssh_t, dy_t, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)

    # Interpolate the data
    ssh_dy_t = operators.interpolation(ssh_dy_v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
    ssh_dy_u = operators.interpolation(ssh_dy_t, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)

    ssh_dx_t = operators.interpolation(ssh_dx_u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
    ssh_dx_v = operators.interpolation(ssh_dx_t, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)

    coriolis_factor_u = operators.interpolation(
        coriolis_factor_t, mask, axis=1, padding="right"
    )  # (T(i), T(i+1)) -> U(i)
    coriolis_factor_v = operators.interpolation(
        coriolis_factor_t, mask, axis=0, padding="right"
    )  # (T(j), T(j+1)) -> V(j)

    # Computing the geostrophic velocities
    u_geos_u = - geometry.GRAVITY * ssh_dy_u / coriolis_factor_u  # U(i)
    v_geos_v = geometry.GRAVITY * ssh_dx_v / coriolis_factor_v  # V(j)

    return u_geos_u, v_geos_v
