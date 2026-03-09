from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float

from .utils import geometry, operators, sanitize


class GeostrophyResult(NamedTuple):
    """
    Result of geostrophic velocity computation.

    This NamedTuple provides named access to results, avoiding positional unpacking errors.
    All fields except ``ug`` and ``vg`` are optional and depend on the
    ``return_grids`` flag passed to the computation function.

    Attributes
    ----------
    ug : Float[jax.Array, "lat lon"]
        U component of geostrophic velocity on U grid
    vg : Float[jax.Array, "lat lon"]
        V component of geostrophic velocity on V grid
    lat_u : Float[jax.Array, "lat lon"] | None
        Latitudes of U grid (if ``return_grids=True``)
    lon_u : Float[jax.Array, "lat lon"] | None
        Longitudes of U grid (if ``return_grids=True``)
    lat_v : Float[jax.Array, "lat lon"] | None
        Latitudes of V grid (if ``return_grids=True``)
    lon_v : Float[jax.Array, "lat lon"] | None
        Longitudes of V grid (if ``return_grids=True``)
    """

    ug: Float[jax.Array, "lat lon"]
    vg: Float[jax.Array, "lat lon"]
    lat_u: Float[jax.Array, "lat lon"] | None = None
    lon_u: Float[jax.Array, "lat lon"] | None = None
    lat_v: Float[jax.Array, "lat lon"] | None = None
    lon_v: Float[jax.Array, "lat lon"] | None = None


# =============================================================================
# Geostrophy
# =============================================================================

def geostrophy(
    ssh_t: Float[jax.Array, "lat lon"],
    lat_t: Float[jax.Array, "lat lon"],
    lon_t: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"] = None,
    return_grids: bool = True
) -> GeostrophyResult:
    """
    Computes the geostrophic Sea Surface Current (SSC) velocity field from a Sea Surface Height (SSH) field.

    The geostrophic SSC velocity field is computed on a C-grid, following NEMO convention.
    For curvilinear grids where grid axes are not aligned with geographic east-west/north-south,
    the function properly rotates gradients from grid coordinates to geographic coordinates.

    Parameters
    ----------
    ssh_t : Float[jax.Array, "lat lon"]
        SSH field (on the T grid)
    lat_t : Float[jax.Array, "lat lon"]
        Latitudes of the T grid
    lon_t : Float[jax.Array, "lat lon"]
        Longitudes of the T grid
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` `nan` values
    return_grids : bool, optional
        If `True`, returns the U and V grids.

        Defaults to `True`

    Returns
    -------
    GeostrophyResult
        A named tuple containing:

        - **ug** : U component of the geostrophic SSC velocity field (on the U grid)
        - **vg** : V component of the geostrophic SSC velocity field (on the V grid)
        - **lat_u** : Latitudes of the U grid (if ``return_grids=True``, else ``None``)
        - **lon_u** : Longitudes of the U grid (if ``return_grids=True``, else ``None``)
        - **lat_v** : Latitudes of the V grid (if ``return_grids=True``, else ``None``)
        - **lon_v** : Longitudes of the V grid (if ``return_grids=True``, else ``None``)
    """
    # Make sure the mask is initialized
    is_land = sanitize.init_land_mask(ssh_t, mask)

    # Compute spatial steps and Coriolis factors
    dx_t, dy_t = geometry.compute_spatial_step(lat_t, lon_t)
    coriolis_factor_t = geometry.compute_coriolis_factor(lat_t)

    # Compute grid rotation angle for curvilinear grids
    grid_angle_t = geometry.compute_grid_angle(lat_t, lon_t)

    # Handle spurious and masked data
    ssh_t = sanitize.sanitize_data(ssh_t, jnp.nan, is_land)

    ug_u, vg_v = _geostrophy(ssh_t, dx_t, dy_t, coriolis_factor_t, grid_angle_t, is_land)

    # Handle masked data (set land cells to NaN)
    ug_u = sanitize.sanitize_data(ug_u, jnp.nan, is_land)
    vg_v = sanitize.sanitize_data(vg_v, jnp.nan, is_land)

    lat_u, lon_u, lat_v, lon_v = None, None, None, None
    if return_grids:
        # Compute U and V grids
        lat_u, lon_u, lat_v, lon_v = geometry.compute_uv_grids(lat_t, lon_t)

    return GeostrophyResult(
        ug=ug_u,
        vg=vg_v,
        lat_u=lat_u,
        lon_u=lon_u,
        lat_v=lat_v,
        lon_v=lon_v
    )


@jax.jit
def _geostrophy(
    ssh_t: Float[jax.Array, "lat lon"],
    dx_t: Float[jax.Array, "lat lon"],
    dy_t: Float[jax.Array, "lat lon"],
    coriolis_factor_t: Float[jax.Array, "lat lon"],
    grid_angle_t: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"]
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]]:
    """
    Internal JIT-compiled geostrophy computation with curvilinear grid support.

    Computes gradients in grid coordinates (i, j axes), then rotates them to
    geographic coordinates (x=east, y=north) before computing geostrophic velocities.
    """
    # Compute the gradient of the ssh in grid coordinates
    # deta_di: gradient along grid i-axis (axis=1)
    # deta_dj: gradient along grid j-axis (axis=0)
    deta_di_u = operators.derivative(ssh_t, dx_t, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)
    deta_dj_v = operators.derivative(ssh_t, dy_t, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)

    # Interpolate grid-coordinate gradients to have both components at both U and V points
    deta_dj_t = operators.interpolation(deta_dj_v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
    deta_dj_u = operators.interpolation(deta_dj_t, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)

    deta_di_t = operators.interpolation(deta_di_u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
    deta_di_v = operators.interpolation(deta_di_t, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)

    # Interpolate grid angle to U and V points
    grid_angle_u = operators.interpolation(grid_angle_t, mask, axis=1, padding="right")
    grid_angle_v = operators.interpolation(grid_angle_t, mask, axis=0, padding="right")

    # Rotate gradients from grid coordinates to geographic coordinates
    _, deta_dy_u = operators.rotate_to_geographic(deta_di_u, deta_dj_u, grid_angle_u)
    deta_dx_v, _ = operators.rotate_to_geographic(deta_di_v, deta_dj_v, grid_angle_v)

    # Interpolate Coriolis factor to U and V points
    coriolis_factor_u = operators.interpolation(
        coriolis_factor_t, mask, axis=1, padding="right"
    )  # (T(i), T(i+1)) -> U(i)
    coriolis_factor_v = operators.interpolation(
        coriolis_factor_t, mask, axis=0, padding="right"
    )  # (T(j), T(j+1)) -> V(j)

    # Computing the geostrophic velocities (in geographic coordinates)
    # u = -g/f * dη/dy  (eastward velocity)
    # v =  g/f * dη/dx  (northward velocity)
    ug_u = -geometry.GRAVITY * deta_dy_u / coriolis_factor_u  # U(i)
    vg_v = geometry.GRAVITY * deta_dx_v / coriolis_factor_v   # V(j)

    return ug_u, vg_v
