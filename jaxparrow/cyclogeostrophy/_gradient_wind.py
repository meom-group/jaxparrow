import jax
import jax.numpy as jnp
from jaxtyping import Float

from ._core import (
    CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result, _radius_of_curvature
)
from ..utils import kinematics, operators


def gradient_wind(
    ssh_t: Float[jax.Array, "lat lon"],
    lat_t: Float[jax.Array, "lat lon"],
    lon_t: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"] = None,
    return_geos: bool = False,
    return_grids: bool = True
) -> CyclogeostrophyResult:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field from a Sea Surface Height (SSH) field
    using the gradient wind approximation.

    The cyclogeostrophic SSC velocity field is computed on a C-grid, following NEMO convention.

    Parameters
    ----------
    ssh_t : Float[jax.Array, "lat lon"]
        SSH field (on the T grid)
    lat_t : Float[jax.Array, "lat lon"]
        Latitude of the T grid
    lon_t : Float[jax.Array, "lat lon"]
        Longitude of the T grid
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` `nan` values

        Defaults to `None`
    return_geos : bool, optional
        If `True`, returns the geostrophic SSC velocity field in addition to the cyclogeostrophic one.

        Defaults to `False`
    return_grids : bool, optional
        If `True`, returns the U and V grids.

        Defaults to `True`

    Returns
    -------
    CyclogeostrophyResult
        Named tuple containing:
        - ``ucg``: U component of cyclogeostrophic velocity (on U grid)
        - ``vcg``: V component of cyclogeostrophic velocity (on V grid)
        - ``ug``, ``vg``: Geostrophic velocities (if ``return_geos=True``)
        - ``lat_u``, ``lon_u``, ``lat_v``, ``lon_v``: Grid coordinates (if ``return_grids=True``)
    """
    setup = setup_cyclogeostrophy(ssh_t, lat_t, lon_t, mask)

    ucg_u, vcg_v = _gradient_wind(
        setup.ug_u, setup.vg_v,
        setup.dx_u, setup.dx_v, setup.dy_u, setup.dy_v,
        setup.coriolis_factor_t, setup.is_land, setup.grid_angle_t
    )

    return assemble_result(
        ucg_u, vcg_v, setup,
        return_geos=return_geos,
        return_grids=return_grids,
    )


@jax.jit
def _gradient_wind(
    ug_u: Float[jax.Array, "lat lon"],
    vg_v: Float[jax.Array, "lat lon"],
    dx_u: Float[jax.Array, "lat lon"],
    dx_v: Float[jax.Array, "lat lon"],
    dy_u: Float[jax.Array, "lat lon"],
    dy_v: Float[jax.Array, "lat lon"],
    coriolis_factor_t: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    grid_angle_t: Float[jax.Array, "lat lon"]
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]]:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field from a Sea Surface Height (SSH) field
    using the gradient wind equation.

    The cyclogeostrophic SSC velocity field is computed on a C-grid, following NEMO convention.

    Parameters
    ----------
    ug_u : Float[jax.Array, "lat lon"]
        U component of the geostrophic SSC velocity field
    vg_v : Float[jax.Array, "lat lon"]
        V component of the geostrophic SSC velocity field
    dx_u : Float[jax.Array, "lat lon"]
        Spatial steps in meters along `x` on the U grid
    dx_v : Float[jax.Array, "lat lon"]
        Spatial steps in meters along `x` on the V grid
    dy_u : Float[jax.Array, "lat lon"]
        Spatial steps in meters along `y` on the U grid
    dy_v : Float[jax.Array, "lat lon"]
        Spatial steps in meters along `y` on the V grid
    coriolis_factor_t : Float[jax.Array, "lat lon"]
        Coriolis factor on the T grid
    mask : Float[jax.Array, "lat lon"]
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
    
    Returns
    -------
    ucg_u : Float[jax.Array, "lat lon"]
        U component of the cyclogeostrophic SSC velocity field (on the U grid)
    vcg_v : Float[jax.Array, "lat lon"]
        V component of the cyclogeostrophic SSC velocity field (on the V grid)
    """
    R = _radius_of_curvature(
        ug_u, vg_v, dx_u, dx_v, dy_u, dy_v, mask, vel_on_uv=True, grid_angle_t=grid_angle_t
    )

    V_g = kinematics.magnitude(ug_u, vg_v, mask)
    V_gr = 2 * V_g / (1 + jnp.sqrt(1 + 4 * V_g / (coriolis_factor_t * R)))

    ratio = V_gr / V_g

    ratio_u = operators.interpolation(ratio, mask, axis=1, padding="right")
    ratio_v = operators.interpolation(ratio, mask, axis=0, padding="right")

    ucg_u = ratio_u * ug_u
    vcg_v = ratio_v * vg_v

    return ucg_u, vcg_v
