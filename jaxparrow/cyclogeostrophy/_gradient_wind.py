import jax
import jax.numpy as jnp
from jaxtyping import Float

from ._core import (
    CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result
)
from ..utils import kinematics


def gradient_wind(
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    ssh_t: Float[jax.Array, "y x"] = None,
    ug_t: Float[jax.Array, "y x"] = None,
    vg_t: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    return_geos: bool = False
) -> CyclogeostrophyResult:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field
    using the gradient wind approximation.

    There are two modes of operation:

    1. **SSH mode**: Provide ``lat_t``, ``lon_t``, ``ssh_t`` (and optionally ``mask``).
       Geostrophic velocities will be computed from SSH.

    2. **Geostrophic mode**: Provide ``lat_t``, ``lon_t``, ``ug_t``, ``vg_t``
       (and optionally ``land_mask``). Geostrophic velocities are provided on the T grid.

    Parameters
    ----------
    lat_t : Float[jax.Array, "y x"]
        Latitude of the T grid
    lon_t : Float[jax.Array, "y x"]
        Longitude of the T grid
    ssh_t : Float[jax.Array, "y x"], optional
        SSH field (on the T grid)

        Defaults to `None`, required if geostrophic velocities are not provided
    ug_t : Float[jax.Array, "y x"], optional
        U component of geostrophic velocity on T grid
        
        Defaults to `None`, required if ``ssh_t`` is not provided
    vg_t : Float[jax.Array, "y x"], optional
        V component of geostrophic velocity on T grid
        
        Defaults to `None`, required if ``ssh_t`` is not provided
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` or ``ug_t`` `nan` values

        Defaults to `None`
    return_geos : bool, optional
        If `True`, returns the geostrophic SSC velocity field in addition to the cyclogeostrophic one.

        Defaults to `False`

    Returns
    -------
    CyclogeostrophyResult
        Named tuple containing:
        - ``ucg``: $u$ component of cyclogeostrophic velocity, on the T grid
        - ``vcg``: $v$ component of cyclogeostrophic velocity, on the T grid
        - ``ug``, ``vg``: Geostrophic velocities (if ``return_geos=True``)
    """
    setup = setup_cyclogeostrophy(
        lat_t, lon_t, ssh_t=ssh_t, ug_t=ug_t, vg_t=vg_t, land_mask=land_mask
    )

    ucg, vcg = _gradient_wind(
        setup.ug_t, setup.vg_t,
        setup.dx_e_t, setup.dx_n_t, setup.dy_e_t, setup.dy_n_t, setup.J_t,
        setup.coriolis_factor_t, 
        setup.land_mask
    )

    return assemble_result(ucg, vcg, setup, return_geos=return_geos)


@jax.jit
def _gradient_wind(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"]
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    R = kinematics._radius_of_curvature(ug_t, vg_t, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, land_mask)

    V_g = kinematics.magnitude(ug_t, vg_t, land_mask, uv_on_t=True)
    V_gr = 2 * V_g / (1 + jnp.sqrt(1 + 4 * V_g / (coriolis_factor_t * R)))

    ratio = V_gr / V_g

    ucg = ratio * ug_t
    vcg = ratio * vg_t

    return ucg, vcg
