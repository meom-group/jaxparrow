from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from .tools import tools

__all__ = ["geostrophy"]


# =============================================================================
# Geostrophy
# =============================================================================

def geostrophy(
        ssh: Float[Array, "lat lon"],
        lat: Float[Array, "lat lon"],
        lon: Float[Array, "lat lon"],
        lat_u: Float[Array, "lat lon"],
        lat_v: Float[Array, "lat lon"],
        mask_t: Float[Array, "lat lon"] = None,
        mask_u: Float[Array, "lat lon"] = None,
        mask_v: Float[Array, "lat lon"] = None
) -> Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """Computes the geostrophic balance

    :param ssh: Sea Surface Height (SSH)
    :type ssh: Float[Array, "lat lon"]
    :param lat: latitude of the T points
    :type lat: Float[Array, "lat lon"]
    :param lon: longitude of the T points
    :type lon: Float[Array, "lat lon"]
    :param lat_u: latitude of the U points
    :type lat_u: Float[Array, "lat lon"]
    :param lat_v: latitude of the V points
    :type lat_v: Float[Array, "lat lon"]
    :param mask_t: mask to apply at T points, 1 or True stands for masked, defaults to None
    :type mask_t: Float[Array, "lat lon"], optional
    :param mask_u: mask to apply at U points, 1 or True stands for masked, defaults to None
    :type mask_u: Float[Array, "lat lon"], optional
    :param mask_v: mask to apply at V points, 1 or True stands for masked, defaults to None
    :type mask_v: Float[Array, "lat lon"], optional

    :returns: U and V geostrophic velocities
    :rtype: Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]
    """
    if mask_u is None:
        mask_u = mask_t
    if mask_v is None:
        mask_v = mask_t

    # Computing spatial steps and Coriolis factors
    dx_ssh, dy_ssh = tools.compute_spatial_step(lat, lon)
    coriolis_factor_u = tools.compute_coriolis_factor(lat_u)
    coriolis_factor_v = tools.compute_coriolis_factor(lat_v)

    # Handling spurious and masked data
    ssh = tools.sanitise_data(ssh, jnp.nan, mask_t)  # avoid spurious velocities near the coast
    dx_ssh = tools.sanitise_data(dx_ssh, jnp.nan, mask_t)
    dy_ssh = tools.sanitise_data(dy_ssh, jnp.nan, mask_t)
    coriolis_factor_u = tools.sanitise_data(coriolis_factor_u, jnp.nan, mask_u)
    coriolis_factor_v = tools.sanitise_data(coriolis_factor_v, jnp.nan, mask_v)

    u_geos, v_geos = _geostrophy(ssh, dx_ssh, dy_ssh, coriolis_factor_u, coriolis_factor_v)

    # Handling masked data
    u_geos = tools.sanitise_data(u_geos, jnp.nan, mask_u)
    v_geos = tools.sanitise_data(v_geos, jnp.nan, mask_v)

    return u_geos, v_geos


def _geostrophy(
        ssh: Float[Array, "lat lon"],
        dx_ssh: Float[Array, "lat lon"],
        dy_ssh: Float[Array, "lat lon"],
        coriolis_factor_u: Float[Array, "lat lon"],
        coriolis_factor_v: Float[Array, "lat lon"]
) -> Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """Computes the geostrophic balance (internal)

    :param ssh: Sea Surface Height (SSH)
    :type ssh: Float[Array, "lat lon"]
    :param dx_ssh: spatial steps along x at T points
    :type dx_ssh: Float[Array, "lat lon"]
    :param dy_ssh: spatial steps along y at T points
    :type dy_ssh: Float[Array, "lat lon"]
    :param coriolis_factor_u: coriolis factor at U points
    :type coriolis_factor_u: Float[Array, "lat lon"]
    :param coriolis_factor_v: coriolis factor at V points
    :type coriolis_factor_v: Float[Array, "lat lon"]

    :returns: U and V geostrophic velocities
    :rtype: Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]
    """
    # Computing the gradient of the ssh
    grad_ssh_x, grad_ssh_y = tools.compute_gradient(ssh, dx_ssh, dy_ssh)

    # Interpolation of the data (moving the grad into the u and v position)  # TODO: not sure about this
    grad_ssh_y = tools.interpolate_south_west(grad_ssh_y, axis=0)  # t point
    grad_ssh_y = tools.interpolate_north_east(grad_ssh_y, axis=1)  # u point

    grad_ssh_x = tools.interpolate_south_west(grad_ssh_x, axis=1)  # t point
    grad_ssh_x = tools.interpolate_north_east(grad_ssh_x, axis=0)  # v point

    # Computing the geostrophic velocities
    u_geos = - tools.GRAVITY * grad_ssh_y / coriolis_factor_u
    v_geos = tools.GRAVITY * grad_ssh_x / coriolis_factor_v

    return u_geos, v_geos
