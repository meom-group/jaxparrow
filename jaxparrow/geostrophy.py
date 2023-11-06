from typing import Tuple, Union

import numpy as np

from .tools import tools

__all__ = ["geostrophy"]


# =============================================================================
# Geostrophy
# =============================================================================

def geostrophy(ssh: Union[np.ndarray, np.ma.MaskedArray],
               dx_ssh: Union[np.ndarray, np.ma.MaskedArray], dy_ssh: Union[np.ndarray, np.ma.MaskedArray],
               coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray],
               coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray]) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Computes the geostrophic balance

    :param ssh: Sea Surface Height (SSH), NxM grid
    :type ssh: Union[np.ndarray, np.ma.MaskedArray]
    :param dx_ssh: SSH spatial step along x, NxM grid
    :type dx_ssh: Union[np.ndarray, np.ma.MaskedArray]
    :param dy_ssh: SSH spatial step along y, NxM grid
    :type dy_ssh: Union[np.ndarray, np.ma.MaskedArray]
    :param coriolis_factor_u: U Coriolis factor, NxM grid
    :type coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray]
    :param coriolis_factor_v: V Coriolis factor, NxM grid
    :type coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray]

    :returns: U and V geostrophic velocities, NxM grids
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Computing the gradient of the ssh
    grad_ssh_x, grad_ssh_y = tools.compute_gradient(ssh, dx_ssh, dy_ssh)

    # Interpolation of the data (moving the grad into the u and v position)
    grad_ssh_y = tools.interpolate(grad_ssh_y, axis=0)
    grad_ssh_y = tools.interpolate(grad_ssh_y, axis=1)

    grad_ssh_x = tools.interpolate(grad_ssh_x, axis=1)
    grad_ssh_x = tools.interpolate(grad_ssh_x, axis=0)

    # Interpolating the coriolis
    cu = tools.interpolate(coriolis_factor_u, axis=0)
    cu = tools.interpolate(cu, axis=1)

    cv = tools.interpolate(coriolis_factor_v, axis=1)
    cv = tools.interpolate(cv, axis=0)

    # Computing the geostrophic velocities
    u_geos = - tools.GRAVITY * grad_ssh_y / cu
    v_geos = tools.GRAVITY * grad_ssh_x / cv

    return np.nan_to_num(u_geos, nan=0, posinf=0, neginf=0), np.nan_to_num(v_geos, nan=0, posinf=0, neginf=0)
