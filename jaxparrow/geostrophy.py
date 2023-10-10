from typing import Tuple, Union

import numpy as np

from jaxparrow.tools import geometry as geo

GRAVITY = 9.81


# =============================================================================
# Geostrophy
# =============================================================================

def geostrophy(ssh: Union[np.ndarray, np.ma.MaskedArray],
               dx_ssh: Union[np.ndarray, np.ma.MaskedArray], dy_ssh: Union[np.ndarray, np.ma.MaskedArray],
               coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray],
               coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray]) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Computes the geostrophic balance

    :param ssh: Sea Surface Height (SSH) value
    :type ssh: Union[np.ndarray, np.ma.MaskedArray]
    :param dx_ssh: SSH spatial step along x
    :type dx_ssh: Union[np.ndarray, np.ma.MaskedArray]
    :param dy_ssh: SSH spatial step along y
    :type dy_ssh: Union[np.ndarray, np.ma.MaskedArray]
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray]
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray]

    :returns: U and V geostrophic velocities
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Computing the gradient of the ssh
    grad_ssh_x, grad_ssh_y = geo.compute_gradient(ssh, dx_ssh, dy_ssh)

    # Interpolation of the data (moving the grad into the u and v position)
    grad_ssh_y = geo.interpolate(grad_ssh_y, axis=0)
    grad_ssh_y = geo.interpolate(grad_ssh_y, axis=1)

    grad_ssh_x = geo.interpolate(grad_ssh_x, axis=1)
    grad_ssh_x = geo.interpolate(grad_ssh_x, axis=0)

    # Interpolating the coriolis
    cu = geo.interpolate(coriolis_factor_u, axis=0)
    cu = geo.interpolate(cu, axis=1)

    cv = geo.interpolate(coriolis_factor_v, axis=1)
    cv = geo.interpolate(cv, axis=0)

    # Computing the geostrophic velocities
    u_geos = - GRAVITY * grad_ssh_y / cu
    v_geos = GRAVITY * grad_ssh_x / cv

    return u_geos, v_geos
