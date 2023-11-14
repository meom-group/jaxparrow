from typing import Tuple, Union

import jax
import numpy as np
import jax.numpy as jnp

#: Approximate earth angular speed
EARTH_ANG_SPEED = 7.292115e-5
#: Approximate earth radius
EARTH_RADIUS = 6370e3
#: Approximate gravity
GRAVITY = 9.81
P0 = np.pi / 180

__all__ = ["compute_coriolis_factor", "compute_spatial_step"]


# =============================================================================
# Functions without JAX
# =============================================================================

def compute_coriolis_factor(lat: Union[int, np.ndarray, np.ma.MaskedArray]) -> Union[np.ndarray, np.ma.MaskedArray]:
    """Computes the Coriolis factor from latitudes

    :param lat: latitude, NxM grid
    :type lat: Union[np.ndarray, np.ma.MaskedArray]

    :returns: Coriolis factor, NxM grid
    :rtype: Union[np.ndarray, np.ma.MaskedArray]
    """
    return 2 * EARTH_ANG_SPEED * np.sin(lat * np.pi / 180)


def _neuman_forward(field: Union[np.ndarray, np.ma.MaskedArray], axis: int = 0) -> np.ndarray:
    """Applies Von Neuman boundary conditions to the field

    :param field: field values
    :type field: Union[np.ndarray, np.ma.MaskedArray]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: field values with boundary conditions
    :rtype: np.ndarray
    """
    f = np.copy(field)
    if axis == 0:
        f[-1, :] = field[-2, :]
    if axis == 1:
        f[:, -1] = field[:, -2]
    return f


def compute_spatial_step(lat: Union[np.ndarray, np.ma.MaskedArray], lon: Union[np.ndarray, np.ma.MaskedArray],
                         bounds: Tuple[float, float] = (1e2, 1e4), fill_value: float = 1e3) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Computes dx and dy spatial steps of a grid defined by lat, lon.
    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of cos and arccos functions
    to avoid truncation issues.
    Applies Von Neuman boundary conditions to the spatial steps fields.

    :param lat: latitude, NxM grid
    :type lat: Union[np.ndarray, np.ma.MaskedArray]
    :param lon: longitude, NxM grid
    :type lon: Union[np.ndarray, np.ma.MaskedArray]
    :param bounds: range of acceptable values, defaults to (1e2, 1e4). Out of this range, set to fill_value
    :type bounds: Tuple[float, float], optional
    :param fill_value: fill value, defaults to 1e12
    :type fill_value: float, optional

    :returns: dx and dy spatial steps, NxM grids
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    dx, dy = np.zeros_like(lon), np.zeros_like(lon)
    # dx
    dlat, dlon = P0 * (lat[:, 1:] - lat[:, :-1]), P0 * (lon[:, 1:] - lon[:, :-1])
    dx[:, :-1] = EARTH_RADIUS * np.sqrt(dlat ** 2 + np.cos(P0 * lat[:, :-1]) * np.cos(P0 * lat[:, 1:]) * dlon ** 2)
    dx = _neuman_forward(dx, axis=1)
    dx = np.where(dx > bounds[0], dx, fill_value)  # avoid zero or huge steps due to
    dx = np.where(dx < bounds[1], dx, fill_value)  # spurious zero values in lon lat arrays
    # dy
    dlat, dlon = P0 * (lat[1:, :] - lat[:-1, :]), P0 * (lon[1:, :] - lon[:-1, :])
    dy[:-1, :] = EARTH_RADIUS * np.sqrt(dlat ** 2 + np.cos(P0 * lat[:-1, :]) * np.cos(P0 * lat[1:, :]) * dlon ** 2)
    dy = _neuman_forward(dy, axis=0)
    dy = np.where(dy > bounds[0], dy, fill_value)
    dy = np.where(dy < bounds[1], dy, fill_value)
    return dx, dy


def interpolate(field: Union[np.ndarray, np.ma.MaskedArray], axis: int = 0) -> np.ndarray:
    """Interpolates the values of a field in the y direction

    :param field: field values
    :type field: Union[np.ndarray, np.ma.MaskedArray]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: interpolated field values
    :rtype: np.ndarray
    """
    f = np.copy(field)
    if axis == 0:
        f[:-1, :] = 0.5 * (field[:-1, :] + field[1:, :])
    if axis == 1:
        f[:, :-1] = 0.5 * (field[:, :-1] + field[:, 1:])
    f = _neuman_forward(f, axis=axis)
    return f


def compute_derivative(field: Union[np.ndarray, np.ma.MaskedArray], dxy: Union[np.ndarray, np.ma.MaskedArray],
                       axis: int = 0) -> np.ndarray:
    """Computes the x or y derivatives of a 2D field using finite differences

    :param field: field values, NxM grid
    :type field: Union[np.ndarray, np.ma.MaskedArray]
    :param dxy: spatial steps, NxM grid
    :type dxy: Union[np.ndarray, np.ma.MaskedArray]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: derivatives, NxM grid
    :rtype: np.ndarray
    """
    f = np.copy(field)
    if axis == 0:
        f[:-1, :] = field[1:, :] - field[:-1, :]
    if axis == 1:
        f[:, :-1] = field[:, 1:] - field[:, :-1]
    f = _neuman_forward(f, axis=axis)
    return f / dxy


def compute_gradient(field: Union[np.ndarray, np.ma.MaskedArray],
                     dx: Union[np.ndarray, np.ma.MaskedArray], dy: Union[np.ndarray, np.ma.MaskedArray]) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Computes the gradient of a field

    :param field: field values
    :type field: Union[np.ndarray, np.ma.MaskedArray]
    :param dx: spatial steps along x
    :type dx: Union[np.ndarray, np.ma.MaskedArray]
    :param dy: spatial steps along y
    :type dy: Union[np.ndarray, np.ma.MaskedArray]

    :returns: gradients
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    fx, fy = compute_derivative(field, dx, axis=1), compute_derivative(field, dy, axis=0)
    return fx, fy


def compute_advection_u(u: Union[np.ndarray, np.ma.MaskedArray], v: Union[np.ndarray, np.ma.MaskedArray],
                        dx: Union[np.ndarray, np.ma.MaskedArray], dy: Union[np.ndarray, np.ma.MaskedArray]) \
        -> np.ndarray:
    """Computes the advection term for a velocity field in the direction x.
    The function also interpolate the values to a V point

    :param u: U velocity
    :type u: Union[np.ndarray, np.ma.MaskedArray]
    :param v: V velocity
    :type v: Union[np.ndarray, np.ma.MaskedArray]
    :param dx: spatial steps along x
    :type dx: Union[np.ndarray, np.ma.MaskedArray]
    :param dy: spatial steps along y
    :type dy: Union[np.ndarray, np.ma.MaskedArray]

    :returns: advection term
    :rtype: np.ndarray
    """
    u_adv = np.copy(u)
    v_adv = np.copy(v)

    dudx = compute_derivative(u, dx, axis=1)  # h points
    dudx = interpolate(dudx, axis=0)  # v points

    dudy = compute_derivative(u, dy, axis=0)  # vorticity points
    dudy = interpolate(dudy, axis=1)  # v points

    u_adv = interpolate(u_adv, axis=1)  # h points
    u_adv = interpolate(u_adv, axis=0)  # v points

    adv_u = u_adv * dudx + v_adv * dudy  # v points
    return adv_u


def compute_advection_v(u: Union[np.ndarray, np.ma.MaskedArray], v: Union[np.ndarray, np.ma.MaskedArray],
                        dx: Union[np.ndarray, np.ma.MaskedArray], dy: Union[np.ndarray, np.ma.MaskedArray]) \
        -> np.ndarray:
    """Computes the advection term for a velocity field in the direction x.
    The function also interpolate the values to a U point

    :param u: U velocity
    :type u: Union[np.ndarray, np.ma.MaskedArray]
    :param v: V velocity
    :type v: Union[np.ndarray, np.ma.MaskedArray]
    :param dx: spatial steps along x
    :type dx: Union[np.ndarray, np.ma.MaskedArray]
    :param dy: spatial steps along y
    :type dy: Union[np.ndarray, np.ma.MaskedArray]

    :returns: advection term
    :rtype: np.ndarray
    """
    u_adv = np.copy(u)
    v_adv = np.copy(v)

    dvdx = compute_derivative(v, dx, axis=1)  # vorticity points
    dvdx = interpolate(dvdx, axis=0)  # u points

    dvdy = compute_derivative(v, dy, axis=0)  # h points
    dvdy = interpolate(dvdy, axis=1)  # u points

    v_adv = interpolate(v_adv, axis=1)  # vorticity points
    v_adv = interpolate(v_adv, axis=0)  # u points

    adv_v = u_adv * dvdx + v_adv * dvdy  # u points
    return adv_v


# =============================================================================
# Functions with JAX
# =============================================================================

def _neuman_forward_jax(field: Union[np.ndarray, jax.Array], axis: int = 0) -> jax.Array:
    """Applies Von Neuman boundary conditions to the field

    :param field: field values
    :type field: Union[np.ndarray, jax.Array]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: field values with boundary conditions
    :rtype: jax.Array
    """
    f = jnp.copy(field)
    if axis == 0:
        f = f.at[-1, :].set(field[-2, :])
    if axis == 1:
        f = f.at[:, -1].set(field[:, -2])
    return f


def _interpolate_jax(field: Union[np.ndarray, jax.Array], axis: int = 0) -> jax.Array:
    """Interpolates the values of a field in the y direction

    :param field: field values
    :type field: Union[np.ndarray, jax.Array]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: interpolated field values
    :rtype: jax.Array
    """
    f = jnp.copy(field)
    if axis == 0:
        f = f.at[:-1, :].set(0.5 * (field[:-1, :] + field[1:, :]))
    if axis == 1:
        f = f.at[:, :-1].set(0.5 * (field[:, :-1] + field[:, 1:]))
    f = _neuman_forward_jax(f, axis=axis)
    return f


def _compute_derivative_jax(field: np.ndarray, dxy: np.ndarray, axis: int = 0) -> jax.Array:
    """Computes the x or y derivatives of a 2D field using finite differences

    :param field: field values
    :type field: np.ndarray
    :param dxy: spatial steps
    :type dxy: np.ndarray
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: derivatives
    :rtype: jax.Array
    """
    f = jnp.copy(field)
    if axis == 0:
        f = f.at[:-1, :].set(field[1:, :] - field[:-1, :])
    if axis == 1:
        f = f.at[:, :-1].set(field[:, 1:] - field[:, :-1])
    f = _neuman_forward_jax(f, axis=axis)
    return f / dxy


def compute_gradient_jax(field: Union[np.ndarray, jax.Array], dx: np.ndarray, dy: np.ndarray) \
        -> Tuple[jax.Array, jax.Array]:
    """Computes the gradient of a field

    :param field: field values
    :type field: Union[np.ndarray, jax.Array]
    :param dx: spatial steps along x
    :type dx: np.ndarray
    :param dy: spatial steps along y
    :type dy: np.ndarray

    :returns: gradients
    :rtype: Tuple[jax.Array, jax.Array]
    """
    fx, fy = _compute_derivative_jax(field, dx, axis=1), _compute_derivative_jax(field, dy, axis=0)
    return fx, fy


def compute_advection_u_jax(u: Union[np.ndarray, jax.Array], v: Union[np.ndarray, jax.Array],
                            dx: np.ndarray, dy: np.ndarray) -> jax.Array:
    """Computes the advection term for a velocity field in the direction x.
    The function also interpolate the values to a V point

    :param u: U velocity
    :type u: Union[np.ndarray, jax.Array]
    :param v: V velocity
    :type v: Union[np.ndarray, jax.Array]
    :param dx: spatial steps along x
    :type dx: np.ndarray
    :param dy: spatial steps along y
    :type dy: np.ndarray

    :returns: advection term
    :rtype: jax.Array
    """
    u_adv = jnp.copy(u)
    v_adv = jnp.copy(v)

    dudx = _compute_derivative_jax(u, dx, axis=1)  # h points
    dudx = _interpolate_jax(dudx, axis=0)  # v points

    dudy = _compute_derivative_jax(u, dy, axis=0)  # vorticity points
    dudy = _interpolate_jax(dudy, axis=1)  # v points

    u_adv = _interpolate_jax(u_adv, axis=1)  # h points
    u_adv = _interpolate_jax(u_adv, axis=0)  # v points

    adv_u = u_adv * dudx + v_adv * dudy  # v points
    return adv_u


def compute_advection_v_jax(u: np.ndarray, v: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> jax.Array:
    """Computes the advection term for a velocity field in the direction x.
    The function also interpolate the values to a U point

    :param u: U velocity
    :type u: np.ndarray
    :param v: V velocity
    :type v: np.ndarray
    :param dx: spatial steps along x
    :type dx: np.ndarray
    :param dy: spatial steps along y
    :type dy: np.ndarray

    :returns: advection term
    :rtype: jax.Array
    """
    u_adv = jnp.copy(u)
    v_adv = jnp.copy(v)

    dvdx = _compute_derivative_jax(v, dx, axis=1)  # vorticity points
    dvdx = _interpolate_jax(dvdx, axis=0)  # u points

    dvdy = _compute_derivative_jax(v, dy, axis=0)  # h points
    dvdy = _interpolate_jax(dvdy, axis=1)  # u points

    v_adv = _interpolate_jax(v_adv, axis=1)  # vorticity points
    v_adv = _interpolate_jax(v_adv, axis=0)  # u points

    adv_v = u_adv * dvdx + v_adv * dvdy  # u points
    return adv_v


def compute_cyclogeostrophic_diff_jax(u_geos: np.ndarray, v_geos: np.ndarray,
                                      u_cyclo: Union[np.ndarray, jax.Array], v_cyclo: Union[np.ndarray, jax.Array],
                                      dx_u: np.ndarray, dx_v: np.ndarray, dy_u: np.ndarray, dy_v: np.ndarray,
                                      coriolis_factor_u: np.ndarray, coriolis_factor_v: np.ndarray) -> jax.Array:
    """Computes the cyclogeostrophic imbalance

    :param u_geos: U geostrophic velocity value
    :type u_geos: np.ndarray
    :param v_geos: V geostrophic velocity value
    :type v_geos: np.ndarray
    :param u_cyclo: U cyclogeostrophic velocity value
    :type u_cyclo: Union[np.ndarray, jax.Array]
    :param v_cyclo: V cyclogeostrophic velocity value
    :type v_cyclo: Union[np.ndarray, jax.Array]
    :param dx_u: U spatial step along x
    :type dx_u: np.ndarray
    :param dx_v: V spatial step along x
    :type dx_v: np.ndarray
    :param dy_u: U spatial step along y
    :type dy_u: np.ndarray
    :param dy_v: V spatial step along y
    :type dy_v: np.ndarray
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: np.ndarray
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: np.ndarray

    :returns: the loss
    :rtype: jax.Array
    """
    J_u = jnp.sum(
        (u_cyclo + compute_advection_v_jax(u_cyclo, v_cyclo, dx_v, dy_v) / coriolis_factor_u - u_geos) ** 2)
    J_v = jnp.sum(
        (v_cyclo - compute_advection_u_jax(u_cyclo, v_cyclo, dx_u, dy_u) / coriolis_factor_v - v_geos) ** 2)
    return J_u + J_v
