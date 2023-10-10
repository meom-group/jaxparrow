from typing import Tuple, Union

import jax
import numpy as np
import jax.numpy as jnp

EARTH_RADIUS = 6370e3
P0 = np.pi / 180


# =============================================================================
# Functions without JAX
# =============================================================================

def compute_coriolis_factor(lat: Union[np.ndarray, np.ma.MaskedArray]) -> Union[np.ndarray, np.ma.MaskedArray]:
    """Computes the Coriolis factor from latitudes

    :param lat: latitude
    :type lat: Union[np.ndarray, np.ma.MaskedArray]

    :returns: Coriolis factor
    :rtype: Union[np.ndarray, np.ma.MaskedArray]
    """
    return 2 * 7.2722e-05 * np.sin(lat * np.pi / 180)


def neuman_forward(field: Union[np.ndarray, np.ma.MaskedArray], axis: int = 0) -> np.ndarray:
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
                         bounds: Tuple[float, float] = (1e2, 1e4), fill_value: float = 1e12) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Computes dx and dy spatial steps of a grid defined by lat, lon.
    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of cos and arccos functions
    to avoid truncation issues.

    :param lat: latitude
    :type lat: Union[np.ndarray, np.ma.MaskedArray]
    :param lon: longitude
    :type lon: Union[np.ndarray, np.ma.MaskedArray]
    :param bounds: range of acceptable values, defaults to (1e2, 1e4). Out of this range, set to fill_value
    :type bounds: Tuple[float, float], optional
    :param fill_value: flll value, defaults to 1e12
    :type fill_value: float, optional

    :returns: dx and dy spatial steps
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    dx, dy = np.zeros_like(lon), np.zeros_like(lon)
    # dx
    dlat, dlon = P0 * (lat[:, 1:] - lat[:, :-1]), P0 * (lon[:, 1:] - lon[:, :-1])
    dx[:, :-1] = EARTH_RADIUS * np.sqrt(dlat ** 2 + np.cos(P0 * lat[:, :-1]) * np.cos(P0 * lat[:, 1:]) * dlon ** 2)
    dx = neuman_forward(dx, axis=1)
    dx = np.where(dx > bounds[0], dx, fill_value)  # avoid zero or huge steps due to
    dx = np.where(dx < bounds[1], dx, fill_value)  # spurious zero values in lon lat arrays
    # dy
    dlat, dlon = P0 * (lat[1:, :] - lat[:-1, :]), P0 * (lon[1:, :] - lon[:-1, :])
    dy[:-1, :] = EARTH_RADIUS * np.sqrt(dlat ** 2 + np.cos(P0 * lat[:-1, :]) * np.cos(P0 * lat[1:, :]) * dlon ** 2)
    dy = neuman_forward(dy, axis=0)
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
    f = neuman_forward(f, axis=axis)
    return f


def compute_derivative(field: Union[np.ndarray, np.ma.MaskedArray], dxy: Union[np.ndarray, np.ma.MaskedArray],
                       axis: int = 0) -> np.ndarray:
    """Computes the x or y derivatives of a 2D field using finite differences

    :param field: field values
    :type field: Union[np.ndarray, np.ma.MaskedArray]
    :param dxy: spatial steps
    :type dxy: Union[np.ndarray, np.ma.MaskedArray]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: derivatives
    :rtype: np.ndarray
    """
    f = np.copy(field)
    if axis == 0:
        f[:-1, :] = field[1:, :] - field[:-1, :]
    if axis == 1:
        f[:, :-1] = field[:, 1:] - field[:, :-1]
    f = neuman_forward(f, axis=axis)
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

def neuman_forward_jax(field: Union[np.ndarray, jax.Array], axis: int = 0) -> jax.Array:
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


def interpolate_jax(field: Union[np.ndarray, jax.Array], axis: int = 0) -> jax.Array:
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
    f = neuman_forward_jax(f, axis=axis)
    return f


def compute_derivative_jax(field: np.ndarray, dxy: np.ndarray, axis: int = 0) -> jax.Array:
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
    f = neuman_forward_jax(f, axis=axis)
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
    fx, fy = compute_derivative_jax(field, dx, axis=1), compute_derivative_jax(field, dy, axis=0)
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

    dudx = compute_derivative_jax(u, dx, axis=1)  # h points
    dudx = interpolate_jax(dudx, axis=0)  # v points

    dudy = compute_derivative_jax(u, dy, axis=0)  # vorticity points
    dudy = interpolate_jax(dudy, axis=1)  # v points

    u_adv = interpolate_jax(u_adv, axis=1)  # h points
    u_adv = interpolate_jax(u_adv, axis=0)  # v points

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

    dvdx = compute_derivative_jax(v, dx, axis=1)  # vorticity points
    dvdx = interpolate_jax(dvdx, axis=0)  # u points

    dvdy = compute_derivative_jax(v, dy, axis=0)  # h points
    dvdy = interpolate_jax(dvdy, axis=1)  # u points

    v_adv = interpolate_jax(v_adv, axis=1)  # vorticity points
    v_adv = interpolate_jax(v_adv, axis=0)  # u points

    adv_v = u_adv * dvdx + v_adv * dvdy  # u points
    return adv_v
