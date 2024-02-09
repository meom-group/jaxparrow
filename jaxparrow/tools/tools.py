from typing import Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

#: Approximate earth angular speed
EARTH_ANG_SPEED = 7.292115e-5
#: Approximate earth radius
EARTH_RADIUS = 6370e3
#: Approximate gravity
GRAVITY = 9.81
P0 = jnp.pi / 180


def sanitise_data(
        arr: Float[Array, "lat lon"],
        fill_value: float = None,
        mask: Float[Array, "lat lon"] = None
) -> Float[Array, "lat lon"]:
    """Sanitise data by replacing NaNs with `fill_value` and applying `fill_value` to the masked area.

    :param arr: array to sanitise
    :type arr: Float[Array, "lat lon"]
    :param fill_value: value to replace NaNs and masked part with, use the mean if not specified, defaults to None
    :type fill_value: float, optional
    :param mask: mask to apply, 1 or True stands for masked, defaults to None
    :type mask: Float[Array, "lat lon"], optional

    :returns: field values with boundary conditions
    :rtype: Float[Array, "lat lon"]
    """
    if fill_value is None:
        fill_value = jnp.nanmean(arr)
    arr = jnp.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if mask is not None:
        arr = jnp.where(mask, fill_value, arr)
    return arr


def _neuman_north_east(
        field: Float[Array, "lat lon"],
        axis: int = 0
) -> Float[Array, "lat lon"]:
    """Applies Von Neuman boundary conditions to the field (forward)

    :param field: field values
    :type field: Float[Array, "lat lon"]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: field values with boundary conditions
    :rtype: Float[Array, "lat lon"]
    """
    if axis == 0:
        field = field.at[-1, :].set(field[-2, :])
    if axis == 1:
        field = field.at[:, -1].set(field[:, -2])
    return field


def _neuman_south_west(
        field: Float[Array, "lat lon"],
        axis: int = 0
) -> Float[Array, "lat lon"]:
    """Applies Von Neuman boundary conditions to the field (backward)

    :param field: field values
    :type field: Float[Array, "lat lon"]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: field values with boundary conditions
    :rtype: Float[Array, "lat lon"]
    """
    if axis == 0:
        field = field.at[0, :].set(field[1, :])
    if axis == 1:
        field = field.at[:, 0].set(field[:, 1])
    return field


def compute_spatial_step(
        lat: Float[Array, "lat lon"],
        lon: Float[Array, "lat lon"],
        bounds: Tuple[float, float] = (1, 1e5),
        fill_value: float = 1  # this sounds a bit questionable
) -> Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """Computes dx and dy spatial steps of a grid defined by lat, lon.
    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of cos and arccos functions
    to avoid truncation issues.
    Applies Von Neuman boundary conditions to the spatial steps fields.

    :param lat: latitude
    :type lat: Float[Array, "lat lon"]
    :param lon: longitude
    :type lon: Float[Array, "lat lon"]
    :param bounds: range of acceptable values, defaults to (1, 1e5). Out of this range, set to fill_value
    :type bounds: Tuple[float, float], optional
    :param fill_value: fill value, defaults to 1e5
    :type fill_value: float, optional

    :returns: dx and dy spatial steps, NxM grids
    :rtype: Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]
    """
    def sphere_distance(_lats, _late, _lons, _lone):
        dlat, dlon = P0 * (_late - _lats), P0 * (_lone - _lons)
        return EARTH_RADIUS * jnp.sqrt(dlat ** 2 + jnp.cos(P0 * _lats) * jnp.cos(P0 * _late) * dlon ** 2)

    dx, dy = jnp.zeros_like(lon), jnp.zeros_like(lon)
    # dx
    dx = dx.at[:, :-1].set(sphere_distance(lat[:, :-1], lat[:, 1:], lon[:, :-1], lon[:, 1:]))
    dx = _neuman_north_east(dx, axis=1)
    dx = jnp.where(dx > bounds[0], dx, fill_value)  # avoid zero or huge steps due to
    dx = jnp.where(dx < bounds[1], dx, fill_value)  # spurious zero values in lon lat arrays
    # dy
    dy = dy.at[:-1, :].set(sphere_distance(lat[:-1, :], lat[1:, :], lon[:-1, :], lon[1:, :]))
    dy = _neuman_north_east(dy, axis=0)
    dy = jnp.where(dy > bounds[0], dy, fill_value)
    dy = jnp.where(dy < bounds[1], dy, fill_value)
    return dx, dy


def compute_coriolis_factor(
        lat: Union[int, Float[Array, "lat lon"]]
) -> Float[Array, "lat lon"]:
    """Computes the Coriolis factor from latitudes

    :param lat: latitude
    :type lat: Float[Array, "lat lon"]

    :returns: Coriolis factor
    :rtype: Float[Array, "lat lon"]
    """
    return 2 * EARTH_ANG_SPEED * jnp.sin(lat * P0)


def interpolate_north_east(
        field: Float[Array, "lat lon"],
        axis: int = 0,
        neuman: bool = True
) -> Float[Array, "lat lon"]:
    """Interpolates the values of a field in the northward (axis=0) or eastward (axis=1) direction

    :param field: field values
    :type field: Float[Array, "lat lon"]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional
    :param neuman: apply Neuman boundary condition, defaults to True
    :type neuman: bool, optional

    :returns: interpolated field values
    :rtype: Float[Array, "lat lon"]
    """
    f = jnp.copy(field)  # make sure we manipulate Jax Array
    if axis == 0:
        f = f.at[:-1, :].set(0.5 * (f[:-1, :] + f[1:, :]))
    if axis == 1:
        f = f.at[:, :-1].set(0.5 * (f[:, :-1] + f[:, 1:]))
    if neuman:
        f = _neuman_north_east(f, axis=axis)
    return f


def interpolate_south_west(
        field: Float[Array, "lat lon"],
        axis: int = 0,
        neuman: bool = True
) -> Float[Array, "lat lon"]:
    """Interpolates the values of a field in the southward (axis=0) or westward (axis=1) direction

    :param field: field values
    :type field: Float[Array, "lat lon"]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional
    :param neuman: apply Neuman boundary condition, defaults to True
    :type neuman: bool, optional

    :returns: interpolated field values
    :rtype: Float[Array, "lat lon"]
    """
    f = jnp.copy(field)  # make sure we manipulate Jax Array
    if axis == 0:
        f = f.at[1:, :].set(0.5 * (f[:-1, :] + f[1:, :]))
    if axis == 1:
        f = f.at[:, 1:].set(0.5 * (f[:, :-1] + f[:, 1:]))
    if neuman:
        f = _neuman_south_west(f, axis=axis)
    return f


def compute_derivative_north_east(
        field: Float[Array, "lat lon"],
        dxy: Float[Array, "lat lon"],
        axis: int = 0
) -> Float[Array, "lat lon"]:
    """Computes the x or y derivatives of a 2D field using finite differences (forward)

    :param field: field values
    :type field: Float[Array, "lat lon"]
    :param dxy: spatial steps
    :type dxy: Float[Array, "lat lon"]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: derivatives
    :rtype: Float[Array, "lat lon"]
    """
    f = jnp.copy(field)  # make sure we manipulate Jax Array
    if axis == 0:
        f = f.at[:-1, :].set(f[1:, :] - f[:-1, :])
    if axis == 1:
        f = f.at[:, :-1].set(f[:, 1:] - f[:, :-1])
    f = _neuman_north_east(f, axis=axis)
    return f / dxy


def compute_derivative_south_west(
        field: Float[Array, "lat lon"],
        dxy: Float[Array, "lat lon"],
        axis: int = 0
) -> Float[Array, "lat lon"]:
    """Computes the x or y derivatives of a 2D field using finite differences (backward)

    :param field: field values
    :type field: Float[Array, "lat lon"]
    :param dxy: spatial steps
    :type dxy: Float[Array, "lat lon"]
    :param axis: axis along which boundary conditions are applied, defaults to 0
    :type axis: int, optional

    :returns: derivatives
    :rtype: Float[Array, "lat lon"]
    """
    f = jnp.copy(field)  # make sure we manipulate Jax Array
    if axis == 0:
        f = f.at[1:, :].set(f[1:, :] - f[:-1, :])
    if axis == 1:
        f = f.at[:, 1:].set(f[:, 1:] - f[:, :-1])
    f = _neuman_north_east(f, axis=axis)
    return f / dxy


def compute_gradient(
        field: Float[Array, "lat lon"],
        dx: Float[Array, "lat lon"],
        dy: Float[Array, "lat lon"]
) -> Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """Computes the gradient of a field

    :param field: field values
    :type field: Float[Array, "lat lon"]
    :param dx: spatial steps along x
    :type dx: Float[Array, "lat lon"]
    :param dy: spatial steps along y
    :type dy: Float[Array, "lat lon"]

    :returns: gradients
    :rtype: Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]
    """
    fx, fy = compute_derivative_north_east(field, dx, axis=1), compute_derivative_north_east(field, dy, axis=0)
    return fx, fy


def compute_advection_u(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
        dx: Float[Array, "lat lon"],
        dy: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    """Computes the advection term for a velocity field in the direction x.
    The function also interpolate the values to a V point

    :param u: U velocity
    :type u: Float[Array, "lat lon"]
    :param v: V velocity
    :type v: Float[Array, "lat lon"]
    :param dx: spatial steps along x
    :type dx: Float[Array, "lat lon"]
    :param dy: spatial steps along y
    :type dy: Float[Array, "lat lon"]

    :returns: advection term
    :rtype: Float[Array, "lat lon"]
    """
    dudx = compute_derivative_north_east(u, dx, axis=1)  # t points  # TODO: not sure about this
    dudx = interpolate_north_east(dudx, axis=0)  # v points

    dudy = compute_derivative_north_east(u, dy, axis=0)  # f points
    dudy = interpolate_north_east(dudy, axis=1)  # v points  # TODO: not sure about this

    u = interpolate_north_east(u, axis=1)  # t points
    u = interpolate_north_east(u, axis=0)  # v points  # TODO: not sure about this

    adv_u = u * dudx + v * dudy  # v points
    adv_u = sanitise_data(adv_u, 0)

    return adv_u


def compute_advection_v(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
        dx: Float[Array, "lat lon"],
        dy: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    """Computes the advection term for a velocity field in the direction y.
    The function also interpolate the values to a U point

    :param u: U velocity
    :type u: Float[Array, "lat lon"]
    :param v: V velocity
    :type v: Float[Array, "lat lon"]
    :param dx: spatial steps along x
    :type dx: Float[Array, "lat lon"]
    :param dy: spatial steps along y
    :type dy: Float[Array, "lat lon"]

    :returns: advection term
    :rtype: Float[Array, "lat lon"]
    """
    dvdx = compute_derivative_north_east(v, dx, axis=1)  # f points
    dvdx = interpolate_north_east(dvdx, axis=0)  # u points  # TODO: not sure about this

    dvdy = compute_derivative_north_east(v, dy, axis=0)  # t points  # TODO: not sure about this
    dvdy = interpolate_north_east(dvdy, axis=1)  # u points

    v = interpolate_north_east(v, axis=1)  # t points
    v = interpolate_north_east(v, axis=0)  # u points  # TODO: not sure about this

    adv_v = u * dvdx + v * dvdy  # u points
    adv_v = sanitise_data(adv_v, 0)

    return adv_v


def compute_cyclogeostrophic_diff(
        u_geos: Float[Array, "lat lon"],
        v_geos: Float[Array, "lat lon"],
        u_cyclo: Float[Array, "lat lon"],
        v_cyclo: Float[Array, "lat lon"],
        adv_u: Float[Array, "lat lon"],
        adv_v: Float[Array, "lat lon"],
        coriolis_factor_u: Float[Array, "lat lon"],
        coriolis_factor_v: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    """Computes the cyclogeostrophic imbalance

    :param u_geos: U geostrophic velocity value
    :type u_geos: Float[Array, "lat lon"]
    :param v_geos: V geostrophic velocity value
    :type v_geos: Float[Array, "lat lon"]
    :param u_cyclo: U cyclogeostrophic velocity value
    :type u_cyclo: Float[Array, "lat lon"]
    :param v_cyclo: V cyclogeostrophic velocity value
    :type v_cyclo: Float[Array, "lat lon"]
    :param adv_u: U advection term
    :type adv_u: Float[Array, "lat lon"]
    :param adv_v: V advection term
    :type adv_v: Float[Array, "lat lon"]
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: Float[Array, "lat lon"]
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: Float[Array, "lat lon"]

    :returns: the loss
    :rtype: Float[Array, "lat lon"]
    """
    J_u = jnp.nansum((u_cyclo + adv_v / coriolis_factor_u - u_geos) ** 2)
    J_v = jnp.nansum((v_cyclo - adv_u / coriolis_factor_v - v_geos) ** 2)
    return J_u + J_v


def compute_magnitude(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    """Computes the magnitude of a velocity field

    :param u: U velocity value
    :type u: Float[Array, "lat lon"]
    :param v: V velocity value
    :type v: Float[Array, "lat lon"]

    :returns: the magnitude
    :rtype: Float[Array, "lat lon"]
    """
    u = interpolate_south_west(u, axis=1, neuman=False)  # t point
    v = interpolate_south_west(v, axis=0, neuman=False)  # t point

    return jnp.sqrt(u ** 2 + v ** 2)


def compute_norm_vorticity(
        u: Float[Array, "lat lon"],
        v: Float[Array, "lat lon"],
        lat_u: Float[Array, "lat lon"],
        lon_u: Float[Array, "lat lon"],
        lat_v: Float[Array, "lat lon"],
        lon_v: Float[Array, "lat lon"],
        mask_u: Float[Array, "lat lon"] = None,
        mask_v: Float[Array, "lat lon"] = None
) -> Float[Array, "lat lon"]:
    """Computes the normalised relative vorticity of a velocity field

    :param u: U velocity value
    :type u: Float[Array, "lat lon"]
    :param v: V velocity value
    :type v: Float[Array, "lat lon"]
    :param lat_u: latitude at U points
    :type lat_u: Float[Array, "lat lon"]
    :param lon_u: longitude at U points
    :type lon_u: Float[Array, "lat lon"]
    :param lat_v: mask at U points
    :type lat_v: Float[Array, "lat lon"]
    :param lon_v: longitude at V points
    :type lon_v: Float[Array, "lat lon"]
    :param mask_u: mask at U points
    :type mask_u: Float[Array, "lat lon"]
    :param mask_v: mask at V points
    :type mask_v: Float[Array, "lat lon"]

    :returns: the normalised relative vorticity
    :rtype: Float[Array, "lat lon"]
    """
    if mask_u is not None and mask_v is not None:
        mask_f = mask_u + mask_v
    else:
        mask_f = mask_u if mask_u is not None else mask_v

    _, dy_u = compute_spatial_step(lat_u, lon_u)
    dx_v, _ = compute_spatial_step(lat_v, lon_v)
    f = compute_coriolis_factor(lat_u)  # u point

    dy_u = sanitise_data(dy_u, jnp.nan, mask_u)
    dx_v = sanitise_data(dx_v, jnp.nan, mask_v)
    f = sanitise_data(f, jnp.nan, mask_u)

    du_dy = compute_derivative_north_east(u, dy_u, axis=0)  # f point
    dv_dx = compute_derivative_north_east(v, dx_v, axis=1)  # f point
    f = interpolate_north_east(f, axis=0, neuman=False)  # f point

    w = (dv_dx - du_dy) / f
    w = sanitise_data(w, jnp.nan, mask_f)

    return w
