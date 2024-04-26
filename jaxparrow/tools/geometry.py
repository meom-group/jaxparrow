from jax import jit
import jax.numpy as jnp
from jaxtyping import Array, Float


#: Approximate earth angular speed
EARTH_ANG_SPEED = 7.292115e-5
#: Approximate earth radius
EARTH_RADIUS = 6370e3
#: Approximate gravity
GRAVITY = 9.81
#: Degrees / radians conversion factor
P0 = jnp.pi / 180


@jit
def compute_spatial_step(
        lat: Float[Array, "lat lon"],
        lon: Float[Array, "lat lon"]
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Computes the spatial steps of a grid (in meters) along `x` and `y`.

    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of `cos` and `arccos`
    functions to avoid truncation issues.

    Parameters
    ----------
    lat : Float[Array, "lat lon"]
        Latitude grid
    lon : Float[Array, "lat lon"]
        Longitude grid

    Returns
    -------
    dx : Float[Array, "lat lon"]
        Spatial steps in meters along `x`
    dy : Float[Array, "lat lon"]
        Spatial steps in meters along `y`
    """
    def sphere_distance(lat_s, lat_e, lon_s, lon_e):
        dlat, dlon = P0 * (lat_e - lat_s), P0 * (lon_e - lon_s)
        return EARTH_RADIUS * jnp.sqrt(dlat ** 2 + jnp.cos(P0 * lat_s) * jnp.cos(P0 * lat_e) * dlon ** 2)

    dx, dy = jnp.zeros_like(lat), jnp.zeros_like(lat)
    # dx
    dx = dx.at[:, :-1].set(sphere_distance(lat[:, :-1], lat[:, 1:], lon[:, :-1], lon[:, 1:]))
    dx = dx.at[:, -1].set(dx[:, -2])
    # dy
    dy = dy.at[:-1, :].set(sphere_distance(lat[:-1, :], lat[1:, :], lon[:-1, :], lon[1:, :]))
    dy = dy.at[-1, :].set(dy[-2, :])
    return dx, dy


@jit
def compute_coriolis_factor(
        lat: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    """
    Computes the Coriolis factor from a latitude grid.

    Parameters
    ----------
    lat : Float[Array, "lat lon"]
        Latitudes grid

    Returns
    -------
    cf : Float[Array, "lat lon"]
        Coriolis factor grid
    """
    return 2 * EARTH_ANG_SPEED * jnp.sin(lat * P0)


@jit
def compute_uv_grids(
        lat_t: Float[Array, "lat lon"],
        lon_t: Float[Array, "lat lon"]
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Computes the U and V grids associated to a T grid following NEMO convention [1]_.

    Parameters
    ----------
    lat_t : Float[Array, "lat lon"]
        Latitudes of the T grid
    lon_t : Float[Array, "lat lon"]
        Longitudes of the T grid

    Returns
    -------
    lat_u : Float[Array, "lat lon"]
        Latitudes of the U grid
    lon_u : Float[Array, "lat lon"]
        Longitudes of the U grid
    lat_v : Float[Array, "lat lon"]
        Latitudes of the V grid
    lon_v : Float[Array, "lat lon"]
        Longitudes of the V grid
    """
    from .operators import interpolation

    lat_u = interpolation(lat_t, axis=1, pad_left=False)
    lat_u = lat_u.at[:, -1].set(2 * lat_t[:, -1] - lat_t[:, -2])
    lon_u = interpolation(lon_t, axis=1, pad_left=False)
    lon_u = lon_u.at[:, -1].set(2 * lon_t[:, -1] - lon_t[:, -2])

    lat_v = interpolation(lat_t, axis=0, pad_left=False)
    lat_v = lat_v.at[-1, :].set(2 * lat_t[-1, :] - lat_t[-2, :])
    lon_v = interpolation(lon_t, axis=0, pad_left=False)
    lon_v = lon_v.at[-1, :].set(2 * lon_t[-1, :] - lon_t[-2, :])

    return lat_u, lon_u, lat_v, lon_v
