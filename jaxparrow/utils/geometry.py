import jax
import jax.numpy as jnp
from jaxtyping import Float

from .operators import interpolation


#: Approximate earth angular speed
EARTH_ANG_SPEED = 7.292115e-5
#: Approximate earth radius
EARTH_RADIUS = 6370e3
#: Approximate gravity
GRAVITY = 9.81
#: Degrees / radians conversion factor
P0 = jnp.pi / 180


def compute_spatial_step(
    lat: Float[jax.Array, "lat lon"],
    lon: Float[jax.Array, "lat lon"]
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]]:
    """
    Computes the spatial steps of a grid (in meters) along `x` and `y`.

    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of `cos` and `arccos`
    functions to avoid truncation issues.

    Parameters
    ----------
    lat : Float[jax.Array, "lat lon"]
        Latitude grid
    lon : Float[jax.Array, "lat lon"]
        Longitude grid

    Returns
    -------
    dx : Float[jax.Array, "lat lon"]
        Spatial steps in meters along `x`
    dy : Float[jax.Array, "lat lon"]
        Spatial steps in meters along `y`
    """
    def haversine_distance(lat1, lat2, lon1, lon2):
        # convert to radians
        lat1_rad = jnp.radians(lat1)
        lat2_rad = jnp.radians(lat2)

        # difference in radians; normalize lon diff to [-180, 180] before radians to handle dateline
        dlon = lon2 - lon1
        dlon = (dlon + 180.0) % 360.0 - 180.0   # now in [-180,180]
        dlon_rad = jnp.radians(dlon)

        dlat_rad = jnp.radians(lat2 - lat1)

        a = jnp.sin(dlat_rad / 2.0) ** 2 + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * (jnp.sin(dlon_rad / 2.0) ** 2)
        c = 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))
        d = EARTH_RADIUS * c
        return d
    
    dx, dy = jnp.zeros_like(lat), jnp.zeros_like(lat)
    # dx
    dx = dx.at[:, :-1].set(haversine_distance(lat[:, :-1], lat[:, 1:], lon[:, :-1], lon[:, 1:]))
    dx = dx.at[:, -1].set(dx[:, -2])
    # dy
    dy = dy.at[:-1, :].set(haversine_distance(lat[:-1, :], lat[1:, :], lon[:-1, :], lon[1:, :]))
    dy = dy.at[-1, :].set(dy[-2, :])

    return dx, dy


def compute_coriolis_factor(
    lat: Float[jax.Array, "lat lon"]
) -> Float[jax.Array, "lat lon"]:
    """
    Computes the Coriolis factor from a latitude grid.

    Parameters
    ----------
    lat : Float[jax.Array, "lat lon"]
        Latitudes grid

    Returns
    -------
    cf : Float[jax.Array, "lat lon"]
        Coriolis factor grid
    """
    return 2 * EARTH_ANG_SPEED * jnp.sin(lat * P0)


def compute_grid_angle(
    lat: Float[jax.Array, "lat lon"],
    lon: Float[jax.Array, "lat lon"]
) -> Float[jax.Array, "lat lon"]:
    """
    Computes the local angle of the grid i-axis (axis=1) relative to geographic east.

    For curvilinear grids (e.g., SWOT swaths, tripolar grids), the grid axes are not aligned
    with geographic east-west/north-south directions. This function computes the rotation angle
    needed to transform gradients from grid coordinates to geographic coordinates.

    The angle is measured counterclockwise from geographic east to the grid i-direction.

    Parameters
    ----------
    lat : Float[jax.Array, "lat lon"]
        Latitude grid
    lon : Float[jax.Array, "lat lon"]
        Longitude grid

    Returns
    -------
    angle : Float[jax.Array, "lat lon"]
        Rotation angle in radians, measured counterclockwise from geographic east
        to the grid i-direction. Range is [-pi, pi].

    Notes
    -----
    The angle is computed using the initial bearing formula between adjacent grid points
    along the i-axis (axis=1). The formula computes the azimuth from north (clockwise positive),
    which is then converted to angle from east (counterclockwise positive).

    For orthogonal grids, the j-axis direction is at angle + pi/2.
    """
    # Use central differences where possible, forward/backward at boundaries
    lat_rad = jnp.radians(lat)

    # Compute differences in longitude (handling wraparound)
    dlon = jnp.zeros_like(lon)
    dlon = dlon.at[:, 1:-1].set(lon[:, 2:] - lon[:, :-2])  # central diff
    dlon = dlon.at[:, 0].set(lon[:, 1] - lon[:, 0])  # forward diff at left
    dlon = dlon.at[:, -1].set(lon[:, -1] - lon[:, -2])  # backward diff at right

    # Normalize to [-180, 180]
    dlon = (dlon + 180.0) % 360.0 - 180.0
    dlon_rad = jnp.radians(dlon)

    # Compute latitude at neighboring points for bearing calculation
    lat1_rad = jnp.zeros_like(lat_rad)
    lat1_rad = lat1_rad.at[:, 1:-1].set(lat_rad[:, :-2])
    lat1_rad = lat1_rad.at[:, 0].set(lat_rad[:, 0])
    lat1_rad = lat1_rad.at[:, -1].set(lat_rad[:, -2])

    lat2_rad = jnp.zeros_like(lat_rad)
    lat2_rad = lat2_rad.at[:, 1:-1].set(lat_rad[:, 2:])
    lat2_rad = lat2_rad.at[:, 0].set(lat_rad[:, 1])
    lat2_rad = lat2_rad.at[:, -1].set(lat_rad[:, -1])

    # Initial bearing formula: bearing from point 1 to point 2
    # bearing = atan2(sin(dlon)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon))
    # This gives bearing measured clockwise from north
    x = jnp.sin(dlon_rad) * jnp.cos(lat2_rad)
    y = jnp.cos(lat1_rad) * jnp.sin(lat2_rad) - jnp.sin(lat1_rad) * jnp.cos(lat2_rad) * jnp.cos(dlon_rad)
    bearing = jnp.arctan2(x, y)  # radians, clockwise from north

    # Convert bearing (clockwise from north) to angle (counterclockwise from east)
    # If bearing = 0 (north), angle = pi/2
    # If bearing = pi/2 (east), angle = 0
    # angle = pi/2 - bearing
    angle = jnp.pi / 2 - bearing

    return angle


def compute_uv_grids(
    lat_t: Float[jax.Array, "lat lon"],
    lon_t: Float[jax.Array, "lat lon"]
) -> tuple[
    Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]
]:
    """
    Computes the U and V grids associated to a T grid following NEMO convention.

    Parameters
    ----------
    lat_t : Float[jax.Array, "lat lon"]
        Latitudes of the T grid
    lon_t : Float[jax.Array, "lat lon"]
        Longitudes of the T grid

    Returns
    -------
    lat_u : Float[jax.Array, "lat lon"]
        Latitudes of the U grid
    lon_u : Float[jax.Array, "lat lon"]
        Longitudes of the U grid
    lat_v : Float[jax.Array, "lat lon"]
        Latitudes of the V grid
    lon_v : Float[jax.Array, "lat lon"]
        Longitudes of the V grid
    """
    lat_mask = jnp.zeros_like(lat_t, dtype=bool)
    lon_mask = jnp.zeros_like(lon_t, dtype=bool)

    lat_u = interpolation(lat_t, lat_mask, axis=1, padding="right")
    lat_u = lat_u.at[:, -1].set(2 * lat_t[:, -1] - lat_t[:, -2])
    lon_u = interpolation(lon_t, lon_mask, axis=1, padding="right")
    lon_u = lon_u.at[:, -1].set(2 * lon_t[:, -1] - lon_t[:, -2])

    lat_v = interpolation(lat_t, lat_mask, axis=0, padding="right")
    lat_v = lat_v.at[-1, :].set(2 * lat_t[-1, :] - lat_t[-2, :])
    lon_v = interpolation(lon_t, lon_mask, axis=0, padding="right")
    lon_v = lon_v.at[-1, :].set(2 * lon_t[-1, :] - lon_t[-2, :])

    return lat_u, lon_u, lat_v, lon_v
