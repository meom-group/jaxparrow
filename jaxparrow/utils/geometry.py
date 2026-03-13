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


def grid_metrics(
    lat: Float[jax.Array, "y x"], lon: Float[jax.Array, "y x"]
) -> tuple[
    Float[jax.Array, "y x"], 
    Float[jax.Array, "y x"], 
    Float[jax.Array, "y x"], 
    Float[jax.Array, "y x"], 
    Float[jax.Array, "y x"]
]:
    """
    Computes the physical displacement vectors and the Jacobian associated with one grid-index step, 
    used to transform derivatives from grid coordinates to geographic coordinates.

    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of `cos` and `arccos`
    functions to avoid truncation issues.

    Parameters
    ----------
    lat : Float[jax.Array, "y x"]
        Latitude grid
    lon : Float[jax.Array, "y x"]
        Longitude grid

    Returns
    -------
    dx_e : Float[jax.Array, "y x"]
        Eastward displacement associated with one step in the x-index direction
    dx_n : Float[jax.Array, "y x"]
        Northward displacement associated with one step in the x-index direction
    dy_e : Float[jax.Array, "y x"]
        Eastward displacement associated with one step in the y-index direction
    dy_n : Float[jax.Array, "y x"]
        Northward displacement associated with one step in the y-index direction
    J : Float[jax.Array, "y x"]
        Jacobian of the transformation from grid to geographic coordinates
    """
    def displacement_components(lat1, lat2, lon1, lon2):
        # convert to radians
        lat1_rad = jnp.radians(lat1)
        lat2_rad = jnp.radians(lat2)

        # difference in radians; normalize lon diff to [-180, 180] before radians to handle dateline
        dlon = lon2 - lon1
        dlon = (dlon + 180.0) % 360.0 - 180.0   # now in [-180,180]
        dlon_rad = jnp.radians(dlon)

        dlat_rad = jnp.radians(lat2 - lat1)

        # haversine distance
        a = jnp.sin(dlat_rad / 2.0) ** 2 + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * (jnp.sin(dlon_rad / 2.0) ** 2)
        c = 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))
        d = EARTH_RADIUS * c

        # bearing from point 1 to point 2
        theta = jnp.arctan2(
            jnp.sin(dlon_rad) * jnp.cos(lat2_rad),
            jnp.cos(lat1_rad) * jnp.sin(lat2_rad) - jnp.sin(lat1_rad) * jnp.cos(lat2_rad) * jnp.cos(dlon_rad)
        )

        de = d * jnp.sin(theta)
        dn = d * jnp.cos(theta)

        return de, dn

    # physical displacements
    dy_e, dy_n = displacement_components(lat[:-1, :], lat[1:, :], lon[:-1, :], lon[1:, :])
    dx_e, dx_n = displacement_components(lat[:, :-1], lat[:, 1:], lon[:, :-1], lon[:, 1:])

    dx_e = jnp.pad(dx_e, ((0, 0), (0, 1)), mode="edge")
    dx_n = jnp.pad(dx_n, ((0, 0), (0, 1)), mode="edge")
    dy_e = jnp.pad(dy_e, ((0, 1), (0, 0)), mode="edge")
    dy_n = jnp.pad(dy_n, ((0, 1), (0, 0)), mode="edge")

    # jacobian
    J = dx_e * dy_n - dx_n * dy_e

    return dx_e, dx_n, dy_e, dy_n, J


def coriolis_factor(lat: Float[jax.Array, "y x"]) -> Float[jax.Array, "y x"]:
    """
    Computes the Coriolis factor from a latitude grid.

    Parameters
    ----------
    lat : Float[jax.Array, "y x"]
        Latitudes grid

    Returns
    -------
    cf : Float[jax.Array, "y x"]
        Coriolis factor grid
    """
    return 2 * EARTH_ANG_SPEED * jnp.sin((jnp.radians(lat)))


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
    lat_u = interpolation(lat_t, axis=1, padding="right")
    lat_u = lat_u.at[:, -1].set(2 * lat_t[:, -1] - lat_t[:, -2])
    lon_u = interpolation(lon_t, axis=1, padding="right")
    lon_u = lon_u.at[:, -1].set(2 * lon_t[:, -1] - lon_t[:, -2])

    lat_v = interpolation(lat_t, axis=0, padding="right")
    lat_v = lat_v.at[-1, :].set(2 * lat_t[-1, :] - lat_t[-2, :])
    lon_v = interpolation(lon_t, axis=0, padding="right")
    lon_v = lon_v.at[-1, :].set(2 * lon_t[-1, :] - lon_t[-2, :])

    return lat_u, lon_u, lat_v, lon_v
