import jax
import jax.numpy as jnp

from jaxparrow.utils import geometry


def simulate_gaussian_eddy(
    R0: float,
    eta0: float,
) -> tuple[
    jax.Array, 
    jax.Array, 
    jax.Array, 
    jax.Array, 
    jax.Array, 
    jax.Array, 
    jax.Array, 
    jax.Array, 
    jax.Array, 
    jax.Array, 
    jax.Array
]:
    # in the Alboran Sea
    latlon0 = (35.92744, -4.03238)
    # grid resolution (in °)
    dlatlon = 0.01

    # domain size (in °)
    hw_deg = 2 * R0 / 111e3
    # create lat lon grids
    lat_1d = latlon0[0] + jnp.arange(-hw_deg, hw_deg + dlatlon, dlatlon)
    lon_1d = latlon0[1] + jnp.arange(-hw_deg, hw_deg + dlatlon, dlatlon)
    lon, lat = jnp.meshgrid(lon_1d, lat_1d)

    # grid distance to latlon0 (in m)
    R = haversine_distance(lat, latlon0[0], lon, latlon0[1])
    # distace along x and y axes (in m)
    X = haversine_distance(lat, lat, lon, latlon0[1])
    Y = haversine_distance(lat, latlon0[0], lon, lon)
    # correct signs
    X = jnp.where(lon < latlon0[1], -X, X)
    Y = jnp.where(lat < latlon0[0], -Y, Y)

    coriolis_factor = jnp.ones_like(R) * geometry.coriolis_factor(lat)

    ssh = simulate_gaussian_ssh(R0, eta0, R)
    ug, vg = simulate_gaussian_geos(R0, X, Y, ssh, coriolis_factor)
    ucg, vcg = simulate_gaussian_cyclo(R0, X, Y, R, ssh, coriolis_factor)

    land_mask = jax.numpy.full_like(ssh, 0, dtype=bool)  # no land in square oceans

    return lat, lon, ssh, ug, vg, ucg, vcg, land_mask


def haversine_distance(lat, lat0, lon, lon0):
     # convert to radians
    lat_rad = jnp.radians(lat)
    lat0_rad = jnp.radians(lat0)

    # difference in radians; normalize lon diff to [-180, 180] before radians to handle dateline
    dlon = lon - lon0
    dlon = (dlon + 180.0) % 360.0 - 180.0   # now in [-180,180]
    dlon_rad = jnp.radians(dlon)

    dlat_rad = jnp.radians(lat - lat0)

    # haversine distance
    a = jnp.sin(dlat_rad / 2.0) ** 2 + jnp.cos(lat0_rad) * jnp.cos(lat_rad) * (jnp.sin(dlon_rad / 2.0) ** 2)
    c = 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))
    d = geometry.EARTH_RADIUS * c

    return d


def simulate_gaussian_ssh(R0: float, eta0: float, r: jax.Array) -> jax.Array:
    return eta0 * jnp.exp(-(r / R0)**2)


def simulate_gaussian_geos(
    R0: float,
    X: jax.Array,
    Y: jax.Array,
    ssh: jax.Array,
    coriolis_factor: jax.Array
) -> tuple[jax.Array, jax.Array]:
    def f():
        return 2 * geometry.GRAVITY * ssh / (coriolis_factor * R0 ** 2)
    ug = Y * f()
    vg = -X * f()
    return ug, vg


def simulate_gaussian_cyclo(
    R0: float,
    X: jax.Array,
    Y: jax.Array,
    r: jax.Array,
    ssh: jax.Array,
    coriolis_factor: jax.Array
) -> tuple[jax.Array, jax.Array]:
    azim_geos = -(2 * geometry.GRAVITY * r * ssh / (coriolis_factor * R0 ** 2))
    azim_cyclo = 2 * azim_geos / (1 + jnp.sqrt(1 + 4 * azim_geos / (coriolis_factor * r)))
    ucg = -azim_cyclo * Y / r
    vcg = azim_cyclo * X / r
    return ucg, vcg
