import jax
import jax.numpy as jnp

from jaxparrow.utils import geometry


def simulate_gaussian_eddy(
    R0: float,
    dxy: float,
    eta0: float,
    latitude: int
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
    l0 = R0 * 2  # limit boundary impact
    xy = jnp.arange(0, l0, dxy)
    xy = jnp.concatenate((-xy[::-1][:-1], xy))
    X, Y = jnp.meshgrid(xy, xy)
    R = jnp.hypot(X, Y)
    dXY = jnp.ones_like(X) * dxy
    coriolis_factor = jnp.ones_like(R) * geometry.compute_coriolis_factor(latitude)  # noqa

    ssh = simulate_gaussian_ssh(R0, eta0, R)
    ug_t, vg_t = simulate_gaussian_geos(R0, X, Y, ssh, coriolis_factor)
    ucg_t, vcg_t = simulate_gaussian_cyclo(R0, X, Y, R, ssh, coriolis_factor)

    mask = jax.numpy.full_like(ssh, 0, dtype=bool)  # no land in square oceans

    return X, Y, R, dXY, coriolis_factor, ssh, ug_t, vg_t, ucg_t, vcg_t, mask


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


def compute_rmse(y: jax.Array, y_hat: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.nanmean((y - y_hat)**2))
