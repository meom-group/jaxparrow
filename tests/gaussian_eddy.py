import jax
import jax.numpy as jnp

from jaxparrow.tools import geometry, kinematics


def simulate_gaussian_eddy(
        r0: float,
        dxy: float,
        eta0: float,
        latitude: int
) -> [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
      jax.Array]:
    l0 = r0 * 2  # limit boundary impact
    xy = jnp.arange(0, l0, dxy)
    xy = jnp.concatenate((-xy[::-1][:-1], xy))
    X, Y = jnp.meshgrid(xy, xy)
    R = jnp.hypot(X, Y)
    dXY = jnp.ones_like(X) * dxy
    coriolis_factor = jnp.ones_like(R) * geometry.compute_coriolis_factor(latitude)  # noqa

    ssh = simulate_gaussian_ssh(r0, eta0, R)
    u_geos_t, v_geos_t = simulate_gaussian_geos(r0, X, Y, ssh, coriolis_factor)
    u_cyclo_t, v_cyclo_t = simulate_gaussian_cyclo(r0, jnp.arctan2(Y, X), u_geos_t, v_geos_t, coriolis_factor)

    mask = jax.numpy.full_like(ssh, 0, dtype=bool)  # no land in square oceans

    return X, Y, R, dXY, coriolis_factor, ssh, u_geos_t, v_geos_t, u_cyclo_t, v_cyclo_t, mask


def simulate_gaussian_ssh(r0: float, eta0: float, R: jax.Array) -> jax.Array:
    return eta0 * jnp.exp(-(R / r0)**2)


def simulate_gaussian_geos(
        r0: float,
        X: jax.Array,
        Y: jax.Array,
        ssh: jax.Array,
        coriolis_factor: jax.Array
) -> [jax.Array, jax.Array]:
    def f():
        return 2 * geometry.GRAVITY * ssh / (coriolis_factor * r0 ** 2)
    u_geos = Y * f()
    v_geos = -X * f()
    return u_geos, v_geos


def simulate_gaussian_cyclo(
        r0: float,
        theta: jax.Array,
        u_geos: jax.Array,
        v_geos: jax.Array,
        coriolis_factor: jax.Array
) -> [jax.Array, jax.Array]:
    def f():
        return azim_cyclo**2 / (r0 * coriolis_factor)
    azim_geos = kinematics.magnitude(u_geos, v_geos, interpolate=False)
    azim_cyclo = 2 * azim_geos / (1 + jnp.sqrt(1 + 4 * azim_geos / (coriolis_factor * r0)))
    u_cyclo = u_geos + jnp.sin(theta) * f()
    v_cyclo = v_geos - jnp.cos(theta) * f()
    return u_cyclo, v_cyclo


def compute_rmse(y: jax.Array, y_hat: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.nanmean((y - y_hat)**2))
