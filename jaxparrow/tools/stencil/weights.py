from functools import partial

from jax import jit, lax, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar
import numpy as np

from .mask import mask


@jit
def _fornberg_loop(dx: Float[Array, "stencil"], x0: Float[Scalar, ""]) -> Float[Array, "stencil"]:
    """
    Non-vectorized version of Fornberg's algorithm for first order derivative only.
    https://doi.org/10.1137/S0036144596322507
    """
    n = len(dx)
    c1 = 1
    c4 = dx[0] - x0
    c = jnp.zeros((n, 2))
    c = c.at[0, 0].set(1)

    for i in range(1, n):
        c2 = 1
        c5 = c4
        c4 = dx[i] - x0

        for j in range(i):
            c3 = dx[i] - dx[j]
            c2 *= c3

            if j == i - 1:
                c = c.at[i, 1].set(c1 * (c[i - 1, 0] - c5 * c[i - 1, 1]) / c2)
                c = c.at[i, 0].set(-c1 * c5 * c[i - 1, 0] / c2)

            c = c.at[j, 1].set((c4 * c[j, 1] - c[j, 0]) / c3)
            c = c.at[j, 0].set(c4 * c[j, 0] / c3)

        c1 = c2

    c = c[:, -1]
    c = c.at[n // 2].set(c[n // 2] - jnp.sum(c))

    return c


@jit
def _fornberg(dx: Float[Array, "stencil"], x0: Float[Scalar, ""]) -> Float[Array, "stencil"]:
    """
    Vectorized version of Fornberg's algorithm for first order derivative only.
    This version is much faster to compile than the non-vectorized one. Running times are similar.
    https://doi.org/10.1093/imanum/draa006
    """
    n = len(dx)
    c = jnp.zeros((3, n))
    c = c.at[1, 0].set(1)

    a = dx[:, jnp.newaxis] - dx
    b = jnp.cumprod(jnp.hstack((jnp.ones((n, 1)), a)), axis=1)
    rm = jnp.tile(jnp.arange(3), (n - 1, 1)).T

    d0 = jnp.diag(b)
    d1 = d0[:-1] / d0[1:]

    for i in range(1, n):
        c = c.at[1:3, i].set(d1[i - 1] * (rm[:2, 0] * c[:2, i - 1] - (dx[i - 1] - x0) * c[1:3, i - 1]))
        c = c.at[1:3, :i].set(((dx[i] - x0) * c[1:3, :i] - rm[:2, :i] * c[:2, :i]) / (dx[i] - dx[:i]))

    c = c[2, :]
    c = c.at[n // 2].set(c[n // 2] - jnp.sum(c))

    return c


@partial(jit, static_argnames=("stencil_half_width", "n"))
def _weights(
        stencil_half_width: int,
        stencil_weights: Float[Array, "width stencil"],
        stencil_mask: Float[Array, "stencil"],
        dx: Float[Array, "stencil"],
        n: int
) -> Float[Array, "stencil width"]:
    """
    Calculate the stencil weights of every field point via Fornberg's algorithm,
    using the adequate stencil width (based on the point neighborhood).
    """
    def do_compute():
        padded_idx = jnp.arange(dx.size - 2 * n + 1)[:, None] + jnp.arange(2 * n)[None, :]
        dx_stencil = dx[padded_idx]
        x0_stencil = vmap(lambda arr: arr[arr.size // 2 - 1: arr.size // 2 + 1].mean())(dx_stencil)
        _stencil_weights = vmap(_fornberg)(dx_stencil, x0_stencil)
        return jnp.pad(_stencil_weights, ((0, 0), (stencil_half_width - n, stencil_half_width - n)))

    stencil_weights = jnp.where(
        stencil_mask.reshape((-1, 1)),
        do_compute(),
        stencil_weights
    )

    return stencil_weights


def weights(
        field: Float[Array, "field"],
        dx: Float[Array, "field"],
        stencil_width: int,
        pad_left: bool
) -> Float[Array, "field width"]:
    """
    Compute the stencil weights of every field point using Fornberg's algorithm.
    At the domain boundaries, or in the presence of land, the stencil width is decreased adequately,
    and weights are padded with zeros.

    Parameters
    ----------
    field : Float[Array, "field"]
        1d field used to infer land positions
    dx : Float[Array, "field"]
       1d grid of spatial steps
    stencil_width : int
        Width of the stencil
    pad_left : bool
        Padding direction.
        For example, following NEMO convention [1]_,
        interpolating from U to T points requires a `left` padding
        (the midpoint between $U_i$ and $U_{i+1}$ corresponds to $T_{i+1}$),
        and interpolating from T to U points a `right` padding
        (the midpoint between $T_i$ and $T_{i+1}$ corresponds to $U_i$)

    Returns
    -------
    stencil_weights : Float[Array, "field stencil_width"]
       Tensor of stencil weights for every ``field`` point
    """
    stencil_half_width = stencil_width // 2
    stencil_mask = mask(field, stencil_half_width, pad_left)

    stencil_weights = jnp.zeros((field.size, stencil_width))
    for n in np.arange(stencil_half_width, 0, -1):
        dx_padded = lax.cond(
            pad_left,
            lambda: jnp.pad(dx, (n, n - 1)),
            lambda: jnp.pad(dx, (n - 1, n))
        )

        stencil_weights = _weights(
            stencil_half_width, stencil_weights, stencil_mask[n - 1], dx_padded, n
        )

    return stencil_weights
