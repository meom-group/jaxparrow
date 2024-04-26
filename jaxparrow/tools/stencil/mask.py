from functools import partial

from jax import lax, jit
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
import numpy as np


def _land_boundary(field: Float[Array, "field"], pad_left: bool, n: int) -> Bool[Array, "field"]:
    rolling_sum = lax.cond(
        pad_left,
        lambda: lax.reduce_window(field, np.zeros(()), lax.add, (2 * n,), (1,), [(n, n - 1)]),
        lambda: lax.reduce_window(field, np.zeros(()), lax.add, (2 * n,), (1,), [(n - 1, n)])
    )

    return jnp.isfinite(rolling_sum)


def _domain_boundary(stencil_mask: Bool[Array, "field"], pad_left: bool, n: int) -> Bool[Array, "field"]:
    # domain boundary: restrict to start+n and end-n (+- 1 depending on padding)
    stencil_mask = lax.cond(
        pad_left,
        lambda: stencil_mask.at[n:stencil_mask.size - n + 1].set(True),
        lambda: stencil_mask.at[n - 1:-n].set(True)
    )

    return stencil_mask


def _compute_mask(field: Float[Array, "field"], pad_left: bool, n: int) -> Bool[Array, "field"]:
    stencil_mask = jnp.zeros_like(field, dtype=jnp.bool)
    stencil_mask = _domain_boundary(stencil_mask, pad_left, n)
    stencil_mask = stencil_mask & _land_boundary(field, pad_left, n)

    return stencil_mask


@partial(jit, static_argnames="n")
def _merge_mask_across_widths(
        field: Float[Array, "field"],
        pad_left: bool,
        cum_mask: Float[Array, "field"],
        n: int
) -> (Bool[Array, "field"], Bool[Array, "field"]):
    mask_n = _compute_mask(field, pad_left, n)

    mask_n = mask_n & ~cum_mask  # do not consider indexes already available at higher orders
    cum_mask = cum_mask | mask_n  # accumulate indexes already eligible at higher orders

    return cum_mask, mask_n


def mask(
        field: Float[Array, "field"],
        stencil_half_width: int,
        pad_left: bool
) -> Bool[Array, "stencil_order field"]:
    """
    Compute the stencil's mask,
    i.e., indicates for each point of the ``field`` if the stencil of half-width ``half_width`` can be applied
    based on the domain's boundaries and the presence of land.
    """
    cum_mask = jnp.zeros_like(field, dtype=jnp.bool)
    stencil_mask = jnp.zeros((stencil_half_width, field.size), dtype=jnp.bool)
    for n in np.arange(stencil_half_width, 0, -1):
        cum_mask, mask_n = _merge_mask_across_widths(field, pad_left, cum_mask, n)
        stencil_mask = stencil_mask.at[n - 1, :].set(mask_n)

    return stencil_mask
