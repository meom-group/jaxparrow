from typing import Literal

from jax import jit, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import compute_spatial_step
from .derivative import derivative
from .weights import weights

#: Default stencil width, recommended value in Arbic et al. (2012) _[5] for 1/4° AVISO grid
STENCIL_WIDTH = 6


def compute_stencil_weights(
        field: Float[Array, "lat lon"],
        lat: Float[Array, "lat lon"] = None,
        lon: Float[Array, "lat lon"] = None,
        stencil_width: int = STENCIL_WIDTH,
        dxy: Float[Array, "lat lon"] = None
) -> Float[Array, "2 2 lat lon stencil_width"]:
    stencil_weights = jnp.zeros((2, 2, *field.shape, stencil_width))

    # computes spatial steps and apply cumsum to get steps relative to "origin"
    if dxy is None:
        dx, dy = compute_spatial_step(lat, lon)
    else:
        dx, dy = dxy, dxy
    dx = jnp.cumsum(dx, axis=1)
    dy = jnp.cumsum(dy, axis=0)

    # computes weights for the different padding and axis direction configurations
    for pad_left in (False, True):
        for axis, dxx in zip((0, 1), (dy, dx)):
            stencil_weights_i = vmap(
                weights,
                in_axes=(1 - axis, 1 - axis, None, None), out_axes=1 - axis
            )(field, dxx, stencil_width, pad_left)
            stencil_weights = stencil_weights.at[int(pad_left), axis].set(stencil_weights_i)

    return stencil_weights


def compute_stencil_derivative(
        field: Float[Array, "lat lon"],
        stencil_weights: Float[Array, "2 2 lat lon stencil_width"],
        axis: Literal[0, 1],
        pad_left: bool
) -> Float[Array, "lat lon"]:
    stencil_derivative = vmap(
        derivative,
        in_axes=(1 - axis, 1 - axis, None), out_axes=1 - axis
    )(field, stencil_weights[int(pad_left), axis, :], pad_left)

    return stencil_derivative
