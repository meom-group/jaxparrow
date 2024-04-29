from functools import partial
from typing import Literal

from jax import jit, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..geometry import compute_spatial_step
from .derivative import derivative
from .weights import weights

#: Default stencil width, recommended value in Arbic et al. (2012) [5]_ for 1/4° AVISO grid
STENCIL_WIDTH = 6


def compute_stencil_weights(
        field: Float[Array, "lat lon"],
        lat: Float[Array, "lat lon"] = None,
        lon: Float[Array, "lat lon"] = None,
        dx: Float[Array, "lat lon"] = None,
        dy: Float[Array, "lat lon"] = None,
        stencil_width: int = STENCIL_WIDTH
) -> Float[Array, "2 2 lat lon stencil_width"]:
    """
    Compute the stencil weights of every field point using Fornberg's algorithm [6]_.
    At every point, four combinations of stencil weights are computed in both directions and padding.
    At the domain boundaries, or in the presence of land, the stencil width is decreased adequately,
    and weights are padded with zeros.

    Parameters
    ----------
    field : Float[Array, "lat lon"]
        2d field used to infer land positions
    lat : Float[Array, "lat lon"], optional
        2d grid of latitude points, used to compute spatial steps if ``dx`` or ``dy`` is not provided.

        Defaults to `None`
    lon : Float[Array, "lat lon"], optional
        2d grid of longitude points, used to compute spatial steps if ``dx`` or ``dy`` is not provided.

        Defaults to `None`
    dx : Float[Array, "lat lon"], optional
        2d grid of spatial steps along the x-direction.

        Defaults to `None`
    dy : Float[Array, "lat lon"], optional
        2d grid of spatial steps along the y-direction.

        Defaults to `None`
    stencil_width : int, optional
        Width of the stencil.
        As we use C-grids, it should be an even integer.

        Defaults to ``stencil_width``

    Returns
    -------
    stencil_weights : Float[Array, "2 2 lat lon stencil_width"]
        Tensor of stencil weights for every ``field`` point in both directions and for both padding
    """
    stencil_weights = jnp.zeros((2, 2, *field.shape, stencil_width))

    # computes spatial steps and apply cumsum to get steps relative to "origin"
    if dx is None or dy is None:
        dx, dy = compute_spatial_step(lat, lon)
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


@partial(jit, static_argnames=("axis", "pad_left"))
def compute_stencil_derivative(
        field: Float[Array, "lat lon"],
        stencil_weights: Float[Array, "2 2 lat lon stencil_width"],
        axis: Literal[0, 1],
        pad_left: bool
) -> Float[Array, "lat lon"]:
    """
    Compute the stencil weights of every field point using Fornberg's algorithm [6]_.
    At every point, four combinations of stencil weights are computed in both directions and paddings.
    At the domain boundaries, or in the presence of land, the stencil width is decreased adequately,
    and weights are padded with zeros.

    Parameters
    ----------
    field : Float[Array, "lat lon"]
        2d field to differentiate
    stencil_weights : Float[Array, "2 2 lat lon stencil_width"]
        Tensor of stencil weights of every ``field`` point in both directions and for both padding
    axis : Literal[0, 1]
        Axis along which interpolation is performed
    pad_left : bool
        Padding direction.
        For example, following NEMO convention [1]_,
        interpolating from U to T points requires a `left` padding
        (the midpoint between $U_i$ and $U_{i+1}$ corresponds to $T_{i+1}$),
        and interpolating from T to U points a `right` padding
        (the midpoint between $T_i$ and $T_{i+1}$ corresponds to $U_i$)

    Returns
    -------
    field_derivative : Float[Array, "lat lon"]
        Differentiated 2d field
    """
    field_derivative = vmap(
        derivative,
        in_axes=(1 - axis, 1 - axis, None), out_axes=1 - axis
    )(field, stencil_weights[int(pad_left), axis, :], pad_left)

    return field_derivative
