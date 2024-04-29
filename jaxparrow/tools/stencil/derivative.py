from jax import jit, lax, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float


@jit
def derivative(
        field: Float[Array, "field"],
        stencil_weights: Float[Array, "field stencil_width"],
        pad_left: bool
) -> Float[Array, "field"]:
    """
    Differentiates a 1d ``field``, using finite differences, and a stencil of width ``stencil_width``.
    Stencil weights are computed using Fornberg's algorithm [6]_.
    Applies `0` padding to the `left` or to the `right`.

    Parameters
    ----------
    field : Float[Array, "field"]
        Field to differentiate
    stencil_weights : Float[Array, "field width"]
        Spatial steps
    pad_left : bool
        Padding direction

    Returns
    -------
    field : Float[Array, "field"]
        Differentiated field
    """
    def nandot(arr1, arr2):
        return jnp.nansum(arr1 * arr2)

    stencil_width = stencil_weights.shape[1]
    stencil_half_width = stencil_width // 2

    padded_field = lax.cond(
        pad_left,
        lambda: jnp.pad(field, (stencil_half_width, stencil_half_width - 1)),
        lambda: jnp.pad(field, (stencil_half_width - 1, stencil_half_width))
    )
    stencil_idx = jnp.arange(field.size)[:, None] + jnp.arange(stencil_width)[None, :]
    field_stencil = padded_field[stencil_idx]

    field_derivative = vmap(nandot)(stencil_weights, field_stencil)

    field_derivative = jnp.where(jnp.isfinite(field), field_derivative, jnp.nan)

    return field_derivative
