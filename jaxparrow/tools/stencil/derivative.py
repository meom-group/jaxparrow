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
    Stencil weights are computed using Fornberg's algorithm _[6].
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
    stencil_width = stencil_weights.shape[1]
    stencil_half_width = stencil_width // 2

    padded_field = lax.cond(
        pad_left,
        lambda: jnp.pad(field, (stencil_half_width, stencil_half_width - 1)),
        lambda: jnp.pad(field, (stencil_half_width - 1, stencil_half_width))
    )
    padded_idx = jnp.arange(padded_field.size - stencil_width + 1)[:, None] + jnp.arange(stencil_width)[None, :]
    field_stencil = padded_field[padded_idx]

    field_derivative = vmap(jnp.dot)(stencil_weights, field_stencil)

    return field_derivative
