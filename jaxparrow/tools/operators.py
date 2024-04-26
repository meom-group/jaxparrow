from functools import partial
from typing import Literal

from jax import jit, lax
from jaxtyping import Array, Float

from .sanitize import handle_land_boundary
from .stencil.stencil import compute_stencil_derivative


@jit
def interpolation(
        field: Float[Array, "lat lon"],
        axis: Literal[0, 1],
        pad_left: bool
) -> Float[Array, "lat lon"]:
    """
    Interpolates the values of a ``field`` along a given ``axis`` (`0` for `lat`/`y`, `1` for `lon`/`x`),
    applying ``padding`` to the `left` (i.e. `West` if ``axis=1``, `South` if ``axis=0``) or
    to the `right` (i.e. `East` if ``axis=1``, `North` if ``axis=0``) of the domain,
    using nearest non interpolated value at the padded edge.

    Parameters
    ----------
    field : Float[Array, "lat lon"]
        Field to interpolate
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
    field : Float[Array, "lat lon"]
        Interpolated field
    """
    def do_interpolate(field_b, field_f):
        field_b, field_f = handle_land_boundary(field_b, field_f, pad_left)
        return (field_b + field_f) / 2

    def axis0():
        field_b, field_f = field[:-1, :], field[1:, :]
        midpoint_values = do_interpolate(field_b, field_f)
        return lax.cond(
            pad_left,
            lambda: field.at[1:, :].set(midpoint_values),
            lambda: field.at[:-1, :].set(midpoint_values)
        )

    def axis1():
        field_b, field_f = field[:, :-1], field[:, 1:]
        midpoint_values = do_interpolate(field_b, field_f)
        return lax.cond(
            pad_left,
            lambda: field.at[:, 1:].set(midpoint_values),
            lambda: field.at[:, :-1].set(midpoint_values)
        )

    field = lax.cond(axis == 0, axis0, axis1)

    return field


@partial(jit, static_argnames=("axis", "pad_left"))
def derivative(
        field: Float[Array, "lat lon"],
        stencil_weights: Float[Array, "2 2 lat lon stencil_width"],
        axis: Literal[0, 1],
        pad_left: bool
) -> Float[Array, "lat lon"]:
    """
    Differentiates a ``field`` using finite differences, along a given ``axis`` (`0` for `lat`/`y`, `1` for `lon`/'x').
    Uses stencil of width ``stencil_width`` to compute the derivative of the field, as advised by Arbic et al. _[5].
    Applies 0 ``padding`` to the `left` (i.e. `West` if ``axis=1``, `South` if ``axis=0``) or
    to the `right` (i.e. `East` if ``axis=1``, `North` if ``axis=0``) of the domain.

    Parameters
    ----------
    field : Float[Array, "lat lon"]
        Field to differentiate
    stencil_weights : Float[Array, "2 2 lat lon stencil_width"]
        Stencil weights of every ``field`` point in both directions and for both padding
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
    field : Float[Array, "lat lon"]
        Differentiated field
    """
    field = compute_stencil_derivative(field, stencil_weights, axis, pad_left)

    return field
