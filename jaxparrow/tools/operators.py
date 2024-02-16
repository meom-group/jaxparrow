from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float


def interpolation(
        field: Float[Array, "lat lon"],
        axis: Literal[0, 1],
        padding: Literal["left", "right"]
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
    padding : Literal["left", "right"]
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
    if axis == 0:
        midpoint_values = 0.5 * (field[:-1, :] + field[1:, :])
        if padding == "left":
            field = field.at[1:, :].set(midpoint_values)
        else:  # padding == "right"
            field = field.at[:-1, :].set(midpoint_values)
    else:  # axis == 1
        midpoint_values = 0.5 * (field[:, :-1] + field[:, 1:])
        if padding == "left":
            field = field.at[:, 1:].set(midpoint_values)
        else:
            field = field.at[:, :-1].set(midpoint_values)
    return field


def derivative(
        field: Float[Array, "lat lon"],
        dxy: Float[Array, "lat lon"],
        axis: Literal[0, 1],
        padding: Literal["left", "right"]
) -> Float[Array, "lat lon"]:
    """
    Differentiates a ``field``, using finite differences, along a given ``axis`` (`0` for `lat`/`y`, `1` for `lon`/'x'),
    applying ``padding`` to the `left` (i.e. `West` if ``axis=1``, `South` if ``axis=0``) or
    to the `right` (i.e. `East` if ``axis=1``, `North` if ``axis=0``) of the domain,
    using nearest derivative value at the padded edge.

    Parameters
    ----------
    field : Float[Array, "lat lon"]
        Field to differentiate
    dxy : Float[Array, "lat lon"]
        Spatial steps
    axis : Literal[0, 1]
        Axis along which interpolation is performed
    padding : Literal["left", "right"]
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
    if axis == 0:
        midpoint_values = field[1:, :] - field[:-1, :]
        if padding == "left":
            pad_width = ((1, 0), (0, 0))
        else:  # padding == "right"
            pad_width = ((0, 1), (0, 0))
    else:  # axis == 1
        midpoint_values = field[:, 1:] - field[:, :-1]
        if padding == "left":
            pad_width = ((0, 0), (1, 0))
        else:
            pad_width = ((0, 0), (0, 1))
    field = jnp.pad(midpoint_values, pad_width=pad_width, mode="edge") / dxy
    return field
