from typing import Literal

from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .sanitize import handle_land_boundary


def interpolation(
        field: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"],
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
    mask : Float[Array, "lat lon"]
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
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
    def do_interpolate(field_b, field_f, mask_b, mask_f, pad_left):
        field_b, field_f = handle_land_boundary(field_b, field_f, mask_b, mask_f, pad_left)
        return 0.5 * (field_b + field_f)

    def axis0(pad_left):
        field_b, field_f = field[:-1, :], field[1:, :]
        mask_b, mask_f = mask[:-1, :], mask[1:, :]
        midpoint_values = do_interpolate(field_b, field_f, mask_b, mask_f, pad_left)

        arr = lax.cond(
            pad_left,
            lambda: field.at[1:, :].set(midpoint_values),
            lambda: field.at[:-1, :].set(midpoint_values)
        )

        return arr

    def axis1(pad_left):
        field_b, field_f = field[:, :-1], field[:, 1:]
        mask_b, mask_f = mask[:, :-1], mask[:, 1:]
        midpoint_values = do_interpolate(field_b, field_f, mask_b, mask_f, pad_left)

        arr = lax.cond(
            pad_left,
            lambda: field.at[:, 1:].set(midpoint_values),
            lambda: field.at[:, :-1].set(midpoint_values)
        )

        return arr

    field = lax.cond(
        axis == 0,
        lambda pad_left: axis0(pad_left),
        lambda pad_left: axis1(pad_left),
        padding == "left"
    )

    return field


def derivative(
        field: Float[Array, "lat lon"],
        dxy: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"],
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
    mask : Float[Array, "lat lon"]
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
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
    def do_differentiate(field_b, field_f, mask_b, mask_f, pad_left):
        field_b, field_f = handle_land_boundary(field_b, field_f, mask_b, mask_f, pad_left)
        return field_f - field_b

    def axis0(pad_left):
        field_b, field_f = field[:-1, :], field[1:, :]
        mask_b, mask_f = mask[:-1, :], mask[1:, :]
        midpoint_values = do_differentiate(field_b, field_f, mask_b, mask_f, pad_left)

        arr = lax.cond(
            pad_left,
            lambda: jnp.pad(midpoint_values, pad_width=((1, 0), (0, 0)), mode="edge"),
            lambda: jnp.pad(midpoint_values, pad_width=((0, 1), (0, 0)), mode="edge")
        )

        return arr

    def axis1(pad_left):
        field_b, field_f = field[:, :-1], field[:, 1:]
        mask_b, mask_f = mask[:, :-1], mask[:, 1:]
        midpoint_values = do_differentiate(field_b, field_f, mask_b, mask_f, pad_left)

        arr = lax.cond(
            pad_left,
            lambda: jnp.pad(midpoint_values, pad_width=((0, 0), (1, 0)), mode="edge"),
            lambda: jnp.pad(midpoint_values, pad_width=((0, 0), (0, 1)), mode="edge")
        )

        return arr

    field = lax.cond(
        axis == 0,
        lambda pad_left: axis0(pad_left),
        lambda pad_left: axis1(pad_left),
        padding == "left"
    )

    return field / dxy
