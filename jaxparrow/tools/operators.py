from typing import Literal

from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .sanitize import handle_land_boundary


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
    def axis0(arr, pad_left):
        arr_b, arr_f = arr[:-1, :], arr[1:, :]
        arr_b, arr_f = handle_land_boundary(arr_b, arr_f, pad_left)
        midpoint_values = 0.5 * (arr_b + arr_f)
        arr = lax.cond(
            pad_left,
            lambda operands: operands[0].at[1:, :].set(operands[1]),
            lambda operands: operands[0].at[:-1, :].set(operands[1]),
            (arr, midpoint_values)
        )
        return arr

    def axis1(arr, pad_left):
        arr_b, arr_f = arr[:, :-1], arr[:, 1:]
        arr_b, arr_f = handle_land_boundary(arr_b, arr_f, pad_left)
        midpoint_values = 0.5 * (arr_b + arr_f)
        arr = lax.cond(
            pad_left,
            lambda operands: operands[0].at[:, 1:].set(operands[1]),
            lambda operands: operands[0].at[:, :-1].set(operands[1]),
            (arr, midpoint_values)
        )
        return arr

    field = lax.cond(
        axis == 0,
        lambda operands: axis0(*operands), lambda operands: axis1(*operands),
        (field, padding == "left")
    )

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
    def do_interpolate(field_b, field_f, pad_left):
        field_b, field_f = handle_land_boundary(field_b, field_f, pad_left)
        return field_f - field_b

    def axis0(_field, pad_left):
        def pad_left_fn(_field_b, _field_f):
            midpoint_values = do_interpolate(_field_b, _field_f, True)
            return jnp.pad(midpoint_values, pad_width=((1, 0), (0, 0)), mode="edge") / dxy

        def pad_right_fn(_field_b, _field_f):
            midpoint_values = do_interpolate(_field_b, _field_f, False)
            return jnp.pad(midpoint_values, pad_width=((0, 1), (0, 0)), mode="edge") / dxy

        field_b, field_f = _field[:-1, :], _field[1:, :]
        _field = lax.cond(
            pad_left,
            lambda _operands: pad_left_fn(*_operands), lambda _operands: pad_right_fn(*_operands),
            (field_b, field_f)
        )

        return _field

    def axis1(_field, pad_left):
        def pad_left_fn(_field_b, _field_f):
            midpoint_values = do_interpolate(_field_b, _field_f, True)
            return jnp.pad(midpoint_values, pad_width=((0, 0), (1, 0)), mode="edge") / dxy

        def pad_right_fn(_field_b, _field_f):
            midpoint_values = do_interpolate(_field_b, _field_f, False)
            return jnp.pad(midpoint_values, pad_width=((0, 0), (0, 1)), mode="edge") / dxy

        field_b, field_f = _field[:, :-1], _field[:, 1:]
        _field = lax.cond(
            pad_left,
            lambda _operands: pad_left_fn(*_operands), lambda _operands: pad_right_fn(*_operands),
            (field_b, field_f)
        )

        return _field

    field = lax.cond(
        axis == 0,
        lambda operands: axis0(*operands), lambda operands: axis1(*operands),
        (field, padding == "left")
    )

    return field
