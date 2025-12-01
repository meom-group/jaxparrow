from typing import Literal

from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, Float


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
        For example, following NEMO convention,
        interpolating from U to T points requires a `left` padding
        (the midpoint between $U_i$ and $U_{i+1}$ corresponds to $T_{i+1}$),
        and interpolating from T to U points a `right` padding
        (the midpoint between $T_i$ and $T_{i+1}$ corresponds to $U_i$)

    Returns
    -------
    field : Float[Array, "lat lon"]
        Interpolated field
    """
    f = jnp.moveaxis(field, axis, -1)

    mid = (f[:, :-1] + f[:, 1:]) * 0.5
    
    # handle mask: extrapolate at land boundaries (up to 1 cell)
    mid = jnp.where(
        jnp.isnan(mid),
        f[:, :-1],
        mid
    )
    mid = jnp.where(
        jnp.isnan(mid),
        f[:, 1:],
        mid
    )

    # extrapolate at the domain boundary
    mid = lax.cond(
        padding == "left",
        lambda: jnp.concatenate([f[:, :1], mid], axis=-1),
        lambda: jnp.concatenate([mid, f[:, -1:]], axis=-1)
    )

    mid = jnp.moveaxis(mid, -1, axis)

    mid = jnp.where(mask, jnp.nan, mid)

    return mid


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
        For example, following NEMO convention,
        interpolating from U to T points requires a `left` padding
        (the midpoint between $U_i$ and $U_{i+1}$ corresponds to $T_{i+1}$),
        and interpolating from T to U points a `right` padding
        (the midpoint between $T_i$ and $T_{i+1}$ corresponds to $U_i$)

    Returns
    -------
    field : Float[Array, "lat lon"]
        Interpolated field
    """
    f = jnp.moveaxis(field, axis, -1)

    mid = jnp.diff(f, axis=-1)
    
    # handle mask: extrapolate at land boundaries (up to 1 cell)
    mid = jnp.where(
        jnp.isnan(mid),
        jnp.pad(mid[:, 1:], pad_width=((0, 0), (0, 1)), mode="edge"),
        mid
    )
    mid = jnp.where(
        jnp.isnan(mid),
        jnp.pad(mid[:, :-1], pad_width=((0, 0), (1, 0)), mode="edge"),
        mid
    )

    # extrapolate at the domain boundary
    mid = lax.cond(
        padding == "left",
        lambda: jnp.pad(mid, pad_width=((0, 0), (1, 0)), mode="edge"),
        lambda: jnp.pad(mid, pad_width=((0, 0), (0, 1)), mode="edge" )
    )

    mid = jnp.moveaxis(mid, -1, axis)

    mid = jnp.where(mask, jnp.nan, mid)

    return mid / dxy
