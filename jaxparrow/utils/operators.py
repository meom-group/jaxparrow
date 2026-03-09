from typing import Literal

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Float


def interpolation(
    field: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    axis: Literal[0, 1],
    padding: Literal["left", "right"]
) -> Float[jax.Array, "lat lon"]:
    """
    Interpolates the values of a ``field`` along a given ``axis`` (`0` for `lat`/`y`, `1` for `lon`/`x`),
    applying ``padding`` to the `left` (i.e. `West` if ``axis=1``, `South` if ``axis=0``) or
    to the `right` (i.e. `East` if ``axis=1``, `North` if ``axis=0``) of the domain,
    using nearest non interpolated value at the padded edge.

    Parameters
    ----------
    field : Float[jax.Array, "lat lon"]
        Field to interpolate
    mask : Float[jax.Array, "lat lon"]
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
    field : Float[jax.Array, "lat lon"]
        Interpolated field
    """
    f = jnp.moveaxis(field, axis, -1)

    mid = (f[:, :-1] + f[:, 1:]) * 0.5

    # Pad at the domain boundary with edge value (open boundary condition)
    mid = lax.cond(
        padding == "left",
        lambda: jnp.concatenate([mid[:, 0:1], mid], axis=-1),
        lambda: jnp.concatenate([mid, mid[:, -1:]], axis=-1)
    )

    mid = jnp.moveaxis(mid, -1, axis)

    mid = jnp.where(mask, jnp.nan, mid)

    return mid


def derivative(
    field: Float[jax.Array, "lat lon"],
    dxy: Float[jax.Array, "lat lon"],
    land_mask: Float[jax.Array, "lat lon"],
    axis: Literal[0, 1],
    padding: Literal["left", "right"]
) -> Float[jax.Array, "lat lon"]:
    """
    Differentiates a ``field``, using finite differences, along a given ``axis`` (`0` for `lat`/`y`, `1` for `lon`/'x'),
    applying ``padding`` to the `left` (i.e. `West` if ``axis=1``, `South` if ``axis=0``) or
    to the `right` (i.e. `East` if ``axis=1``, `North` if ``axis=0``) of the domain.

    At domain edges, an open boundary condition is applied (zero second derivative),
    meaning the boundary derivative equals the nearest interior derivative. This is
    appropriate for domains with sharp physical boundaries (e.g., SWOT swaths) where
    the signal continues smoothly beyond the observation edge.

    Parameters
    ----------
    field : Float[jax.Array, "lat lon"]
        Field to differentiate
    dxy : Float[jax.Array, "lat lon"]
        Spatial steps
    land_mask : Float[jax.Array, "lat lon"]
        Mask indicating the land domain where extrapolation should be applied.
        `False`/`0` indicates ocean cells, `True`/`1` indicates land cells.
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
    field : Float[jax.Array, "lat lon"]
        Differentiated field
    """
    f = jnp.moveaxis(field, axis, -1)

    mid = jnp.diff(f, axis=-1)

    # Open boundary condition: ∂²f/∂x² = 0 at boundary
    # This means the boundary derivative equals the nearest interior derivative
    mid = lax.cond(
        padding == "left",
        lambda: jnp.concatenate([mid[:, 0:1], mid], axis=-1),
        lambda: jnp.concatenate([mid, mid[:, -1:]], axis=-1)
    )

    mid = jnp.moveaxis(mid, -1, axis)

    mid = jnp.where(land_mask, jnp.nan, mid)

    return mid / dxy


def rotate_to_geographic(
    grad_i: Float[jax.Array, "lat lon"],
    grad_j: Float[jax.Array, "lat lon"],
    grid_angle: Float[jax.Array, "lat lon"]
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]]:
    """
    Rotates gradients from grid coordinates to geographic coordinates.

    For curvilinear grids (e.g., SWOT swaths, tripolar grids), the grid axes are not aligned
    with geographic east-west/north-south directions. This function rotates gradient components
    from grid (i, j) coordinates to geographic (x=east, y=north) coordinates.

    Parameters
    ----------
    grad_i : Float[jax.Array, "lat lon"]
        Gradient component along the grid i-axis (axis=1)
    grad_j : Float[jax.Array, "lat lon"]
        Gradient component along the grid j-axis (axis=0)
    grid_angle : Float[jax.Array, "lat lon"]
        Angle in radians from geographic east to the grid i-direction (counterclockwise positive).
        Can be computed using :func:`geometry.compute_grid_angle`.

    Returns
    -------
    grad_x : Float[jax.Array, "lat lon"]
        Gradient in the geographic x (eastward) direction
    grad_y : Float[jax.Array, "lat lon"]
        Gradient in the geographic y (northward) direction

    Notes
    -----
    The rotation from grid coordinates (i, j) to geographic coordinates (x, y) assumes
    the grid is locally orthogonal (j-axis at angle θ + π/2 from east):

    .. math::

        \\frac{\\partial f}{\\partial x} = \\cos(\\theta) \\frac{\\partial f}{\\partial s_i} 
        - \\sin(\\theta) \\frac{\\partial f}{\\partial s_j}

        \\frac{\\partial f}{\\partial y} = \\sin(\\theta) \\frac{\\partial f}{\\partial s_i} 
        + \\cos(\\theta) \\frac{\\partial f}{\\partial s_j}

    where θ is the angle from geographic east to the grid i-axis.
    """
    cos_theta = jnp.cos(grid_angle)
    sin_theta = jnp.sin(grid_angle)

    grad_x = cos_theta * grad_i - sin_theta * grad_j
    grad_y = sin_theta * grad_i + cos_theta * grad_j

    return grad_x, grad_y
