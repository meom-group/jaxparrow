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
    field: Float[jax.Array, "lat lon"],
    dxy: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    axis: Literal[0, 1],
    padding: Literal["left", "right"]
) -> Float[jax.Array, "lat lon"]:
    """
    Differentiates a ``field``, using finite differences, along a given ``axis`` (`0` for `lat`/`y`, `1` for `lon`/'x'),
    applying ``padding`` to the `left` (i.e. `West` if ``axis=1``, `South` if ``axis=0``) or
    to the `right` (i.e. `East` if ``axis=1``, `North` if ``axis=0``) of the domain,
    using nearest derivative value at the padded edge.

    At mask boundaries, a zero-gradient boundary condition is applied: if one of the two cells
    in the finite difference stencil is masked/NaN, the derivative is set to 0 (flat extrapolation).

    Parameters
    ----------
    field : Float[jax.Array, "lat lon"]
        Field to differentiate
    dxy : Float[jax.Array, "lat lon"]
        Spatial steps
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
        Differentiated field
    """
    f = jnp.moveaxis(field, axis, -1)

    mid = jnp.diff(f, axis=-1)
    
    # Zero-gradient boundary condition: if either cell in the stencil is NaN,
    # set the derivative to 0 (flat extrapolation assumption)
    mid = jnp.nan_to_num(mid, nan=0.0, posinf=0.0, neginf=0.0)

    # extrapolate at the domain boundary
    mid = lax.cond(
        padding == "left",
        lambda: jnp.pad(mid, pad_width=((0, 0), (1, 0)), mode="edge"),
        lambda: jnp.pad(mid, pad_width=((0, 0), (0, 1)), mode="edge")
    )

    mid = jnp.moveaxis(mid, -1, axis)

    mid = jnp.where(mask, jnp.nan, mid)

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


def extrapolate_to_valid(
    field: Float[jax.Array, "lat lon"],
    valid_mask: Float[jax.Array, "lat lon"],
    max_iterations: int = 10
) -> Float[jax.Array, "lat lon"]:
    """
    Extrapolates NaN values in ``field`` within the ``valid_mask`` domain using iterative neighbor averaging.

    This function fills cells that are valid according to ``valid_mask`` but are NaN in ``field``
    (typically edge cells lost due to derivative computation) by averaging from valid neighboring cells.
    The extrapolation is performed iteratively until all fillable cells are filled or ``max_iterations``
    is reached.

    Parameters
    ----------
    field : Float[jax.Array, "lat lon"]
        Field with potential NaN values to extrapolate
    valid_mask : Float[jax.Array, "lat lon"]
        Mask indicating the valid domain where extrapolation should be applied.
        `False`/`0` indicates valid ocean cells, `True`/`1` indicates masked (land) cells.
        NaN cells in ``field`` that are valid according to this mask will be filled.
    max_iterations : int, optional
        Maximum number of extrapolation iterations.
        Defaults to 10

    Returns
    -------
    field : Float[jax.Array, "lat lon"]
        Field with NaN values extrapolated within the valid domain

    Notes
    -----
    The extrapolation uses a simple iterative scheme where each NaN cell within the valid domain
    is filled with the average of its valid (non-NaN) neighbors. This process repeats until
    no more cells can be filled or the maximum iterations are reached.
    """
    def extrapolate_step(f):
        # Shift field in all 4 directions
        north = jnp.roll(f, -1, axis=0)
        south = jnp.roll(f, 1, axis=0)
        east = jnp.roll(f, -1, axis=1)
        west = jnp.roll(f, 1, axis=1)
        
        # Zero out contributions from NaN neighbors and count valid neighbors
        north_valid = jnp.where(jnp.isnan(north), 0.0, north)
        south_valid = jnp.where(jnp.isnan(south), 0.0, south)
        east_valid = jnp.where(jnp.isnan(east), 0.0, east)
        west_valid = jnp.where(jnp.isnan(west), 0.0, west)
        
        north_count = (~jnp.isnan(north)).astype(jnp.float32)
        south_count = (~jnp.isnan(south)).astype(jnp.float32)
        east_count = (~jnp.isnan(east)).astype(jnp.float32)
        west_count = (~jnp.isnan(west)).astype(jnp.float32)
        
        neighbor_sum = north_valid + south_valid + east_valid + west_valid
        neighbor_count = north_count + south_count + east_count + west_count
        
        # Compute average where we have valid neighbors
        avg = jnp.where(neighbor_count > 0, neighbor_sum / neighbor_count, jnp.nan)
        
        # Fill NaN cells that are in valid domain with neighbor average
        needs_fill = jnp.isnan(f) & ~valid_mask
        f_new = jnp.where(needs_fill, avg, f)
        
        return f_new
    
    def cond_fn(state):
        f, i = state
        # Continue if there are still NaN cells in valid domain and we haven't exceeded max_iterations
        has_nan_in_valid = jnp.any(jnp.isnan(f) & ~valid_mask)
        return has_nan_in_valid & (i < max_iterations)
    
    def body_fn(state):
        f, i = state
        return extrapolate_step(f), i + 1
    
    result, _ = lax.while_loop(cond_fn, body_fn, (field, 0))
    
    return result
