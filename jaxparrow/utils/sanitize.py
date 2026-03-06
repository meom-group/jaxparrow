import jax
import jax.numpy as jnp
from jaxtyping import Float


def sanitize_data(
    arr: Float[jax.Array, "lat lon"],
    fill_value: float,
    mask: Float[jax.Array, "lat lon"]
) -> Float[jax.Array, "lat lon"]:
    """
    Sanitizes data by replacing `nan` with ``fill_value`` and applying ``fill_value`` to the masked area.

    Parameters
    ----------
    arr : Float[jax.Array, "lat lon"]
        Array to sanitize
    fill_value : float
        Value to replace `nan` values and masked area with
    mask :  Float[jax.Array, "lat lon"]
        Mask to apply, `1` or `True` for masked

    Returns
    -------
    arr : Float[jax.Array, "lat lon"]
        Sanitized array
    """
    arr = jnp.nan_to_num(arr, copy=False, nan=fill_value, posinf=fill_value, neginf=fill_value)
    arr = jnp.where(mask, fill_value, arr)
    return arr


def init_land_mask(
    field: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"] = None
) -> Float[jax.Array, "lat lon"]:
    """
    If ``mask is None``, initializes it from the `nan` values of ``field``.
    If ``mask is not None``, simply returns it.

    Parameters
    ----------
    field : Float[jax.Array, "lat lon"]
        Field used to initialize the mask (if needed)
    mask :  Float[jax.Array, "lat lon"], optional
        Mask to initialized (if `None`).

        Defaults to `None`

    Returns
    -------
    mask : Float[jax.Array, "lat lon"]
        Initialized (if needed) mask
    """
    if mask is None:
        mask = ~jnp.isfinite(field)
    return mask
