import jax
import jax.numpy as jnp
from jaxtyping import Float


def sanitize_data(
    arr: Float[jax.Array, "y x"],
    fill_value: float,
    land_mask: Float[jax.Array, "y x"]
) -> Float[jax.Array, "y x"]:
    """
    Sanitizes data by replacing `nan` with ``fill_value`` and applying ``fill_value`` to the masked area.

    Parameters
    ----------
    arr : Float[jax.Array, "y x"]
        Array to sanitize
    fill_value : float
        Value to replace `nan` values and masked area with
    land_mask :  Float[jax.Array, "y x"]
        Mask to apply, `1` or `True` for masked

    Returns
    -------
    arr : Float[jax.Array, "y x"]
        Sanitized array
    """
    arr = jnp.nan_to_num(arr, copy=False, nan=fill_value, posinf=fill_value, neginf=fill_value)
    arr = jnp.where(land_mask, fill_value, arr)
    return arr


def init_land_mask(
    field: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"] = None
) -> Float[jax.Array, "y x"]:
    """
    If ``land_mask is None``, initializes it from the `nan` values of ``field``.
    If ``land_mask is not None``, simply returns it.

    Parameters
    ----------
    field : Float[jax.Array, "y x"]
        Field used to initialize the mask (if needed)
    land_mask :  Float[jax.Array, "y x"], optional
        Mask to initialized (if `None`).

        Defaults to `None`

    Returns
    -------
    land_mask : Float[jax.Array, "y x"]
        Initialized (if needed) land mask
    """
    if land_mask is None:
        land_mask = ~jnp.isfinite(field)
    return land_mask
