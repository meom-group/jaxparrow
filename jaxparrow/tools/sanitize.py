import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
from scipy import interpolate


def sanitize_data(
        arr: Float[Array, "lat lon"],
        fill_value: float = None,
        mask: Float[Array, "lat lon"] = None
) -> Float[Array, "lat lon"]:
    """
    Sanitizes data by replacing `nan` with ``fill_value`` and applying ``fill_value`` to the masked area.

    Parameters
    ----------
    arr : Float[Array, "lat lon"]
        Array to sanitize
    fill_value : float, optional
        Value to replace `nan` values and masked area with, defaults to `None` (falls to the mean of ``arr``)
    mask :  Float[Array, "lat lon"], optional
        Mask to apply, `1` or `True` for masked, defaults to `None`

    Returns
    -------
    arr : Float[Array, "lat lon"]
        Sanitized array
    """
    if fill_value is None:
        fill_value = jnp.nanmean(arr)

    arr = jnp.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if mask is not None:
        arr = jnp.where(mask, fill_value, arr)
    return arr


def init_mask(
        field: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"] = None
) -> Float[Array, "lat lon"]:
    """
    If ``mask is None``, initializes it from the `nan` values of ``field``.
    If ``mask is not None``, simply returns it.

    Parameters
    ----------
    field : Float[Array, "lat lon"]
        Field used to initialize the mask (if needed)
    mask :  Float[Array, "lat lon"], optional
        Mask to initialized (if `None`).

        Defaults to `None`

    Returns
    -------
    mask : Float[Array, "lat lon"]
        Initialized (if needed) mask
    """
    if mask is None:
        mask = jnp.isfinite(field)
    return mask


def handle_land_boundary(
        field1: Float[Array, "lat lon"],
        field2: Float[Array, "lat lon"]
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Replaces the non-finite values of ``field1`` (``field2``) with values of ``field2`` (``field1``), element-wise.

    It allows to introduce less non-finite values when applying grid operators.
    In such cases, ``field1`` and ``field2`` are left and right shifted versions of a field.

    Parameters
    ----------
    field1 : Float[Array, "lat lon"]
        A field
    field2 : Float[Array, "lat lon"]
        Another field

    Returns
    -------
    field1 : Float[Array, "lat lon"]
        A field whose non-finite values have been replaced with the ones from ``field2``
    field2 : Float[Array, "lat lon"]
        A field whose non-finite values have been replaced with the ones from ``field1``
    """
    field1 = jnp.where(jnp.isfinite(field1), field1, field2)
    field2 = jnp.where(jnp.isfinite(field2), field2, field1)
    return field1, field2


def sanitize_grid_np(
        lat: Float[Array, "lat lon"],
        lon: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"] = None
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Sanitizes (unstructured) grids by interpolated and extrapolated `nan` or masked values to avoid spurious
    (`0`, `nan`, `inf`) spatial steps and Coriolis factors.

    Helper function written using ``numpy`` and ``scipy``, and as such not used internally,
    because incompatible with ``jax.vmap``.
    Should be used before calling ``jaxparrow.geostrophy`` or ``jaxparrow.cyclogeostrophy``
    in case of suspicious latitudes or longitudes T grids.

    Caution: because it uses ``scipy.interpolate.RBFInterpolator``,
    it's memory usage grows quadratically with the number of grid points.

    Parameters
    ----------
    lat : Float[Array, "lat lon"]
        Grid latitudes
    lon : Float[Array, "lat lon"]
        Grid longitudes
    mask :  Float[Array, "lat lon"], optional
        Mask to apply, `1` or `True` for masked, defaults to `None`

    Returns
    -------
    lat : Float[Array, "lat lon"]
        Grid latitudes
    lon : Float[Array, "lat lon"]
        Grid longitudes
    """
    def fill_nan(arr: Float[Array, "lat lon"]) -> Float[Array, "lat lon"]:
        x = np.arange(0, arr.shape[1])
        y = np.arange(0, arr.shape[0])
        # mask invalid values
        arr = np.ma.masked_invalid(arr)
        xx, yy = np.meshgrid(x, y)
        # get only the valid values
        valid_x = xx[~arr.mask]
        valid_y = yy[~arr.mask]
        valid_arr = arr[~arr.mask]
        rbf = interpolate.RBFInterpolator(np.array([valid_x, valid_y]).T, valid_arr)
        # get the invalid ones
        invalid_x = xx[arr.mask]
        invalid_y = yy[arr.mask]
        invalid_arr = rbf(np.array([invalid_x, invalid_y]).T)
        # fill
        arr[arr.mask] = invalid_arr
        return arr.data

    # make sure nan are used behind masked pixels (and not 0)
    lat = sanitize_data(lat, fill_value=jnp.nan, mask=mask)
    lon = sanitize_data(lon, fill_value=jnp.nan, mask=mask)
    # fill nan using RBF interpolation
    lat = fill_nan(lat)
    lon = fill_nan(lon)
    return lat, lon
