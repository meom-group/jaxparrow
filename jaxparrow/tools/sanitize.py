from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
from scipy import interpolate


def sanitize_data(
        arr: Float[Array, "lat lon"],
        fill_value: float,
        mask: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    """
    Sanitizes data by replacing `nan` with ``fill_value`` and applying ``fill_value`` to the masked area.

    Parameters
    ----------
    arr : Float[Array, "lat lon"]
        Array to sanitize
    fill_value : float
        Value to replace `nan` values and masked area with
    mask :  Float[Array, "lat lon"]
        Mask to apply, `1` or `True` for masked

    Returns
    -------
    arr : Float[Array, "lat lon"]
        Sanitized array
    """
    arr = jnp.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    arr = jnp.where(mask, fill_value, arr)
    return arr


def init_land_mask(
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
        mask = ~jnp.isfinite(field)
    return mask


def handle_land_boundary(
        field1: Float[Array, "lat lon"],
        field2: Float[Array, "lat lon"],
        mask1: Float[Array, "lat lon"],
        mask2: Float[Array, "lat lon"],
        pad_left: bool
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Replaces the masked values of ``field1`` (``field2``) with values of ``field2`` (``field1``), element-wise.

    It allows computing more coherent values when applying grid operators.
    In such cases, ``field1`` and ``field2`` are left and right shifted versions of a field (along one of the axes).

    Parameters
    ----------
    field1 : Float[Array, "lat lon"]
        A field
    field2 : Float[Array, "lat lon"]
        Another field
    mask1 : Float[Array, "lat lon"]
        A mask defining the marine area of ``field1`` spatial domain; `1` or `True` stands for masked (i.e. land)
    mask2 : Float[Array, "lat lon"]
        A mask defining the marine area of ``field2`` spatial domain; `1` or `True` stands for masked (i.e. land)
    pad_left : bool
        If `True`, apply padding in the `left` direction (i.e. `West` or `South`) ;
        if `False`, apply padding in the `right` direction (i.e. `East` or `North`).

    Returns
    -------
    field1 : Float[Array, "lat lon"]
        A field whose masked values have been replaced with the ones from ``field2``
    field2 : Float[Array, "lat lon"]
        A field whose masked values have been replaced with the ones from ``field1``
    """
    field1, field2 = lax.cond(
        pad_left,
        lambda: (jnp.where(mask1, field2, field1), field2),
        lambda: (field1, jnp.where(mask2, field1, field2))
    )
    return field1, field2


def sanitize_grid_np(
        lat: Float[Array, "lat lon"],
        lon: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"] = None
) -> [Float[Array, "lat lon"], Float[Array, "lat lon"]]:
    """
    Sanitizes (unstructured) grids by interpolated and extrapolated `nan` or masked values to avoid spurious
    (`0`, `nan`, `inf`) spatial steps and Coriolis factors.

    Helper function written using pure ``numpy`` and ``scipy``, and as such not used internally,
    because incompatible with ``jax.vmap`` and likes.
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
    lat = sanitize_data(lat, jnp.nan, mask)
    lon = sanitize_data(lon, jnp.nan, mask)
    # fill nan using RBF interpolation
    lat = fill_nan(lat)
    lon = fill_nan(lon)
    return lat, lon
