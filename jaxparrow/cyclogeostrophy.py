from collections.abc import Callable
from functools import partial
import numbers
from typing import Literal, Union

from jax import value_and_grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, Scalar
import optax

from .tools import geometry, kinematics, sanitize
from .geostrophy import geostrophy

#: Default maximum number of iterations for Penven and Ioannou approaches
N_IT_IT = 20
#: Default residual tolerance for Penven and Ioannou approaches
RES_EPS_IT = 0.01
#: Default residual value used during the first iteration for Penven and Ioannou approaches
RES_INIT_IT = "same"
#: Default size of the grid points used to compute the residual for Ioannou's approach
RES_FILTER_SIZE_IT = 3

#: Default maximum number of iterations for our variational approach
N_IT_VAR = 2000
#: Default learning rate for the gradient descent for our variational approach
LR_VAR = 0.005


# =============================================================================
# Cyclogeostrophy
# =============================================================================

def cyclogeostrophy(
        ssh_t: Float[Array, "lat lon"],
        lat_t: Float[Array, "lat lon"],
        lon_t: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"] = None,
        method: Literal["variational", "iterative"] = "variational",
        n_it: int = None,
        optim: Union[optax.GradientTransformation, str] = "sgd",
        optim_kwargs: dict = None,
        res_eps: float = RES_EPS_IT,
        res_init: Union[float, Literal["same"]] = RES_INIT_IT,
        use_res_filter: bool = False,
        res_filter_size: int = RES_FILTER_SIZE_IT,
        return_geos: bool = False,
        return_grids: bool = True,
        return_losses: bool = False
) -> [Float[Array, "lat lon"], ...]:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field from a Sea Surface Height (SSH) field
    using a variational (default) or iterative method.

    The cyclogeostrophic SSC velocity field is computed on a C-grid, following NEMO convention [1]_.

    Parameters
    ----------
    ssh_t : Float[Array, "lat lon"]
        SSH field (on the T grid)
    lat_t : Float[Array, "lat lon"]
        Latitude of the T grid
    lon_t : Float[Array, "lat lon"]
        Longitude of the T grid
    mask : Float[Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` `nan` values
    method : Literal["variational", "iterative"], optional
        Estimation method to use.
        If ``method="variational"``, then use our variational formulation.
        If ``method="iterative"``, then use an iterative approach [2]_, [3]_.

        Defaults to `variational`
    n_it : int, optional
        Maximum number of iterations.

        Defaults to ``N_IT_VAR`` or ``N_IT_IT``, based on ``method``
    optim : Union[optax.GradientTransformation, str], optional
        Optimizer to use.
        Can be an ``optax.GradientTransformation`` optimizer, or a ``string`` referring to such an optimizer [4]_.

        Defaults to `sgd`
    optim_kwargs : dict, optional
        Optimizer arguments (such as learning rate, etc...).

        If `None`, only the learning rate is enforced to ``LR_VAR``
    res_eps : float, optional
        Residual tolerance of the iterative approach.
        When residuals are smaller, the iterative approach considers local convergence to cyclogeostrophy.

        Defaults to ``RES_EPS_IT``
    res_init : Union[float | Literal["same"]], optional
        Residual initial value of the iterative approach.
        When residuals are larger at the first iteration,
        the iterative approach considers local divergence to cyclogeostrophy.

        If equals to `same` (default) absolute values of the geostrophic velocities are used
    use_res_filter : bool, optional
        Use of a convolution filter for the iterative approach when computing the residuals [3]_ or not [2]_.

        Defaults to `False`
    res_filter_size : int, optional
        Size of the convolution filter of the iterative approach, when ``use_res_filter=True``.

        Defaults to ``RES_FILTER_SIZE_IT``
    return_geos : bool, optional
        If `True`, returns the geostrophic SSC velocity field in addition to the cyclogeostrophic one.

        Defaults to `False`
    return_grids : bool, optional
        If `True`, returns the U and V grids.

        Defaults to `True`
    return_losses : bool, optional
        If `True`, returns the losses (cyclogeostrophic imbalance) over iterations.

        Defaults to `False`

    Returns
    -------
    u_cyclo_u : Float[Array, "lat lon"]
        U component of the cyclogeostrophic SSC velocity field (on the U grid)
    v_cyclo_v : Float[Array, "lat lon"]
        V component of the cyclogeostrophic SSC velocity field (on the V grid)
    lat_u : Float[Array, "lat lon"]
        Latitudes of the U grid, if ``return_grids=True``
    lon_u : Float[Array, "lat lon"]
        Longitudes of the U grid, if ``return_grids=True``
    lat_v : Float[Array, "lat lon"]
        Latitudes of the V grid, if ``return_grids=True``
    lon_v : Float[Array, "lat lon"]
        Longitudes of the V grid, if ``return_grids=True``
    u_geos_u : Float[Array, "lat lon"]
        U component of the geostrophic SSC velocity field (on the U grid), if ``return_geos=True``
    v_geos_v : Float[Array, "lat lon"]
        V component of the geostrophic SSC velocity field (on the V grid), if ``return_geos=True``
    losses: Float[Array, "n_it"]
        Cyclogeostrophic imbalance evaluated at each iteration, if ``return_losses=True``
    """
    # Make sure the mask is initialized
    mask = sanitize.init_mask(ssh_t, mask)

    # Compute geostrophic SSC velocity field
    u_geos_u, v_geos_v, lat_u, lon_u, lat_v, lon_v = geostrophy(ssh_t, lat_t, lon_t, mask,
                                                                return_grids=True)

    # Compute spatial steps and Coriolis factors
    dx_u, dy_u = geometry.compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = geometry.compute_spatial_step(lat_v, lon_v)
    coriolis_factor_u = geometry.compute_coriolis_factor(lat_u)
    coriolis_factor_v = geometry.compute_coriolis_factor(lat_v)

    # Handle spurious and masked data
    dx_u = sanitize.sanitize_data(dx_u, jnp.nan, mask)
    dy_u = sanitize.sanitize_data(dy_u, jnp.nan, mask)
    dx_v = sanitize.sanitize_data(dx_v, jnp.nan, mask)
    dy_v = sanitize.sanitize_data(dy_v, jnp.nan, mask)
    coriolis_factor_u = sanitize.sanitize_data(coriolis_factor_u, jnp.nan, mask)
    coriolis_factor_v = sanitize.sanitize_data(coriolis_factor_v, jnp.nan, mask)

    if method == "variational":
        res = _variational(u_geos_u, v_geos_v, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                           n_it, optim, optim_kwargs, return_losses)
    elif method == "iterative":
        res = _iterative(u_geos_u, v_geos_v, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, mask,
                         n_it, res_eps, res_init, use_res_filter, res_filter_size, return_losses)
    else:
        raise ValueError("method should be one of [\"variational\", \"iterative\"]")

    # Handle masked data
    u_cyclo_u, v_cyclo_v, losses = res
    u_cyclo_u = sanitize.sanitize_data(u_cyclo_u, jnp.nan, mask)
    v_cyclo_v = sanitize.sanitize_data(v_cyclo_v, jnp.nan, mask)

    res = (u_cyclo_u, v_cyclo_v)
    if return_geos:
        res = res + (u_geos_u, v_geos_v)
    if return_grids:
        res = res + (lat_u, lon_u, lat_v, lon_v)
    if return_losses:
        res = res + (losses,)

    return res


# =============================================================================
# Iterative method
# =============================================================================

def _iterative(
        u_geos_u: Float[Array, "lat lon"],
        v_geos_v: Float[Array, "lat lon"],
        dx_u: Float[Array, "lat lon"],
        dx_v: Float[Array, "lat lon"],
        dy_u: Float[Array, "lat lon"],
        dy_v: Float[Array, "lat lon"],
        coriolis_factor_u: Float[Array, "lat lon"],
        coriolis_factor_v: Float[Array, "lat lon"],
        mask: Float[Array, "lat lon"],
        n_it: int,
        res_eps: float,
        res_init: Union[float, str],
        use_res_filter: bool,
        res_filter_size: int,
        return_losses: bool
) -> [Float[Array, "lat lon"], ...]:
    if mask is None:
        mask = jnp.zeros_like(u_geos_u)
    if n_it is None:
        n_it = N_IT_IT
    if res_init == "same":
        res_n = jnp.maximum(jnp.abs(u_geos_u), jnp.abs(v_geos_v))
    elif isinstance(res_init, numbers.Number):
        res_n = res_init * jnp.ones_like(u_geos_u)
    else:
        raise ValueError("res_init should be equal to \"same\" or be a number.")

    u_cyclo, v_cyclo = jnp.copy(u_geos_u), jnp.copy(v_geos_v)
    res_filter = jnp.ones((res_filter_size, res_filter_size))
    res_weights = jsp.signal.correlate(jnp.ones_like(u_geos_u), res_filter, mode="same")
    losses = jnp.ones(n_it) * jnp.nan
    for i in jnp.arange(n_it):
        # next it
        u_adv_v, v_adv_u = kinematics.advection(u_cyclo, v_cyclo, dx_u, dy_u, dx_v, dy_v)
        u_np1 = u_geos_u - v_adv_u / coriolis_factor_u
        v_np1 = v_geos_v + u_adv_v / coriolis_factor_v

        # compute dist to u_cyclo and v_cyclo
        res_np1 = jnp.abs(u_np1 - u_cyclo) + jnp.abs(v_np1 - v_cyclo)
        if use_res_filter:
            res_np1 = jsp.signal.correlate(res_np1, res_filter, mode="same") / res_weights  # apply filter
        # compute intermediate masks
        mask_jnp1 = jnp.where(res_np1 >= res_eps, 0, 1)  # nan comp. equiv. to jnp.where(res_np1 < res_eps, 1, 0)
        mask_n = jnp.where(res_np1 <= res_n, 0, 1)  # nan comp. equiv. to jnp.where(res_np1 > res_n, 1, 0)

        # compute loss
        if return_losses:
            loss = _cyclogeostrophic_diff(u_geos_u, v_geos_v, u_cyclo, v_cyclo, u_adv_v, v_adv_u,
                                          coriolis_factor_u, coriolis_factor_v)
            losses = losses.at[i].set(loss)

        # update cyclogeostrophic velocities
        u_cyclo = mask * u_cyclo + (1 - mask) * (mask_n * u_cyclo + (1 - mask_n) * u_np1)
        v_cyclo = mask * v_cyclo + (1 - mask) * (mask_n * v_cyclo + (1 - mask_n) * v_np1)

        # update mask and residuals
        mask = jnp.maximum(mask, jnp.maximum(mask_jnp1, mask_n))
        res_n = res_np1

        if jnp.all(mask == 1):
            break

    return u_cyclo, v_cyclo, losses


# =============================================================================
# Variational method
# =============================================================================

@partial(jit, static_argnums=(0, 3))
def _step(
        f: Callable[[[Float[Array, "lat lon"], Float[Array, "lat lon"]]], Float[Scalar, ""]],
        u_cyclo_u: Float[Array, "lat lon"],
        v_cyclo_v: Float[Array, "lat lon"],
        optim: optax.GradientTransformation,
        opt_state: optax.OptState
) -> [Float[Array, "lat lon"], ...]:
    params = (u_cyclo_u, v_cyclo_v)
    # evaluate the cost function and compute its gradient
    loss, grads = value_and_grad(f)(params)
    # make sure to remove nan values
    grads = (sanitize.sanitize_data(grads[0], 0), sanitize.sanitize_data(grads[1], 0))
    # update the optimizer
    updates, opt_state = optim.update(grads, opt_state, params)
    # apply updates to the parameters
    u_n, v_n = optax.apply_updates(params, updates)
    return u_n, v_n, opt_state, loss


def _gradient_descent(
        u_geos_u: Float[Array, "lat lon"],
        v_geos_v: Float[Array, "lat lon"],
        f: Callable[[[Float[Array, "lat lon"], Float[Array, "lat lon"]]], Float[Scalar, ""]],
        n_it: int,
        optim: optax.GradientTransformation,
        return_losses: bool
) -> [Float[Array, "lat lon"], ...]:
    # initial guess is the geostrophy
    u_cyclo_u, v_cyclo_v = jnp.copy(u_geos_u), jnp.copy(v_geos_v)
    opt_state = optim.init((u_cyclo_u, v_cyclo_v))
    losses = jnp.ones(n_it) * jnp.nan
    for i in jnp.arange(n_it):
        # update the solution
        u_cyclo_u, v_cyclo_v, opt_state, loss = _step(f, u_cyclo_u, v_cyclo_v, optim, opt_state)
        if return_losses:
            losses = losses.at[i].set(loss)

    return u_cyclo_u, v_cyclo_v, losses


def _variational(
        u_geos_u: Float[Array, "lat lon"],
        v_geos_v: Float[Array, "lat lon"],
        dx_u: Float[Array, "lat lon"],
        dx_v: Float[Array, "lat lon"],
        dy_u: Float[Array, "lat lon"],
        dy_v: Float[Array, "lat lon"],
        coriolis_factor_u: Float[Array, "lat lon"],
        coriolis_factor_v: Float[Array, "lat lon"],
        n_it: int,
        optim: Union[optax.GradientTransformation, str],
        optim_kwargs: dict,
        return_losses: bool
) -> [Float[Array, "lat lon"], ...]:
    if n_it is None:
        n_it = N_IT_VAR

    if isinstance(optim, str):
        if optim_kwargs is None:
            optim_kwargs = {"learning_rate": LR_VAR}
        optim = getattr(optax, optim)(**optim_kwargs)
    elif not isinstance(optim, optax.GradientTransformation):
        raise TypeError("optim should be an optax.GradientTransformation optimizer, or a string referring to such an "
                        "optimizer.")

    # define the cost function
    def f(uv: [Float[Array, "lat lon"], Float[Array, "lat lon"]]) -> Float[Array, "lat lon"]:
        u, v = uv
        u_adv_v, v_adv_u = kinematics.advection(u, v, dx_u, dy_u, dx_v, dy_v)
        return _cyclogeostrophic_diff(u_geos_u, v_geos_v, u, v, u_adv_v, v_adv_u,
                                      coriolis_factor_u, coriolis_factor_v)

    return _gradient_descent(u_geos_u, v_geos_v, f, n_it, optim, return_losses)


def _cyclogeostrophic_diff(
        u_geos_u: Float[Array, "lat lon"],
        v_geos_v: Float[Array, "lat lon"],
        u_cyclo_u: Float[Array, "lat lon"],
        v_cyclo_v: Float[Array, "lat lon"],
        u_adv_v: Float[Array, "lat lon"],
        v_adv_u: Float[Array, "lat lon"],
        coriolis_factor_u: Float[Array, "lat lon"],
        coriolis_factor_v: Float[Array, "lat lon"]
) -> Float[Array, "lat lon"]:
    J_u = jnp.nansum((u_cyclo_u + v_adv_u / coriolis_factor_u - u_geos_u) ** 2)
    J_v = jnp.nansum((v_cyclo_v - u_adv_v / coriolis_factor_v - v_geos_v) ** 2)
    return J_u + J_v
