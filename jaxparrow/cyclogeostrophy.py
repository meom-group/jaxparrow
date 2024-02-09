from collections.abc import Callable
from functools import partial
import numbers
from typing import Literal, Tuple, Union

from jax import value_and_grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float, Scalar
import optax

from .tools import tools

#: Default maximum number of iterations for Penven and Ioannou approaches
N_IT_IT = 20
#: Default residual tolerance of Penven and Ioannou approaches
RES_EPS_IT = 0.01
#: Default residual value used during the first iteration of Penven and Ioannou approaches
RES_INIT_IT = "same"
#: Default size of the grid points used to compute the residual in Ioannou's approach
RES_FILTER_SIZE_IT = 3

#: Default maximum number of iterations for our variational approach
N_IT_VAR = 2000
#: Default learning rate for the gradient descent of our variational approach
LR_VAR = 0.005

__all__ = ["cyclogeostrophy", "LR_VAR", "N_IT_IT", "N_IT_VAR", "RES_EPS_IT", "RES_INIT_IT", "RES_FILTER_SIZE_IT"]


# =============================================================================
# Cyclogeostrophy
# =============================================================================

def cyclogeostrophy(
        u_geos: Float[Array, "lat lon"],
        v_geos: Float[Array, "lat lon"],
        lat_u: Float[Array, "lat lon"],
        lon_u: Float[Array, "lat lon"],
        lat_v: Float[Array, "lat lon"],
        lon_v: Float[Array, "lat lon"],
        mask_u: Float[Array, "lat lon"] = None,
        mask_v: Float[Array, "lat lon"] = None,
        method: Literal["variational", "iterative"] = "variational",
        n_it: int = None,
        optim: Union[optax.GradientTransformation, str] = "sgd",
        optim_kwargs: dict = None,
        res_eps: float = RES_EPS_IT,
        res_init: Union[float, Literal["same"]] = RES_INIT_IT,
        use_res_filter: bool = False,
        res_filter_size: int = RES_FILTER_SIZE_IT,
        return_losses: bool = False
) -> Union[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]],
           Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "n_it"]]]:
    """
    Computes velocities from cyclogeostrophic approximation using a variational (default) or iterative method.

    :param u_geos: U geostrophic velocity
    :type u_geos: Float[Array, "lat lon"]
    :param v_geos: V geostrophic velocity
    :type v_geos: Float[Array, "lat lon"]
    :param lat_u: U latitude
    :type lat_u: Float[Array, "lat lon"]
    :param lon_u: U longitude
    :type lon_u: Float[Array, "lat lon"]
    :param lat_v: V latitude
    :type lat_v: Float[Array, "lat lon"]
    :param lon_v: V longitude
    :type lon_v: Float[Array, "lat lon"]
    :param mask_u: mask to apply at U points, 1 or True stands for masked, defaults to None
    :type mask_u: Float[Array, "lat lon"], optional
    :param mask_v: mask to apply at V points, 1 or True stands for masked, defaults to None
    :type mask_v: Float[Array, "lat lon"], optional
    :param method: estimation method to use, defaults to "variational"
    :type method: Literal["variational", "iterative"], optional
    :param n_it: maximum number of iterations, defaults to N_IT_VAR or N_IT_IT based on the method argument
    :type n_it: int, optional
    :param optim: optimizer to use. Can be an optax.GradientTransformation optimizer, or a string referring to such an
                  optimizer. Defaults to "sgd".
    :type optim: Union[optax.GradientTransformation, str], optional
    :param optim_kwargs: optimizer arguments (such as learning rate, etc...)
    :type optim_kwargs: dict
    :param res_eps: residual tolerance of the iterative approach.
                    When residuals are smaller, we consider them as equal to 0.
                    Defaults to EPS_IT
    :type res_eps: float, optional
    :param res_init: residual initial value of the iterative approach.
                     When residuals are larger at the first iteration, we consider that the solution diverges.
                     If equals to "same" (default) absolute values of the geostrophic velocities are used.
                     Defaults to RES_INIT_IT
    :type res_init: Union[float | Literal["same"]], optional
    :param use_res_filter: use of a convolution filter for the iterative approach when computing the residuals
                           (method from Ioannou et al.) or not (original method from Penven et al.), defaults to False
    :type use_res_filter: bool, optional
    :param res_filter_size: size of the convolution filter used for the iterative approach when computing the residuals,
                            defaults to RES_FILTER_SIZE_IT
    :type res_filter_size: int, optional
    :param return_losses: should return losses over iteration? defaults to False
    :type return_losses: bool, optional

    :returns: U and V cyclogeostrophic velocities, NxM grids; and eventually losses over iterations
    :rtype: Union[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]],
                  Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "n_it"]]]
    """
    # Computing spatial steps and Coriolis factors
    dx_u, dy_u = tools.compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = tools.compute_spatial_step(lat_v, lon_v)
    coriolis_factor_u = tools.compute_coriolis_factor(lat_u)
    coriolis_factor_v = tools.compute_coriolis_factor(lat_v)

    # Handling spurious and masked data
    u_geos = tools.sanitise_data(u_geos, jnp.nan, mask_u)
    v_geos = tools.sanitise_data(v_geos, jnp.nan, mask_v)
    dx_u = tools.sanitise_data(dx_u, jnp.nan, mask_u)
    dy_u = tools.sanitise_data(dy_u, jnp.nan, mask_u)
    dx_v = tools.sanitise_data(dx_v, jnp.nan, mask_v)
    dy_v = tools.sanitise_data(dy_v, jnp.nan, mask_v)
    coriolis_factor_u = tools.sanitise_data(coriolis_factor_u, jnp.nan, mask_u)
    coriolis_factor_v = tools.sanitise_data(coriolis_factor_v, jnp.nan, mask_v)

    if method == "variational":
        res = _variational(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                           n_it, optim, optim_kwargs, return_losses)
    elif method == "iterative":
        res = _iterative(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, mask_u, mask_v,
                         n_it, res_eps, res_init, use_res_filter, res_filter_size, return_losses)
    else:
        raise ValueError("method should be one of [\"variational\", \"iterative\"]")

    # Handling masked data
    u_cyclo, v_cyclo = res[:2]
    u_cyclo = tools.sanitise_data(u_cyclo, jnp.nan, mask_u)
    v_cyclo = tools.sanitise_data(v_cyclo, jnp.nan, mask_v)
    res = (u_cyclo, v_cyclo) + res[2:]

    return res


# =============================================================================
# Iterative method
# =============================================================================

def _iterative(
        u_geos: Float[Array, "lat lon"],
        v_geos: Float[Array, "lat lon"],
        dx_u: Float[Array, "lat lon"],
        dx_v: Float[Array, "lat lon"],
        dy_u: Float[Array, "lat lon"],
        dy_v: Float[Array, "lat lon"],
        coriolis_factor_u: Float[Array, "lat lon"],
        coriolis_factor_v: Float[Array, "lat lon"],
        mask_u: Float[Array, "lat lon"],
        mask_v: Float[Array, "lat lon"],
        n_it: int,
        res_eps: float,
        res_init: Union[float, str],
        use_res_filter: bool,
        res_filter_size: int,
        return_losses: bool
) -> Union[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]],
           Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "n_it"]]]:
    """
    Computes velocities from cyclogeostrophic approximation using the iterative method from Penven et al. (2014)

    :param u_geos: U geostrophic velocity value
    :type u_geos: Float[Array, "lat lon"]
    :param v_geos: V geostrophic velocity value
    :type v_geos: Float[Array, "lat lon"]
    :param dx_u: U spatial step along x
    :type dx_u: Float[Array, "lat lon"]
    :param dx_v: V spatial step along x
    :type dx_v: Float[Array, "lat lon"]
    :param dy_u: U spatial step along y
    :type dy_u: Float[Array, "lat lon"]
    :param dy_v: V spatial step along y
    :type dy_v: Float[Array, "lat lon"]
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: Float[Array, "lat lon"]
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: Float[Array, "lat lon"]
    :param mask: initial mask
    :type mask: Float[Array, "lat lon"]
    :param n_it: maximum number of iterations
    :type n_it: int
    :param res_eps: residual tolerance: if residuals are smaller, we consider them as equal to 0
    :type res_eps: float
    :param res_init: residual initial value: if residuals are larger at the first iteration, we consider that the
                     solution diverges. If equals to "same" (default) absolute values of the geostrophic velocities are
                     used
    :type res_init: float | str
    :param use_res_filter: use of a convolution filter when computing the residuals (method from Ioannou et al.) or not
                           (original method from Penven et al.)
    :type use_res_filter: bool
    :param res_filter_size: size of the convolution filter used when computing the residuals
    :type res_filter_size: int
    :param return_losses: should return losses over iteration?
    :type return_losses: bool

    :returns: U and V cyclogeostrophic velocities, and eventually losses over iterations
    :rtype: Union[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]],
                  Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "n_it"]]]
    """
    if mask_u is not None and mask_v is not None:
        mask = mask_u + mask_v
    else:
        mask = jnp.zeros_like(u_geos)
    if n_it is None:
        n_it = N_IT_IT
    if res_init == "same":
        res_n = jnp.maximum(jnp.abs(u_geos), jnp.abs(v_geos))
    elif isinstance(res_init, numbers.Number):
        res_n = res_init * jnp.ones_like(u_geos)
    else:
        raise ValueError("res_init should be equal to \"same\" or be a number.")

    u_cyclo, v_cyclo = jnp.copy(u_geos), jnp.copy(v_geos)
    res_filter = jnp.ones((res_filter_size, res_filter_size))
    res_weights = jsp.signal.correlate(jnp.ones_like(u_geos), res_filter, mode="same")
    losses = jnp.zeros(n_it)
    for i in jnp.arange(n_it):
        # next it
        adv_v = tools.compute_advection_v(u_cyclo, v_cyclo, dx_v, dy_v)
        adv_u = tools.compute_advection_u(u_cyclo, v_cyclo, dx_u, dy_u)
        u_np1 = u_geos - adv_v / coriolis_factor_u
        v_np1 = v_geos + adv_u / coriolis_factor_v

        # compute dist to u_cyclo and v_cyclo
        res_np1 = jnp.abs(u_np1 - u_cyclo) + jnp.abs(v_np1 - v_cyclo)
        if use_res_filter:
            res_np1 = jsp.signal.correlate(res_np1, res_filter, mode="same") / res_weights  # apply filter
        # compute intermediate masks
        mask_jnp1 = jnp.where(res_np1 >= res_eps, 0, 1)  # nan comp. equiv. to jnp.where(res_np1 < res_eps, 1, 0)
        mask_n = jnp.where(res_np1 <= res_n, 0, 1)  # nan comp. equiv. to jnp.where(res_np1 > res_n, 1, 0)

        # compute loss
        if return_losses:
            loss = tools.compute_cyclogeostrophic_diff(u_geos, v_geos, u_cyclo, v_cyclo,
                                                       adv_u, adv_v,
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

    if return_losses:
        res = u_cyclo, v_cyclo, losses[:i]  # noqa
    else:
        res = u_cyclo, v_cyclo

    return res


# =============================================================================
# Variational method
# =============================================================================

@partial(jit, static_argnums=(0, 3))
def _step(
        f: Callable[[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]], Float[Scalar, ""]],
        u_cyclo: Float[Array, "lat lon"],
        v_cyclo: Float[Array, "lat lon"],
        optim: optax.GradientTransformation,
        opt_state: optax.OptState
) -> Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], optax.OptState, Float[Scalar, ""]]:
    """Executes one iteration of the variational approach, using gradient descent

    :param f: loss function
    :type f: Callable[[Float[Array, "lat lon"], Float[Array, "lat lon"]], Float[Scalar, ""]]
    :param u_cyclo: U cyclogeostrophic velocity value
    :type u_cyclo: Float[Array, "lat lon"]
    :param v_cyclo: V cyclogeostrophic velocity value
    :type v_cyclo: Float[Array, "lat lon"]
    :param optim: optimizer to use
    :type optim: optax.GradientTransformation

    :returns: updated U and V cyclogeostrophic velocities, optimizer state, and loss value
    :rtype: Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], optax.OptState, Float[Scalar, ""]]
    """
    params = (u_cyclo, v_cyclo)
    loss, grads = value_and_grad(f)(params)
    grads = (tools.sanitise_data(grads[0], 0), tools.sanitise_data(grads[1], 0))
    updates, opt_state = optim.update(grads, opt_state, params)
    u_n, v_n = optax.apply_updates(params, updates)
    return u_n, v_n, opt_state, loss


def _gradient_descent(
        u_geos: Float[Array, "lat lon"],
        v_geos: Float[Array, "lat lon"],
        f: Callable[[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]], Float[Scalar, ""]],
        n_it: int,
        optim: optax.GradientTransformation,
        return_losses: bool
) -> Union[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]],
           Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "n_it"]]]:
    """Performs the gradient descent

    :param u_geos: U geostrophic velocity value
    :type u_geos: Float[Array, "lat lon"]
    :param v_geos: V geostrophic velocity value
    :type v_geos: Float[Array, "lat lon"]
    :param f: loss function
    :type f: Callable[[Float[Array, "lat lon"], Float[Array, "lat lon"]], Float[Scalar, ""]]
    :param n_it: maximum number of iterations
    :type n_it: int
    :param optim: optimizer to use
    :type optim: optax.GradientTransformation

    :returns: updated U and V cyclogeostrophic velocities; and eventually losses over iterations
    :rtype: Union[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]],
                  Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "n_it"]]]
    """
    u_cyclo, v_cyclo = jnp.copy(u_geos), jnp.copy(v_geos)
    opt_state = optim.init((u_cyclo, v_cyclo))
    losses = jnp.zeros(n_it)
    for i in jnp.arange(n_it):
        # update x and y using gradient descent
        u_cyclo, v_cyclo, opt_state, loss = _step(f, u_cyclo, v_cyclo, optim, opt_state)
        losses = losses.at[i].set(loss)

    if return_losses:
        res = u_cyclo, v_cyclo, losses
    else:
        res = u_cyclo, v_cyclo

    return res


def _variational(
        u_geos: Float[Array, "lat lon"],
        v_geos: Float[Array, "lat lon"],
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
) -> Union[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]],
           Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "n_it"]]]:
    """Computes the cyclogeostrophic balance using the variational method

    :param u_geos: U geostrophic velocity value
    :type u_geos: Float[Array, "lat lon"]
    :param v_geos: V geostrophic velocity value
    :type v_geos: Float[Array, "lat lon"]
    :param dx_u: U spatial step along x
    :type dx_u: Float[Array, "lat lon"]
    :param dx_v: V spatial step along x
    :type dx_v: Float[Array, "lat lon"]
    :param dy_u: U spatial step along y
    :type dy_u: Float[Array, "lat lon"]
    :param dy_v: V spatial step along y
    :type dy_v: Float[Array, "lat lon"]
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: Float[Array, "lat lon"]
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: Float[Array, "lat lon"]
    :param n_it: maximum number of iterations
    :type n_it: int
    :param optim: optimizer to use. Can be an optax.GradientTransformation optimizer, or a string referring to such an
                  optimizer.
    :type optim: Union[optax.GradientTransformation, str]
    :param optim_kwargs: optimizer arguments (such as learning rate, etc...)
    :type optim_kwargs: dict
    :param return_losses: should return losses over iteration?
    :type return_losses: bool

    :returns: U and V cyclogeostrophic velocities, and eventually losses over iterations
    :rtype: Union[Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]],
                  Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"], Float[Array, "n_it"]]]
    """
    if n_it is None:
        n_it = N_IT_VAR
    if isinstance(optim, str):
        if optim_kwargs is None:
            optim_kwargs = {"learning_rate": LR_VAR}
        optim = getattr(optax, optim)(**optim_kwargs)
    elif not isinstance(optim, optax.GradientTransformation):
        raise TypeError("optim should be an optax.GradientTransformation optimizer, or a string referring to such an "
                        "optimizer.")

    def f(uv: Tuple[Float[Array, "lat lon"], Float[Array, "lat lon"]]) -> Float[Array, "lat lon"]:
        u, v = uv
        adv_u = tools.compute_advection_u(u, v, dx_v, dy_v)
        adv_v = tools.compute_advection_v(u, v, dx_u, dy_u)
        return tools.compute_cyclogeostrophic_diff(u_geos, v_geos, u, v, adv_u, adv_v,
                                                   coriolis_factor_u, coriolis_factor_v)

    return _gradient_descent(u_geos, v_geos, f, n_it, optim, return_losses)
