from collections.abc import Callable
from functools import partial
import numbers
from typing import Literal, Tuple, Union

import jax
from jax import value_and_grad, jit
import jax.numpy as jnp
import numpy as np
import optax
from scipy import signal
from tqdm import tqdm

from .tools import tools

#: Default maximum number of iterations for Penven and Ioannou approaches
N_IT_IT = 100
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
# Entry point function
# =============================================================================

def cyclogeostrophy(u_geos: Union[np.ndarray, np.ma.MaskedArray], v_geos: Union[np.ndarray, np.ma.MaskedArray],
                    dx_u: np.ndarray, dx_v: np.ndarray, dy_u: np.ndarray, dy_v: np.ndarray,
                    coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray],
                    coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray],
                    method: Literal["variational", "iterative"] = "variational",
                    n_it: int = None, optim: Union[optax.GradientTransformation, str] = "sgd",
                    optim_kwargs: dict = None, res_eps: float = RES_EPS_IT,
                    res_init: Union[float, Literal["same"]] = RES_INIT_IT,
                    use_res_filter: bool = False, res_filter_size: int = RES_FILTER_SIZE_IT,
                    return_losses: bool = False) \
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes velocities from cyclogeostrophic approximation using a variational (default) or iterative method.

    :param u_geos: U geostrophic velocity, NxM grid
    :type u_geos: Union[np.ndarray, np.ma.MaskedArray]
    :param v_geos: V geostrophic velocity, NxM grid
    :type v_geos: Union[np.ndarray, np.ma.MaskedArray]
    :param dx_u: U spatial step along x, NxM grid
    :type dx_u: np.ndarray
    :param dx_v: V spatial step along x, NxM grid
    :type dx_v: np.ndarray
    :param dy_u: U spatial step along y, NxM grid
    :type dy_u: np.ndarray
    :param dy_v: V spatial step along y, NxM grid
    :type dy_v: np.ndarray
    :param coriolis_factor_u: U Coriolis factor, NxM grid
    :type coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray]
    :param coriolis_factor_v: V Coriolis factor, NxM grid
    :type coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray]
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
    :rtype: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    """
    mask = np.ma.getmaskarray(u_geos).astype(int)
    if isinstance(u_geos, np.ma.MaskedArray):
        u_geos = u_geos.filled(0)
    if isinstance(v_geos, np.ma.MaskedArray):
        v_geos = v_geos.filled(0)
    if isinstance(coriolis_factor_u, np.ma.MaskedArray):
        coriolis_factor_u = coriolis_factor_u.filled(1)
    if isinstance(coriolis_factor_v, np.ma.MaskedArray):
        coriolis_factor_v = coriolis_factor_v.filled(1)
    u_geos = np.nan_to_num(u_geos, nan=0, posinf=0, neginf=0)
    v_geos = np.nan_to_num(v_geos, nan=0, posinf=0, neginf=0)

    if method == "variational":
        res = _variational(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                           n_it, optim, optim_kwargs, return_losses)
    elif method == "iterative":
        res = _iterative(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                         mask, n_it, res_eps, res_init, use_res_filter, res_filter_size, return_losses)
    else:
        raise ValueError("method should be one of [\"variational\", \"iterative\"]")

    return res


# =============================================================================
# Iterative method
# =============================================================================

def _iterative(u_geos: np.ndarray, v_geos: np.ndarray, dx_u: np.ndarray, dx_v: np.ndarray,
               dy_u: np.ndarray, dy_v: np.ndarray, coriolis_factor_u: np.ndarray, coriolis_factor_v: np.ndarray,
               mask: np.ndarray, n_it: int, res_eps: float, res_init: Union[float, str], use_res_filter: bool,
               res_filter_size: int, return_losses: bool) \
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes velocities from cyclogeostrophic approximation using the iterative method from Penven et al. (2014)

    :param u_geos: U geostrophic velocity value
    :type u_geos: np.ndarray
    :param v_geos: V geostrophic velocity value
    :type v_geos: np.ndarray
    :param dx_u: U spatial step along x
    :type dx_u: np.ndarray
    :param dx_v: V spatial step along x
    :type dx_v: np.ndarray
    :param dy_u: U spatial step along y
    :type dy_u: np.ndarray
    :param dy_v: V spatial step along y
    :type dy_v: np.ndarray
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: np.ndarray
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: np.ndarray
    :param mask: initial data mask
    :type mask: np.ndarray
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
    :rtype: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    """
    def compute_cyclogeostrophic_diff(u_c, v_c, adv_u, adv_v, cf_u, cf_v, u_g, v_g):
        return np.sum((u_c + adv_v / cf_u - u_g)**2 + (v_c - adv_u / cf_v - v_g)**2)

    if n_it is None:
        n_it = N_IT_IT
    if res_init == "same":
        res_n = np.maximum(np.abs(u_geos), np.abs(v_geos))
    elif isinstance(res_init, numbers.Number):
        res_n = res_init * np.ones_like(u_geos)
    else:
        raise ValueError("res_init should be equal to \"same\" or be a number.")

    u_cyclo, v_cyclo = np.copy(u_geos), np.copy(v_geos)
    res_filter = np.ones((res_filter_size, res_filter_size))
    res_weights = signal.correlate(np.ones_like(u_geos), res_filter, mode="same")
    losses = np.zeros(n_it)
    i = 0
    for i in tqdm(range(n_it)):
        # next it
        advec_v = tools.compute_advection_v(u_cyclo, v_cyclo, dx_v, dy_v)
        advec_u = tools.compute_advection_u(u_cyclo, v_cyclo, dx_u, dy_u)
        u_np1 = u_geos - (1 / coriolis_factor_u) * np.nan_to_num(advec_v, nan=0, posinf=0, neginf=0)
        v_np1 = v_geos + (1 / coriolis_factor_v) * np.nan_to_num(advec_u, nan=0, posinf=0, neginf=0)

        # compute dist to u_cyclo and v_cyclo
        res_np1 = np.abs(u_np1 - u_cyclo) + np.abs(v_np1 - v_cyclo)
        if use_res_filter:
            res_np1 = signal.correlate(res_np1, res_filter, mode="same") / res_weights  # apply filter
        # compute intermediate masks
        mask_jnp1 = np.where(res_np1 < res_eps, 1, 0)
        mask_n = np.where(res_np1 > res_n, 1, 0)

        # compute loss
        if return_losses:
            losses[i] = compute_cyclogeostrophic_diff(u_cyclo, v_cyclo, advec_u, advec_v,
                                                      coriolis_factor_u, coriolis_factor_v, u_geos, v_geos)

        # update cyclogeostrophic velocities
        u_cyclo = mask * u_cyclo + (1 - mask) * (mask_n * u_cyclo + (1 - mask_n) * u_np1)
        v_cyclo = mask * v_cyclo + (1 - mask) * (mask_n * v_cyclo + (1 - mask_n) * v_np1)

        # update mask and residuals
        mask = np.maximum(mask, np.maximum(mask_jnp1, mask_n))
        res_n = res_np1

        if np.all(mask == 1):
            break

    if return_losses:
        res = u_cyclo, v_cyclo, losses[:i]
    else:
        res = u_cyclo, v_cyclo

    return res


# =============================================================================
# Variational method
# =============================================================================

@partial(jit, static_argnums=(0, 3))
def _step(f: Callable[[Tuple[jax.Array, jax.Array]], jax.Array], u_cyclo: jax.Array, v_cyclo: jax.Array,
          optim: optax.GradientTransformation, opt_state: optax.OptState) \
        -> Tuple[jax.Array, jax.Array, optax.OptState, jax.Array]:
    """Executes one iteration of the variational approach, using gradient descent

    :param f: loss function
    :type f: Callable[[np.ndarray, np.ndarray], jax.Array]
    :param u_cyclo: U cyclogeostrophic velocity value
    :type u_cyclo: np.ndarray
    :param v_cyclo: V cyclogeostrophic velocity value
    :type v_cyclo: np.ndarray
    :param optim: optimizer to use
    :type optim: optax.GradientTransformation

    :returns: updated U and V cyclogeostrophic velocities, optimizer state, and loss value
    :rtype: Tuple[jax.Array, jax.Array, optax.OptState, jax.Array]
    """
    params = (u_cyclo, v_cyclo)
    loss, grads = value_and_grad(f)(params)
    updates, opt_state = optim.update(grads, opt_state, params)
    u_n, v_n = optax.apply_updates(params, updates)
    return u_n, v_n, opt_state, loss


def _gradient_descent(u_geos: np.ndarray, v_geos: np.ndarray, f: Callable[[Tuple[jax.Array, jax.Array]], jax.Array],
                      n_it: int, optim: optax.GradientTransformation, return_losses: bool) \
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Performs the gradient descent

    :param u_geos: U geostrophic velocity value
    :type u_geos: np.ndarray
    :param v_geos: V geostrophic velocity value
    :type v_geos: np.ndarray
    :param f: loss function
    :type f: Callable[[np.ndarray, np.ndarray], jax.Array]
    :param n_it: maximum number of iterations
    :type n_it: int
    :param optim: optimizer to use
    :type optim: optax.GradientTransformation

    :returns: updated U and V cyclogeostrophic velocities; and eventually losses over iterations
    :rtype: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    """
    u_cyclo, v_cyclo = jnp.copy(u_geos), jnp.copy(v_geos)
    opt_state = optim.init((u_cyclo, v_cyclo))
    losses = np.zeros(n_it)
    for i in tqdm(range(n_it)):
        # update x and y using gradient descent
        u_cyclo, v_cyclo, opt_state, loss = _step(f, u_cyclo, v_cyclo, optim, opt_state)
        losses[i] = loss

    u_cyclo, v_cyclo = np.copy(u_cyclo), np.copy(v_cyclo)
    if return_losses:
        res = u_cyclo, v_cyclo, losses
    else:
        res = u_cyclo, v_cyclo

    return res


def _variational(u_geos: np.ndarray, v_geos: np.ndarray, dx_u: np.ndarray, dx_v: np.ndarray,
                 dy_u: np.ndarray, dy_v: np.ndarray, coriolis_factor_u: np.ndarray, coriolis_factor_v: np.ndarray,
                 n_it: int, optim: Union[optax.GradientTransformation, str], optim_kwargs: dict, return_losses: bool) \
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Computes the cyclogeostrophic balance using the variational method

    :param u_geos: U geostrophic velocity value
    :type u_geos: np.ndarray
    :param v_geos: V geostrophic velocity value
    :type v_geos: np.ndarray
    :param dx_u: U spatial step along x
    :type dx_u: np.ndarray
    :param dx_v: V spatial step along x
    :type dx_v: np.ndarray
    :param dy_u: U spatial step along y
    :type dy_u: np.ndarray
    :param dy_v: V spatial step along y
    :type dy_v: np.ndarray
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: np.ndarray
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: np.ndarray
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
    :rtype: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
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

    def f(uv: Tuple[jax.Array, jax.Array]) -> jax.Array:
        u, v = uv
        return tools.compute_cyclogeostrophic_diff_jax(u_geos, v_geos, u, v, dx_u, dx_v, dy_u, dy_v,
                                                       coriolis_factor_u, coriolis_factor_v)

    return _gradient_descent(u_geos, v_geos, f, n_it, optim, return_losses)
