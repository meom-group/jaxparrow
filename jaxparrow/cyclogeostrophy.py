from collections.abc import Callable
from functools import partial
from typing import Tuple, Union

import jax
from jax import value_and_grad, jit
import jax.numpy as jnp
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from jaxparrow.tools import geometry as geo


# =============================================================================
# Iterative method
# =============================================================================

#: Default maximum number of iterations for the iterative approach
N_IT_IT = 100
#: Default residual tolerance of the iterative approach
EPSILON_IT = 0.0001


def iterative(u_geos: Union[np.ndarray, np.ma.MaskedArray], v_geos: Union[np.ndarray, np.ma.MaskedArray],
              lat_u: Union[np.ndarray, np.ma.MaskedArray], lat_v: Union[np.ndarray, np.ma.MaskedArray],
              lon_u: Union[np.ndarray, np.ma.MaskedArray], lon_v: Union[np.ndarray, np.ma.MaskedArray],
              coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray],
              coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray],
              n_it: int = N_IT_IT, eps: float = EPSILON_IT, errsq_init: float | str = np.inf, err_filter_size: int=1,
              run_inspection: bool = False) \
        -> Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]:
    """
    Computes velocities from cyclogeostrophic approximation using the iterative method from Penven et al. (2014)

    :param u_geos: U geostrophic velocity value
    :type u_geos: Union[np.ndarray, np.ma.MaskedArray]
    :param v_geos: V geostrophic velocity value
    :type v_geos: Union[np.ndarray, np.ma.MaskedArray]
    :param lat_u: U latitude
    :type lat_u: Union[np.ndarray, np.ma.MaskedArray]
    :param lat_v: V latitude
    :type lat_v: Union[np.ndarray, np.ma.MaskedArray]
    :param lon_u: U longitude
    :type lon_u: Union[np.ndarray, np.ma.MaskedArray]
    :param lon_v: V longitude
    :type lon_v: Union[np.ndarray, np.ma.MaskedArray]
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray]
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray]
    :param n_it: maximum number of iterations, defaults to N_IT_IT
    :type n_it: int, optional
    :param eps: residual tolerance, defaults to EPSILON_IT
    :type eps: float, optional

    :returns: U and V cyclogeostrophic velocities
    :rtype: Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]
    """
    dx_u, dy_u = geo.compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = geo.compute_spatial_step(lat_v, lon_v)

    u_cyclo, v_cyclo = np.copy(u_geos), np.copy(v_geos)
    if run_inspection:
        errsq_hist = [None] * n_it
        losses = np.ones(n_it+1) * np.asarray(_loss(u_geos, v_geos, u_cyclo, v_cyclo, dx_u, dx_v, dy_u, dy_v,
                                                    coriolis_factor_u, coriolis_factor_v))
    mask = np.zeros_like(u_geos)
    err_filter = np.ones((err_filter_size, err_filter_size))
    if errsq_init == "proportional":
        errsq_n = (np.abs(u_geos) + np.abs(v_geos)) / 2
    else:
        errsq_n = errsq_init * np.ones_like(u_geos)
    errsq_eps = eps * np.ones_like(u_geos)
    for i in tqdm(range(n_it)):
        # next it
        advec_v = geo.compute_advection_v(u_cyclo, v_cyclo, dx_v, dy_v)
        advec_u = geo.compute_advection_u(u_cyclo, v_cyclo, dx_u, dy_u)
        u_np1 = u_geos - (1 / coriolis_factor_u) * advec_v
        v_np1 = v_geos + (1 / coriolis_factor_v) * advec_u

        # compute dist to u_cyclo and v_cyclo
        errsq_np1 = np.square(u_np1 - u_cyclo) + np.square(v_np1 - v_cyclo)
        errsq_np1 = ndimage.convolve(errsq_np1, err_filter, mode="nearest") / err_filter.size  # apply convolution
        # compute intermediate masks
        mask_jnp1 = np.where(errsq_np1 < errsq_eps, 1, 0)
        mask_n = np.where(errsq_np1 > errsq_n, 1, 0)

        # update cyclogeostrophic velocities
        u_cyclo = mask * u_cyclo + (1 - mask) * (mask_n * u_cyclo + (1 - mask_n) * u_np1)
        v_cyclo = mask * v_cyclo + (1 - mask) * (mask_n * v_cyclo + (1 - mask_n) * v_np1)

        # inspect results
        if run_inspection:
            errsq_hist[i] = np.histogram(errsq_np1.flatten(), bins="doane")
            losses[i+1] = _loss(u_geos, v_geos, u_cyclo, v_cyclo, dx_u, dx_v, dy_u, dy_v,
                                coriolis_factor_u, coriolis_factor_v)

        # update mask
        mask = np.maximum(mask, np.maximum(mask_jnp1, mask_n))

        errsq_n = errsq_np1

        if np.all(mask == 1):
            break

    if run_inspection:
        return u_cyclo, v_cyclo, errsq_hist[:n_it], losses[:n_it+1]
    else:
        return u_cyclo, v_cyclo


# =============================================================================
# Variational method
# =============================================================================

#: Default maximum number of iterations for the variational approach
N_IT_VAR = 2000
#: Default learning rate for the gradient descent of the variational approach
LR_VAR = 0.005
#: Patience after which the learning starts to decrease
LR_PATIENCE = 10
#: Patience after which the learning starts to decrease
LR_DECAY = .9
#: Patience after which the gradient descent algorithm is interrupted
ES_PATIENCE = 100


@partial(jit, static_argnums=0)
def _step(f: Callable[[jax.Array, jax.Array], jax.Array], u_cyclo: jax.Array, v_cyclo: jax.Array, lr: float) \
        -> Tuple[jax.Array, jax.Array, float]:
    """Executes one iteration of the variational approach, using gradient descent

    :param f: loss function
    :type f: Callable[[np.ndarray, np.ndarray], jax.Array]
    :param u_cyclo: U cyclogeostrophic velocity value
    :type u_cyclo: np.ndarray
    :param v_cyclo: V cyclogeostrophic velocity value
    :type v_cyclo: np.ndarray
    :param lr: gradient descent learning rate
    :type lr: float

    :returns: updated U and V cyclogeostrophic velocities
    :rtype: Tuple[jax.Array, jax.Array, float]
    """
    grad_fn = value_and_grad(f, argnums=(0, 1))

    loss_val, (grad_u, grad_v) = grad_fn(u_cyclo, v_cyclo)
    grad_u = jnp.nan_to_num(grad_u)
    grad_v = jnp.nan_to_num(grad_v)

    u_n = u_cyclo - lr * grad_u
    v_n = v_cyclo - lr * grad_v

    return u_n, v_n, loss_val


def _loss(u_geos: np.ndarray, v_geos: np.ndarray,
          u_cyclo: Union[np.ndarray, jax.Array], v_cyclo: Union[np.ndarray, jax.Array],
          dx_u: np.ndarray, dx_v: np.ndarray, dy_u: np.ndarray, dy_v: np.ndarray,
          coriolis_factor_u: np.ndarray, coriolis_factor_v: np.ndarray) -> jax.Array:
    """Computes the loss

    :param u_geos: U geostrophic velocity value
    :type u_geos: np.ndarray
    :param v_geos: V geostrophic velocity value
    :type v_geos: np.ndarray
    :param u_cyclo: U cyclogeostrophic velocity value
    :type u_cyclo: Union[np.ndarray, jax.Array]
    :param v_cyclo: V cyclogeostrophic velocity value
    :type v_cyclo: Union[np.ndarray, jax.Array]
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

    :returns: the loss
    :rtype: jax.Array
    """
    J_u = np.sum(
        (u_cyclo + geo.compute_advection_v_jax(u_cyclo, v_cyclo, dx_v, dy_v) / coriolis_factor_u - u_geos) ** 2)
    J_v = np.sum(
        (v_cyclo - geo.compute_advection_u_jax(u_cyclo, v_cyclo, dx_u, dy_u) / coriolis_factor_v - v_geos) ** 2)
    return J_u + J_v


def _gradient_descent(u_geos: Union[np.ndarray, np.ma.MaskedArray], v_geos: Union[np.ndarray, np.ma.MaskedArray],
                      f: Callable[[jax.Array, jax.Array], jax.Array], n_it: int, lr: float,
                      lr_patience: int, lr_decay: float, es_patience: int) -> Tuple[np.ndarray, np.ndarray]:
    """Performs the gradient descent

    :param u_geos: U geostrophic velocity value
    :type u_geos: Union[np.ndarray, np.ma.MaskedArray]
    :param v_geos: V geostrophic velocity value
    :type v_geos: Union[np.ndarray, np.ma.MaskedArray]
    :param f: loss function
    :type f: Callable[[np.ndarray, np.ndarray], jax.Array]
    :param n_it: maximum number of iterations, defaults to N_IT_VAR
    :type n_it: int, optional
    :param lr: gradient descent learning rate, defaults to LR_VAR
    :type lr: float, optional
    """
    u_cyclo, v_cyclo = jnp.copy(u_geos), jnp.copy(v_geos)
    n_worse = 0
    min_loss = np.inf
    for _ in tqdm(range(n_it)):
        # update x and y using gradient descent
        u_cyclo, v_cyclo, loss_val = _step(f, u_cyclo, v_cyclo, lr)

        # based on the loss value, update the lr, store best estimations, and stop early
        if loss_val >= min_loss:
            n_worse += 1
        else:
            min_loss = loss_val
            n_worse = 0

        if n_worse >= lr_patience:
            lr *= lr_decay

        if lr == 0 or n_worse >= es_patience:
            break

    return np.asarray(u_cyclo), np.asarray(v_cyclo)


def variational(u_geos: Union[np.ndarray, np.ma.MaskedArray], v_geos: Union[np.ndarray, np.ma.MaskedArray],
                dx_u: np.ndarray, dx_v: np.ndarray, dy_u: np.ndarray, dy_v: np.ndarray,
                coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray],
                coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray],
                n_it: int = N_IT_VAR, lr: float = LR_VAR,
                lr_patience: int = LR_PATIENCE, lr_decay: float = LR_DECAY, es_patience: int = ES_PATIENCE) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Computes the cyclogeostrophic balance using the variational method

    :param u_geos: U geostrophic velocity value
    :type u_geos: Union[np.ndarray, np.ma.MaskedArray]
    :param v_geos: V geostrophic velocity value
    :type v_geos: Union[np.ndarray, np.ma.MaskedArray]
    :param dx_u: U spatial step along x
    :type dx_u: np.ndarray
    :param dx_v: V spatial step along x
    :type dx_v: np.ndarray
    :param dy_u: U spatial step along y
    :type dy_u: np.ndarray
    :param dy_v: V spatial step along y
    :type dy_v: np.ndarray
    :param coriolis_factor_u: U Coriolis factor
    :type coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray]
    :param coriolis_factor_v: V Coriolis factor
    :type coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray]
    :param n_it: maximum number of iterations, defaults to N_IT_VAR
    :type n_it: int, optional
    :param lr: gradient descent learning rate, defaults to LR_VAR
    :type lr: float, optional

    :returns: U and V cyclogeostrophic velocities
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    if isinstance(u_geos, np.ma.MaskedArray):
        u_geos = u_geos.filled(0)
    if isinstance(v_geos, np.ma.MaskedArray):
        v_geos = v_geos.filled(0)
    if isinstance(coriolis_factor_u, np.ma.MaskedArray):
        coriolis_factor_u = coriolis_factor_u.filled(1)
    if isinstance(coriolis_factor_v, np.ma.MaskedArray):
        coriolis_factor_v = coriolis_factor_v.filled(1)

    def f(u: jax.Array, v: jax.Array) -> jax.Array:
        return _loss(u_geos, v_geos, u, v, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v)

    return _gradient_descent(u_geos, v_geos, f, n_it, lr, lr_patience, lr_decay, es_patience)
