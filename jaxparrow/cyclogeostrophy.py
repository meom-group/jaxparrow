from collections.abc import Callable
from functools import partial
import numbers
from typing import Literal, Tuple, Union

import jax
from jax import grad, jit
import jax.numpy as jnp
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from tools import tools

#: Default maximum number of iterations for the iterative approach
N_IT_IT = 100
#: Default residual tolerance of the iterative approach
RES_EPS_IT = 0.0001
#: Default residual value used during the first iteration
RES_INIT_IT = "same"
#: Default size of the grid points used to compute the residual in Ioannou's iterative approach
RES_FILTER_SIZE_IT = 3

#: Default maximum number of iterations for the variational approach
N_IT_VAR = 2000
#: Default learning rate for the gradient descent of the variational approach
LR_VAR = 0.005

__all__ = ["cyclogeostrophy", "LR_VAR", "N_IT_IT", "N_IT_VAR", "RES_EPS_IT", "RES_INIT_IT", "RES_FILTER_SIZE_IT"]


# =============================================================================
# Entry point function
# =============================================================================

def cyclogeostrophy(u_geos: Union[np.ndarray, np.ma.MaskedArray], v_geos: Union[np.ndarray, np.ma.MaskedArray],
                    dx_u: np.ndarray, dx_v: np.ndarray, dy_u: np.ndarray, dy_v: np.ndarray,
                    coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray],
                    coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray],
                    method: Literal["variational", "penven", "ioannou"] = "variational",
                    n_it: int = None, lr: float = LR_VAR, res_eps: float = RES_EPS_IT,
                    res_init: Union[float | Literal["same"]] = RES_INIT_IT, res_filter_size: int = RES_FILTER_SIZE_IT) \
        -> Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]:
    """
    Computes velocities from cyclogeostrophic approximation using a variational (default) or iterative method.

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
    :param method: numerical method to use, defaults to "variational"
    :type method: Literal["variational", "penven", "ioannou"], optional
    :param n_it: maximum number of iterations, defaults to N_IT_IT
    :type n_it: int, optional
    :param lr: gradient descent learning rate, defaults to LR_VAR
    :type lr: float, optional
    :param res_eps: residual tolerance: if residuals are smaller, we consider them as equal to 0, defaults to EPS_IT
    :type res_eps: float, optional
    :param res_init: residual initial value: if residuals are larger at the first iteration, we consider that the
                     solution diverges. If equals to "same" (default) absolute values of the geostrophic velocities are
                     used. Defaults to RES_INIT_IT
    :type res_init: Union[float | Literal["same"]], optional
    :param res_filter_size: size of the convolution filter (from Ioannou) used when computing the residuals,
                            defaults to RES_FILTER_SIZE_IT
    :type res_filter_size: int, optional

    :returns: U and V cyclogeostrophic velocities
    :rtype: Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]
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

    if method == "variational":
        u_cyclo, v_cyclo = _variational(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                        n_it, lr)
    elif method == "penven":
        u_cyclo, v_cyclo = _iterative(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                      mask, n_it, res_eps, res_init, res_filter_size=1)
    elif method == "ioannou":
        u_cyclo, v_cyclo = _iterative(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                      mask, n_it, res_eps, res_init, res_filter_size)
    else:
        raise ValueError("method should be one of [\"variational\", \"penven\", \"ioannou\"]")

    return u_cyclo, v_cyclo


# =============================================================================
# Iterative method
# =============================================================================

def _iterative(u_geos: np.ndarray, v_geos: np.ndarray, dx_u: np.ndarray, dx_v: np.ndarray,
               dy_u: np.ndarray, dy_v: np.ndarray, coriolis_factor_u: np.ndarray, coriolis_factor_v: np.ndarray,
               mask: np.ndarray, n_it: int = N_IT_IT, res_eps: float = RES_EPS_IT, res_init: float | str = RES_INIT_IT,
               res_filter_size: int = RES_FILTER_SIZE_IT) \
        -> Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]:
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
    :param n_it: maximum number of iterations, defaults to N_IT_IT
    :type n_it: int, optional
    :param res_eps: residual tolerance: if residuals are smaller, we consider them as equal to 0, defaults to EPS_IT
    :type res_eps: float, optional
    :param res_init: residual initial value: if residuals are larger at the first iteration, we consider that the
                     solution diverges. If equals to "same" (default) absolute values of the geostrophic velocities are
                     used. Defaults to RES_INIT_IT
    :type res_init: float | str, optional
    :param res_filter_size: size of the convolution filter (from Ioannou) used when computing the residuals,
                            defaults to RES_FILTER_SIZE_IT
    :type res_filter_size: int, optional

    :returns: U and V cyclogeostrophic velocities
    :rtype: Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]
    """
    if n_it is None:
        n_it = N_IT_IT
    if res_init == "same":
        res_n = np.maximum(np.abs(u_geos), np.abs(v_geos))
    elif isinstance(res_init, numbers.Number):
        res_n = res_init * np.ones_like(u_geos)
    else:
        raise ValueError("res_init should be equal to \"same\" or be a float.")

    u_cyclo, v_cyclo = np.copy(u_geos), np.copy(v_geos)
    res_filter = np.ones((res_filter_size, res_filter_size))
    res_esp = res_eps * np.ones_like(u_geos)
    for _ in tqdm(range(n_it)):
        # next it
        advec_v = tools.compute_advection_v(u_cyclo, v_cyclo, dx_v, dy_v)
        advec_u = tools.compute_advection_u(u_cyclo, v_cyclo, dx_u, dy_u)
        u_np1 = u_geos - (1 / coriolis_factor_u) * advec_v
        v_np1 = v_geos + (1 / coriolis_factor_v) * advec_u

        # compute dist to u_cyclo and v_cyclo
        res_np1 = np.square(u_np1 - u_cyclo) + np.square(v_np1 - v_cyclo)
        res_np1 = ndimage.convolve(res_np1, res_filter, mode="nearest") / res_filter.size  # apply convolution
        # compute intermediate masks
        mask_jnp1 = np.where(res_np1 < res_esp, 1, 0)
        mask_n = np.where(res_np1 > res_n, 1, 0)

        # update cyclogeostrophic velocities
        u_cyclo = mask * u_cyclo + (1 - mask) * (mask_n * u_cyclo + (1 - mask_n) * u_np1)
        v_cyclo = mask * v_cyclo + (1 - mask) * (mask_n * v_cyclo + (1 - mask_n) * v_np1)

        # update mask and residuals
        mask = np.maximum(mask, np.maximum(mask_jnp1, mask_n))
        res_n = res_np1

        if np.all(mask == 1):
            break

    return u_cyclo, v_cyclo


# =============================================================================
# Variational method
# =============================================================================

@partial(jit, static_argnums=(0, 3))
def _step(f: Callable[[jax.Array, jax.Array], jax.Array], u_cyclo: jax.Array, v_cyclo: jax.Array, lr: float) \
        -> Tuple[jax.Array, jax.Array]:
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
    :rtype: Tuple[jax.Array, jax.Array]
    """
    grad_u = grad(f)
    grad_v = grad(f, argnums=1)

    u_n = u_cyclo - lr * grad_u(u_cyclo, v_cyclo)
    v_n = v_cyclo - lr * grad_v(u_cyclo, v_cyclo)

    return u_n, v_n


def _gradient_descent(u_geos: np.ndarray, v_geos: np.ndarray, f: Callable[[jax.Array, jax.Array], jax.Array],
                      n_it: int, lr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Performs the gradient descent

    :param u_geos: U geostrophic velocity value
    :type u_geos: np.ndarray
    :param v_geos: V geostrophic velocity value
    :type v_geos: np.ndarray
    :param f: loss function
    :type f: Callable[[np.ndarray, np.ndarray], jax.Array]
    :param n_it: maximum number of iterations, defaults to N_IT_VAR
    :type n_it: int, optional
    :param lr: gradient descent learning rate, defaults to LR_VAR
    :type lr: float, optional
    """
    u_cyclo, v_cyclo = jnp.copy(u_geos), jnp.copy(v_geos)
    for _ in tqdm(range(n_it)):
        # update x and y using gradient descent
        u_cyclo, v_cyclo = _step(f, u_cyclo, v_cyclo, lr)
    return np.copy(u_cyclo), np.copy(v_cyclo)


def _variational(u_geos: np.ndarray, v_geos: np.ndarray, dx_u: np.ndarray, dx_v: np.ndarray,
                 dy_u: np.ndarray, dy_v: np.ndarray, coriolis_factor_u: np.ndarray, coriolis_factor_v: np.ndarray,
                 n_it: int = N_IT_VAR, lr: float = LR_VAR) -> Tuple[np.ndarray, np.ndarray]:
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
    :param n_it: maximum number of iterations, defaults to N_IT_VAR
    :type n_it: int, optional
    :param lr: gradient descent learning rate, defaults to LR_VAR
    :type lr: float, optional

    :returns: U and V cyclogeostrophic velocities
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    if n_it is None:
        n_it = N_IT_VAR

    def f(u: jax.Array, v: jax.Array) -> jax.Array:
        return tools.compute_cyclogeostrophic_diff_jax(u_geos, v_geos, u, v, dx_u, dx_v, dy_u, dy_v,
                                                       coriolis_factor_u, coriolis_factor_v)

    return _gradient_descent(u_geos, v_geos, f, n_it, lr)
