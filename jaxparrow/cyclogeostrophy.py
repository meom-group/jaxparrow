from collections.abc import Callable
from functools import partial
from typing import Tuple, Union

import jax
from jax import grad, jit
import jax.numpy as jnp
import numpy as np
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
              dx_u: Union[np.ndarray, np.ma.MaskedArray], dx_v: Union[np.ndarray, np.ma.MaskedArray],
              dy_u: Union[np.ndarray, np.ma.MaskedArray], dy_v: Union[np.ndarray, np.ma.MaskedArray],
              coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray],
              coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray],
              n_it: int = N_IT_IT, eps: float = EPSILON_IT) \
        -> Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]:
    """
    Computes velocities from cyclogeostrophic approximation using the iterative method from Penven et al. (2014)

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
    :param n_it: maximum number of iterations, defaults to N_IT_IT
    :type n_it: int, optional
    :param eps: residual tolerance, defaults to EPSILON_IT
    :type eps: float, optional

    :returns: U and V cyclogeostrophic velocities
    :rtype: Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]
    """
    u_cyclo, v_cyclo = np.copy(u_geos), np.copy(v_geos)
    mask = np.zeros_like(u_geos)
    errsq = np.inf * np.ones_like(u_geos)
    arreps = eps * np.ones_like(u_geos)
    n_iter = 0
    while np.any(mask == 0) and n_iter < n_it:
        # next it
        advec_v = geo.compute_advection_v(u_cyclo, v_cyclo, dx_v, dy_v)
        advec_u = geo.compute_advection_u(u_cyclo, v_cyclo, dx_u, dy_u)
        u_n = u_geos - (1 / coriolis_factor_u) * advec_v
        v_n = v_geos + (1 / coriolis_factor_v) * advec_u

        # update stopping criterion mask (point wise)
        errsq_n = np.square(u_n - u_cyclo) + np.square(v_n - v_cyclo)
        cond_1 = np.where(errsq_n < arreps, 1, 0)
        cond_2 = np.where(errsq_n > errsq, 1, 0)
        mask = np.maximum(mask, np.maximum(cond_1, cond_2))  # TODO: should it be done at the end of the iteration step?

        # update cyclogeostrophic velocities where it should be
        u_cyclo = mask * u_cyclo + (1 - mask) * u_n
        v_cyclo = mask * v_cyclo + (1 - mask) * v_n

        n_iter += 1
        errsq = errsq_n
    return u_cyclo, v_cyclo


# =============================================================================
# Variational method
# =============================================================================

#: Default maximum number of iterations for the variational approach
N_IT_VAR = 2000
#: Default learning rate for the gradient descent of the variational approach
LR_VAR = 0.005


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


def variational(u_geos: Union[np.ndarray, np.ma.MaskedArray], v_geos: Union[np.ndarray, np.ma.MaskedArray],
                dx_u: np.ndarray, dx_v: np.ndarray, dy_u: np.ndarray, dy_v: np.ndarray,
                coriolis_factor_u: Union[np.ndarray, np.ma.MaskedArray],
                coriolis_factor_v: Union[np.ndarray, np.ma.MaskedArray],
                n_it: int = N_IT_VAR, lr: float = LR_VAR) -> Tuple[np.ndarray, np.ndarray]:
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

    u_cyclo, v_cyclo = jnp.copy(u_geos), jnp.copy(v_geos)
    for _ in tqdm(range(n_it)):
        # update x and y using gradient descent
        u_cyclo, v_cyclo = _step(f, u_cyclo, v_cyclo, lr)

    return np.asarray(u_cyclo), np.asarray(v_cyclo)
