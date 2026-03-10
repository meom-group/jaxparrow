from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Bool, Float

from ._core import (
    CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result,
    _advection, _cyclogeostrophic_loss
)


def fixed_point(
    lat_t: Float[jax.Array, "lat lon"],
    lon_t: Float[jax.Array, "lat lon"],
    ssh_t: Float[jax.Array, "lat lon"] = None,
    ug_t: Float[jax.Array, "lat lon"] = None,
    vg_t: Float[jax.Array, "lat lon"] = None,
    mask: Float[jax.Array, "lat lon"] = None,
    return_geos: bool = False,
    return_grids: bool = True,
    return_losses: bool = False,
    n_it: int = 20,
    res_eps: float = 0.01
) -> CyclogeostrophyResult:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field
    using the fixed-point method [Penven et al. (2014)](https://doi.org/10.1002/2013JC009528).

    The cyclogeostrophic SSC velocity field is computed on a C-grid, following NEMO convention.

    There are two modes of operation:

    1. **SSH mode**: Provide ``lat_t``, ``lon_t``, ``ssh_t`` (and optionally ``mask``).
       Geostrophic velocities will be computed from SSH.

    2. **Geostrophic mode**: Provide ``lat_t``, ``lon_t``, ``ug_t``, ``vg_t``
       (and optionally ``mask``). Geostrophic velocities are provided on the T grid
       and will be interpolated to U/V grids internally.

    Parameters
    ----------
    lat_t : Float[jax.Array, "lat lon"]
        Latitude of the T grid.
    lon_t : Float[jax.Array, "lat lon"]
        Longitude of the T grid.
    ssh_t : Float[jax.Array, "lat lon"], optional
        SSH field (on the T grid). Required if geostrophic velocities are not provided.
    ug_t : Float[jax.Array, "lat lon"], optional
        U component of geostrophic velocity on T grid. If provided with ``vg_t``,
        bypasses SSH-based computation. Will be interpolated to U grid.
    vg_t : Float[jax.Array, "lat lon"], optional
        V component of geostrophic velocity on T grid. If provided with ``ug_t``,
        bypasses SSH-based computation. Will be interpolated to V grid.
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` or ``ug_t`` `nan` values

        Defaults to `None`
    return_geos : bool, optional
        If `True`, returns the geostrophic SSC velocity field in addition to the cyclogeostrophic one.

        Defaults to `False`
    return_grids : bool, optional
        If `True`, returns the U and V grids.

        Defaults to `True`
    return_losses : bool, optional
        If `True`, returns the losses (cyclogeostrophic imbalance) over iterations.

        Defaults to `False`
    n_it : int, optional
        Maximum number of iterations.

        Defaults to `20`
    res_eps : float, optional
        Residual tolerance of the iterative approach.
        When residuals are smaller, the iterative approach considers local convergence to cyclogeostrophy.

        Defaults to `0.01`

    Returns
    -------
    CyclogeostrophyResult
        Named tuple containing:
        - ``ucg``: U component of cyclogeostrophic velocity (on U grid)
        - ``vcg``: V component of cyclogeostrophic velocity (on V grid)
        - ``ug``, ``vg``: Geostrophic velocities (if ``return_geos=True``)
        - ``lat_u``, ``lon_u``, ``lat_v``, ``lon_v``: Grid coordinates (if ``return_grids=True``)
        - ``losses``: Cyclogeostrophic imbalance per iteration (if ``return_losses=True``)
    """
    setup = setup_cyclogeostrophy(
        lat_t, lon_t, ssh_t=ssh_t, ug_t=ug_t, vg_t=vg_t, mask=mask
    )

    ucg_u, vcg_v, losses = _fixed_point(
        setup.ug_u, setup.vg_v,
        setup.dx_u, setup.dx_v, setup.dy_u, setup.dy_v,
        setup.coriolis_factor_u, setup.coriolis_factor_v,
        setup.grid_angle_u, setup.grid_angle_v,
        setup.is_land, n_it, res_eps, return_losses
    )

    return assemble_result(
        ucg_u, vcg_v, setup,
        return_geos=return_geos,
        return_grids=return_grids,
        return_losses=return_losses,
        losses=losses
    )


@partial(jax.jit, static_argnames=("n_it"))
def _fixed_point(
    ug_u: Float[jax.Array, "lat lon"],
    vg_v: Float[jax.Array, "lat lon"],
    dx_u: Float[jax.Array, "lat lon"],
    dx_v: Float[jax.Array, "lat lon"],
    dy_u: Float[jax.Array, "lat lon"],
    dy_v: Float[jax.Array, "lat lon"],
    coriolis_factor_u: Float[jax.Array, "lat lon"],
    coriolis_factor_v: Float[jax.Array, "lat lon"],
    grid_angle_u: Float[jax.Array, "lat lon"],
    grid_angle_v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    n_it: int,
    res_eps: float,
    return_losses: bool
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"], Float[jax.Array, "n_it"]]:
    # define step partial: freeze constant over iterations
    def step_fn(carry, _):
        return _fp_step(
            ug_u, vg_v,
            dx_u, dx_v, dy_u, dy_v,
            coriolis_factor_u, coriolis_factor_v,
            grid_angle_u, grid_angle_v,
            mask,
            res_eps, return_losses,
            *carry
        )

    # apply updates
    (ucg, vcg, _, _), losses = lax.scan(
        step_fn,
        (ug_u, vg_v, (1 - mask).astype(bool), jnp.maximum(jnp.abs(ug_u), jnp.abs(vg_v))),
        xs=None, length=n_it
    )

    return ucg, vcg, losses


def _fp_step(
    ug_u: Float[jax.Array, "lat lon"],
    vg_v: Float[jax.Array, "lat lon"],
    dx_u: Float[jax.Array, "lat lon"],
    dx_v: Float[jax.Array, "lat lon"],
    dy_u: Float[jax.Array, "lat lon"],
    dy_v: Float[jax.Array, "lat lon"],
    coriolis_factor_u: Float[jax.Array, "lat lon"],
    coriolis_factor_v: Float[jax.Array, "lat lon"],
    grid_angle_u: Float[jax.Array, "lat lon"],
    grid_angle_v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    res_eps: float,
    return_losses: bool,
    u_n: Float[jax.Array, "lat lon"],
    v_n: Float[jax.Array, "lat lon"],
    mask_update: Bool[jax.Array, "lat lon"],
    res_n: Float[jax.Array, "lat lon"]
) -> tuple[
    tuple[
        Float[jax.Array, "lat lon"], 
        Float[jax.Array, "lat lon"], 
        Float[jax.Array, "lat lon"], 
        Float[jax.Array, "lat lon"]
    ], 
    float
]:
    # compute loss
    loss = lax.cond(
        return_losses,
        lambda: _cyclogeostrophic_loss(
            ug_u, vg_v, u_n, v_n, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, mask,
            grid_angle_u, grid_angle_v
        ),
        lambda: jnp.nan
    )

    # next it
    u_adv_v, v_adv_u = _advection(u_n, v_n, dx_u, dx_v, dy_u, dy_v, mask, grid_angle_u, grid_angle_v)
    u_np1 = ug_u - jnp.nan_to_num(v_adv_u / coriolis_factor_u, copy=False, nan=0, posinf=0, neginf=0)
    v_np1 = vg_v + jnp.nan_to_num(u_adv_v / coriolis_factor_v, copy=False, nan=0, posinf=0, neginf=0)

    # compute dist to ucg and vcg
    res_np1 = jnp.abs(u_np1 - u_n) + jnp.abs(v_np1 - v_n)  # norm1

    # compute stopping criterion masks
    mask_not_div = jnp.where(res_np1 <= res_n, True, False)
    mask_not_conv = jnp.where(res_np1 >= res_eps, True, False)
  
    # update cyclogeostrophic velocities and residuals where it is not diverging
    mask_update &= mask_not_div
    u_n = jnp.where(mask_update, u_np1, u_n)
    v_n = jnp.where(mask_update, v_np1, v_n)
    res_n = jnp.where(mask_update, res_np1, res_n)

    # update stopping criterion mask where it has converged
    mask_update &= mask_not_conv

    return (u_n, v_n, mask_update, res_n), loss
