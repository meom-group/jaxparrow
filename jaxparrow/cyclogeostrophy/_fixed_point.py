from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Bool, Float

from ._core import (
    CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result, _advection, _cyclogeostrophic_loss
)


def fixed_point(
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    ssh_t: Float[jax.Array, "y x"] = None,
    ug_t: Float[jax.Array, "y x"] = None,
    vg_t: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    return_geos: bool = False,
    return_losses: bool = False,
    n_it: int = 20,
    res_eps: float = 0.01
) -> CyclogeostrophyResult:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field
    using the fixed-point method [Penven et al. (2014)](https://doi.org/10.1002/2013JC009528).

    There are two modes of operation:

    1. **SSH mode**: Provide ``lat_t``, ``lon_t``, ``ssh_t`` (and optionally ``mask``).
       Geostrophic velocities will be computed from SSH.

    2. **Geostrophic mode**: Provide ``lat_t``, ``lon_t``, ``ug_t``, ``vg_t``
       (and optionally ``land_mask``). Geostrophic velocities are provided on the T grid.

    Parameters
    ----------
    lat_t : Float[jax.Array, "y x"]
        Latitude of the T grid
    lon_t : Float[jax.Array, "y x"]
        Longitude of the T grid
    ssh_t : Float[jax.Array, "y x"], optional
        SSH field (on the T grid)

        Defaults to `None`, required if geostrophic velocities are not provided
    ug_t : Float[jax.Array, "y x"], optional
        U component of geostrophic velocity on T grid
        
        Defaults to `None`, required if ``ssh_t`` is not provided
    vg_t : Float[jax.Array, "y x"], optional
        V component of geostrophic velocity on T grid
        
        Defaults to `None`, required if ``ssh_t`` is not provided
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)

        If not provided, inferred from ``ssh_t`` or ``ug_t`` `nan` values

        Defaults to `None`
    return_geos : bool, optional
        If `True`, returns the geostrophic SSC velocity field in addition to the cyclogeostrophic one.

        Defaults to `False`
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
        - ``ucg``: $u$ component of cyclogeostrophic velocity, on the T grid
        - ``vcg``: $v$ component of cyclogeostrophic velocity, on the T grid
        - ``ug``, ``vg``: Geostrophic velocities (if ``return_geos=True``)
        - ``losses``: Cyclogeostrophic imbalance per iteration (if ``return_losses=True``)
    """
    setup = setup_cyclogeostrophy(
        lat_t, lon_t, ssh_t=ssh_t, ug_t=ug_t, vg_t=vg_t, land_mask=land_mask
    )

    ucg, vcg, losses = _fixed_point(
        setup.ug_t, setup.vg_t,
        setup.dx_e_t, setup.dx_n_t, setup.dy_e_t, setup.dy_n_t, setup.J_t,
        setup.coriolis_factor_t,
        setup.land_mask, n_it, res_eps, return_losses
    )

    return assemble_result(
        ucg, vcg, setup, return_geos=return_geos, return_losses=return_losses, losses=losses
    )


@partial(jax.jit, static_argnames=("n_it"))
def _fixed_point(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
    n_it: int,
    res_eps: float,
    return_losses: bool
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"], Float[jax.Array, "n_it"]]:
    # define step partial: freeze constant over iterations
    def step_fn(carry, _):
        return _fp_step(
            ug_t, vg_t,
            dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t,
            coriolis_factor_t,
            land_mask,
            res_eps, return_losses,
            *carry
        )

    # apply updates
    (ucg, vcg, _, _), losses = lax.scan(
        step_fn,
        (ug_t, vg_t, (1 - land_mask).astype(bool), jnp.maximum(jnp.abs(ug_t), jnp.abs(vg_t))),
        xs=None, length=n_it
    )

    return ucg, vcg, losses


def _fp_step(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
    res_eps: float,
    return_losses: bool,
    u_n: Float[jax.Array, "y x"],
    v_n: Float[jax.Array, "y x"],
    mask_update: Bool[jax.Array, "y x"],
    res_n: Float[jax.Array, "y x"]
) -> tuple[
    tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"], Float[jax.Array, "y x"], Float[jax.Array, "y x"]], 
    float
]:
    # compute loss
    loss = lax.cond(
        return_losses,
        lambda: _cyclogeostrophic_loss(
            ug_t, vg_t, u_n, v_n, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, coriolis_factor_t, land_mask
        ),
        lambda: jnp.nan
    )

    # next it
    u_adv, v_adv = _advection(u_n, v_n, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, land_mask)
    u_np1 = ug_t - jnp.nan_to_num(v_adv / coriolis_factor_t, copy=False, nan=0, posinf=0, neginf=0)
    v_np1 = vg_t + jnp.nan_to_num(u_adv / coriolis_factor_t, copy=False, nan=0, posinf=0, neginf=0)

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
