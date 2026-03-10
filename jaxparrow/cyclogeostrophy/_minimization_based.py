from collections.abc import Callable
from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Float
import optax

from ._core import CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result, _cyclogeostrophic_loss


def minimization_based(
    lat_t: Float[jax.Array, "lat lon"],
    lon_t: Float[jax.Array, "lat lon"],
    ssh_t: Float[jax.Array, "lat lon"] = None,
    ug_t: Float[jax.Array, "lat lon"] = None,
    vg_t: Float[jax.Array, "lat lon"] = None,
    mask: Float[jax.Array, "lat lon"] = None,
    return_geos: bool = False,
    return_grids: bool = True,
    return_losses: bool = False,
    n_it: int = 2000,
    optim: optax.GradientTransformation | str = "sgd",
    optim_kwargs: dict = None
) -> CyclogeostrophyResult:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field
    using our minimization-based method.

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

        Defaults to `2000`
    optim : Union[optax.GradientTransformation, str], optional
        Optimizer to use.
        Can be an ``optax.GradientTransformation`` optimizer, or a ``string`` referring to such an optimizer.

        Defaults to `sgd`
    optim_kwargs : dict, optional
        Optimizer arguments (such as learning rate, etc...).

        If `None`, only the learning rate is enforced to `0.005`

        Defaults to `None`

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

    if isinstance(optim, str):
        if optim_kwargs is None:
            optim_kwargs = {"learning_rate": 0.005}
        optim = getattr(optax, optim)(**optim_kwargs)
    elif not isinstance(optim, optax.GradientTransformation):
        raise TypeError(
            "optim should be an optax.GradientTransformation optimizer, or a string referring to such an optimizer."
        )
    
    ucg_u, vcg_v, losses = _minimization_based(
        setup.ug_u, setup.vg_v,
        setup.dx_u, setup.dx_v, setup.dy_u, setup.dy_v,
        setup.coriolis_factor_u, setup.coriolis_factor_v,
        setup.grid_angle_u, setup.grid_angle_v,
        setup.is_land, n_it, optim
    )

    return assemble_result(
        ucg_u, vcg_v, setup,
        return_geos=return_geos,
        return_grids=return_grids,
        return_losses=return_losses,
        losses=losses,
    )


@partial(jax.jit, static_argnames=("n_it", "optim"))
def _minimization_based(
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
    optim: optax.GradientTransformation
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"], Float[jax.Array, "n_it"]]:
    def loss_fn(args):
        ucg_u, vcg_v = args
        return _cyclogeostrophic_loss(
            ug_u, vg_v, ucg_u, vcg_v, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
            mask, grid_angle_u, grid_angle_v
        )
    
    def step_fn(carry, _):
        params = carry[:-1]
        opt_state = carry[-1]
    
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = tuple(map(lambda x: jnp.nan_to_num(x, copy=False, nan=0, posinf=0, neginf=0), grads))

        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params + (opt_state,), loss
    
    carry, losses = lax.scan(
        step_fn,
        (ug_u, vg_v, optim.init((ug_u, vg_v))),
        xs=None, length=n_it
    )
    
    return *carry[:-1], losses
