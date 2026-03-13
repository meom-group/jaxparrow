from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Float
import optax

from ._core import CyclogeostrophyResult, setup_cyclogeostrophy, assemble_result, _cyclogeostrophic_loss


def minimization_based(
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    ssh_t: Float[jax.Array, "y x"] = None,
    ug_t: Float[jax.Array, "y x"] = None,
    vg_t: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    return_geos: bool = False,
    return_losses: bool = False,
    n_it: int = 2000,
    optim: optax.GradientTransformation | str = "sgd",
    optim_kwargs: dict = None
) -> CyclogeostrophyResult:
    """
    Computes the cyclogeostrophic Sea Surface Current (SSC) velocity field
    using our minimization-based method.

    There are two modes of operation:

    1. **SSH mode**: Provide ``lat_t``, ``lon_t``, ``ssh_t`` (and optionally ``mask``).
       Geostrophic velocities will be computed from SSH.

    2. **Geostrophic mode**: Provide ``lat_t``, ``lon_t``, ``ug_t``, ``vg_t``
       (and optionally ``mask``). Geostrophic velocities are provided on the T grid
       and will be interpolated to U/V grids internally.

    Parameters
    ----------
    lat_t : Float[jax.Array, "y x"]
        Latitude of the T grid.
    lon_t : Float[jax.Array, "y x"]
        Longitude of the T grid.
    ssh_t : Float[jax.Array, "y x"], optional
        SSH field (on the T grid). Required if geostrophic velocities are not provided.
    ug_t : Float[jax.Array, "y x"], optional
        U component of geostrophic velocity on T grid. If provided with ``vg_t``,
        bypasses SSH-based computation. Will be interpolated to U grid.
    vg_t : Float[jax.Array, "y x"], optional
        V component of geostrophic velocity on T grid. If provided with ``ug_t``,
        bypasses SSH-based computation. Will be interpolated to V grid.
    land_mask : Float[jax.Array, "y x"], optional
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
        - ``ucg``: $u$ component of cyclogeostrophic velocity, on the T grid
        - ``vcg``: $v$ component of cyclogeostrophic velocity, on the T grid
        - ``ug``, ``vg``: Geostrophic velocities (if ``return_geos=True``)
        - ``losses``: Cyclogeostrophic imbalance per iteration (if ``return_losses=True``)
    """
    setup = setup_cyclogeostrophy(
        lat_t, lon_t, ssh_t=ssh_t, ug_t=ug_t, vg_t=vg_t, land_mask=land_mask
    )

    if isinstance(optim, str):
        if optim_kwargs is None:
            optim_kwargs = {"learning_rate": 0.005}
        optim = getattr(optax, optim)(**optim_kwargs)
    elif not isinstance(optim, optax.GradientTransformation):
        raise TypeError(
            "optim should be an optax.GradientTransformation optimizer, or a string referring to such an optimizer."
        )
    
    ucg, vcg, losses = _minimization_based(
        setup.ug_t, setup.vg_t,
        setup.dx_e_t, setup.dx_n_t, setup.dy_e_t, setup.dy_n_t, setup.J_t,
        setup.coriolis_factor_t,
        setup.land_mask, n_it, optim
    )

    return assemble_result(
        ucg, vcg, setup, return_geos=return_geos, return_losses=return_losses, losses=losses
    )


@partial(jax.jit, static_argnames=("n_it", "optim"))
def _minimization_based(
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
    optim: optax.GradientTransformation
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"], Float[jax.Array, "n_it"]]:
    def loss_fn(args):
        ucg, vcg = args
        return _cyclogeostrophic_loss(
            ug_t, vg_t, ucg, vcg, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, coriolis_factor_t, land_mask
        )
    
    def step_fn(carry, _):
        params = carry[:-1]
        opt_state = carry[-1]
    
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = tuple(map(lambda x: jnp.nan_to_num(x, copy=False, nan=0, posinf=0, neginf=0), grads))

        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params + (opt_state,), loss
    
    carry, losses = lax.scan(step_fn, (ug_t, vg_t, optim.init((ug_t, vg_t))), xs=None, length=n_it)
    
    return *carry[:-1], losses
