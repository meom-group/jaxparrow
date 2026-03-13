from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float

from ..geostrophy import geostrophy
from ..utils import geometry, operators, sanitize


# =============================================================================
# Types
# =============================================================================

class CyclogeostrophySetup(NamedTuple):
    """
    Contains precomputed values needed for cyclogeostrophic calculations.

    Attributes
    ----------
    land_mask : Float[jax.Array, "y x"]
        Land mask where `True`/`1` indicates land
    ug_t : Float[jax.Array, "y x"]
        $u$ component of geostrophic velocity, on the T grid
    vg_t : Float[jax.Array, "y x"]
        $v$ component of geostrophic velocity, on the T grid
    dx_e_t : Float[jax.Array, "y x"]
        Eastward displacement associated with one step in the x-index direction, on the T grid
    dx_n_t : Float[jax.Array, "y x"]
        Northward displacement associated with one step in the x-index direction, on the T grid
    dy_e_t : Float[jax.Array, "y x"]
        Eastward displacement associated with one step in the y-index direction, on the T grid
    dy_n_t : Float[jax.Array, "y x"]
        Northward displacement associated with one step in the y-index direction, on the T grid
    J_t : Float[jax.Array, "y x"]
        Jacobian of the transformation from grid to geographic coordinates, on the T grid
    coriolis_factor_t : Float[jax.Array, "y x"]
        Coriolis factor, on the T grid
    """

    land_mask: Float[jax.Array, "y x"]
    ug_t: Float[jax.Array, "y x"]
    vg_t: Float[jax.Array, "y x"]
    dx_e_t: Float[jax.Array, "y x"]
    dx_n_t: Float[jax.Array, "y x"]
    dy_e_t: Float[jax.Array, "y x"]
    dy_n_t: Float[jax.Array, "y x"]
    J_t: Float[jax.Array, "y x"]
    coriolis_factor_t: Float[jax.Array, "y x"]


class CyclogeostrophyResult(NamedTuple):
    """
    Result of cyclogeostrophic velocity computation.

    This NamedTuple provides named access to results, avoiding positional unpacking errors.
    All fields except ``ucg`` and ``vcg`` are optional and depend on the
    ``return_*`` flags passed to the computation function.

    Attributes
    ----------
    ucg : Float[jax.Array, "y x"]
        $u$ component of cyclogeostrophic velocity, on the T grid
    vcg : Float[jax.Array, "y x"]
        $v$ component of cyclogeostrophic velocity, on the T grid
    ug : Float[jax.Array, "y x"] | None
        $u$ component of geostrophic velocity, on the T grid (if ``return_geos=True``)
    vg : Float[jax.Array, "y x"] | None
        $v$ component of geostrophic velocity, on the T grid (if ``return_geos=True``)
    losses : Float[jax.Array, "n_it"] | None
        Cyclogeostrophic imbalance over iterations (if ``return_losses=True``)
    """

    ucg: Float[jax.Array, "y x"]
    vcg: Float[jax.Array, "y x"]
    ug: Float[jax.Array, "y x"] | None = None
    vg: Float[jax.Array, "y x"] | None = None
    losses: Float[jax.Array, "n_it"] | None = None


# =============================================================================
# Setup and Result Assembly
# =============================================================================

def setup_cyclogeostrophy(
    lat_t: Float[jax.Array, "y x"],
    lon_t: Float[jax.Array, "y x"],
    ssh_t: Float[jax.Array, "y x"] = None,
    ug_t: Float[jax.Array, "y x"] = None,
    vg_t: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None
) -> CyclogeostrophySetup:
    """
    Computes all preliminary values needed for cyclogeostrophic calculations.

    This includes: land mask, geostrophic velocities, U/V grids, spatial steps,
    Coriolis factors, and grid rotation angles on all grids.

    There are two modes of operation:

    1. **SSH mode**: Provide ``lat_t``, ``lon_t``, ``ssh_t`` (and optionally ``mask``).
       Geostrophic velocities will be computed from SSH

    2. **Geostrophic mode**: Provide ``lat_t``, ``lon_t``, ``ug_t``, ``vg_t``
       (and optionally ``land_mask``). Geostrophic velocities are provided on the T grid

    Parameters
    ----------
    lat_t : Float[jax.Array, "y x"]
        Latitudes of T grid.
    lon_t : Float[jax.Array, "y x"]
        Longitudes of T grid.
    ssh_t : Float[jax.Array, "y x"], optional
        SSH field on T grid. Required if geostrophic velocities are not provided.
    ug_t : Float[jax.Array, "y x"], optional
        U component of geostrophic velocity on T grid. If provided with ``vg_t``,
        bypasses SSH-based computation. Will be interpolated to U grid.
    vg_t : Float[jax.Array, "y x"], optional
        V component of geostrophic velocity on T grid. If provided with ``ug_t``,
        bypasses SSH-based computation. Will be interpolated to V grid.
    land_mask : Float[jax.Array, "y x"], optional
        Land mask where `1`/`True` is land. If None, inferred from ssh_t or ug_t nan values.

    Returns
    -------
    CyclogeostrophySetup
        Named tuple containing all precomputed values

    Raises
    ------
    ValueError
        If neither SSH nor geostrophic velocity inputs are provided.
    """
    # Check if geostrophic velocities are provided directly
    use_geos_directly = ug_t is not None and vg_t is not None

    if use_geos_directly:
        land_mask = sanitize.init_land_mask(ug_t, land_mask)
    else:
        # SSH-based computation
        if ssh_t is None:
            raise ValueError(
                "Either provide ssh_t to compute geostrophic velocities from SSH, "
                "or provide ug_t, vg_t directly on the T grid."
            )
        
        land_mask = sanitize.init_land_mask(ssh_t, land_mask)

        ug_t, vg_t = geostrophy(ssh_t, lat_t, lon_t, land_mask)

    dx_e, dx_n, dy_e, dy_n, J = geometry.grid_metrics(lat_t, lon_t)

    f = geometry.coriolis_factor(lat_t)

    return CyclogeostrophySetup(
        land_mask=land_mask,
        ug_t=ug_t,
        vg_t=vg_t,
        dx_e_t=dx_e,
        dx_n_t=dx_n,
        dy_e_t=dy_e,
        dy_n_t=dy_n,
        J_t=J,
        coriolis_factor_t=f,
    )


def assemble_result(
    ucg_t: Float[jax.Array, "y x"],
    vcg_t: Float[jax.Array, "y x"],
    setup: CyclogeostrophySetup,
    return_geos: bool = False,
    return_losses: bool = False,
    losses: Float[jax.Array, "n_it"] = None,
) -> CyclogeostrophyResult:
    """
    Assembles the final result, sanitizing outputs and including optional fields.

    Parameters
    ----------
    ucg_t : Float[jax.Array, "y x"]
        $u$ component of cyclogeostrophic velocity
    vcg_t : Float[jax.Array, "y x"]
        $v$ component of cyclogeostrophic velocity
    setup : CyclogeostrophySetup
        Precomputed setup values
    return_geos : bool, optional
        Include geostrophic velocities in result
    return_grids : bool, optional
        Include U/V grid coordinates in result
    return_losses : bool, optional
        Include losses in result
    losses : Float[jax.Array, "n_it"], optional
        Loss values from iterative methods

    Returns
    -------
    CyclogeostrophyResult
        Named tuple with computed velocities and optional fields
    """
    # Handle masked data (set land cells to NaN)
    ucg_t = sanitize.sanitize_data(ucg_t, jnp.nan, setup.land_mask)
    vcg_t = sanitize.sanitize_data(vcg_t, jnp.nan, setup.land_mask)

    return CyclogeostrophyResult(
        ucg=ucg_t,
        vcg=vcg_t,
        ug=setup.ug_t if return_geos else None,
        vg=setup.vg_t if return_geos else None,
        losses=losses if return_losses else None,
    )


# =============================================================================
# Public API Functions
# =============================================================================

def cyclogeostrophic_loss(
    ug: Float[jax.Array, "y x"],
    vg: Float[jax.Array, "y x"],
    ucg: Float[jax.Array, "y x"],
    vcg: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    lat_u: Float[jax.Array, "y x"] = None,
    lon_u: Float[jax.Array, "y x"] = None,
    lat_v: Float[jax.Array, "y x"] = None,
    lon_v: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True
) -> Float[jax.Array, ""]:
    """
    Computes the cyclogeostrophic imbalance loss from a geostrophic and a cyclogeostrophic velocity field.

    The velocity field can be provided either on the T grid (``uv_on_t=True``) or on the U/V grids (``uv_on_t=False``).

    If provided, the ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are expected to follow the NEMO convention.

    Parameters
    ----------
    ug : Float[jax.Array, "y x"]
        $u$ component of the geostrophic velocity field
    vg : Float[jax.Array, "y x"]
        $v$ component of the geostrophic velocity field
    ucg : Float[jax.Array, "y x"]
        $u$ component of the cyclogeostrophic velocity field
    vcg : Float[jax.Array, "y x"]
        $v$ component of the cyclogeostrophic velocity field
    lat_t : Float[jax.Array, "y x"], optional
        Latitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lon_t : Float[jax.Array, "y x"], optional
        Longitudes of the T grid.
       
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lat_u : Float[jax.Array, "y x"], optional
        Latitudes of the U grid.
        
        Defaults to `None`
    lon_u : Float[jax.Array, "y x"], optional
        Longitudes of the U grid.
        
        Defaults to `None`
    lat_v : Float[jax.Array, "y x"], optional
        Latitudes of the V grid.
        
        Defaults to `None`
    lon_v : Float[jax.Array, "y x"], optional
        Longitudes of the V grid.
        
        Defaults to `None`
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
    uv_on_t : bool, optional
        If `True`, the velocity components are assumed to be located on the T grid 
        (this is important when manipulating staggered grids)
        
        Defaults to `True`
    Returns
    -------
    loss : Float[jax.Array, ""]
        Cyclogeostrophic imbalance loss
    """
    u_imbalance, v_imbalance = cyclogeostrophic_imbalance(
        ug, vg, ucg, vcg, lat_t, lon_t, lat_u, lon_u, lat_v, lon_v, land_mask, uv_on_t
    )

    return jnp.nansum(u_imbalance ** 2 + v_imbalance ** 2)


def cyclogeostrophic_imbalance(
    ug: Float[jax.Array, "y x"],
    vg: Float[jax.Array, "y x"],
    ucg: Float[jax.Array, "y x"],
    vcg: Float[jax.Array, "y x"],
    lat_t: Float[jax.Array, "y x"] = None,
    lon_t: Float[jax.Array, "y x"] = None,
    lat_u: Float[jax.Array, "y x"] = None,
    lon_u: Float[jax.Array, "y x"] = None,
    lat_v: Float[jax.Array, "y x"] = None,
    lon_v: Float[jax.Array, "y x"] = None,
    land_mask: Float[jax.Array, "y x"] = None,
    uv_on_t: bool = True,
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    """
    Computes the cyclogeostrophic imbalance of a 2d velocity field.

    The velocity fields can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    ug : Float[jax.Array, "y x"]
        $u$ component of the geostrophic velocity field
    vg : Float[jax.Array, "y x"]
        $v$ component of the geostrophic velocity field
    ucg : Float[jax.Array, "y x"]
        $u$ component of the cyclogeostrophic velocity field
    vcg : Float[jax.Array, "y x"]
        $v$ component of the cyclogeostrophic velocity field
    lat_t : Float[jax.Array, "y x"], optional
        Latitudes of the T grid.
        
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lon_t : Float[jax.Array, "y x"], optional
        Longitudes of the T grid.
       
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        
        Defaults to `None`
    lat_u : Float[jax.Array, "y x"], optional
        Latitudes of the U grid.
        
        Defaults to `None`
    lon_u : Float[jax.Array, "y x"], optional
        Longitudes of the U grid.
        
        Defaults to `None`
    lat_v : Float[jax.Array, "y x"], optional
        Latitudes of the V grid.
        
        Defaults to `None`
    lon_v : Float[jax.Array, "y x"], optional
        Longitudes of the V grid.
        
        Defaults to `None`
    land_mask : Float[jax.Array, "y x"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
    uv_on_t : bool, optional
        If `True`, the velocity components are assumed to be located on the T grid 
        (this is important when manipulating staggered grids)
        
        Defaults to `True`

    Returns
    -------
    u_imbalance : Float[jax.Array, "y x"]
        $u$ component of the cyclogeostrophic imbalance, on the T grid
    v_imbalance : Float[jax.Array, "y x"]
        $v$ component of the cyclogeostrophic imbalance, on the T grid
    """
    if land_mask is None:
        land_mask = sanitize.init_land_mask(ug)

    if not uv_on_t:
        ug = operators.interpolation(ug, axis=1, padding="left", land_mask=land_mask)  # U(i), U(i+1) -> T(i+1)
        vg = operators.interpolation(vg, axis=0, padding="left", land_mask=land_mask)  # U(i), U(i+1) -> T(i+1)
        ucg = operators.interpolation(ucg, axis=1, padding="right", land_mask=land_mask)
        vcg = operators.interpolation(vcg, axis=0, padding="right", land_mask=land_mask)

    if lat_t is None or lon_t is None:
        if lat_u is not None and lon_u is not None:
            lat_t = operators.interpolation(lat_u, axis=1, padding="left", land_mask=land_mask)
            lon_t = operators.interpolation(lon_u, axis=1, padding="left", land_mask=land_mask)
        elif lat_v is not None and lon_v is not None:
            lat_t = operators.interpolation(lat_v, axis=0, padding="left", land_mask=land_mask)
            lon_t = operators.interpolation(lon_v, axis=0, padding="left", land_mask=land_mask)
        else:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
    
    # compute grid metrics once
    dx_e, dx_n, dy_e, dy_n, J = geometry.grid_metrics(lat_t, lon_t)
    f = geometry.coriolis_factor(lat_t)

    return _cyclogeostrophic_imbalance(
        ug, vg, ucg, vcg, dx_e, dx_n, dy_e, dy_n, J, f, land_mask
    )


# =============================================================================
# Internal Functions
# =============================================================================

def _cyclogeostrophic_loss(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    ucg_t: Float[jax.Array, "y x"],
    vcg_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> Float[jax.Array, ""]:
    u_imbalance, v_imbalance = _cyclogeostrophic_imbalance(
        ug_t, vg_t, ucg_t, vcg_t, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, coriolis_factor_t, land_mask
    )

    return jnp.nansum(u_imbalance ** 2 + v_imbalance ** 2)


def _cyclogeostrophic_imbalance(
    ug_t: Float[jax.Array, "y x"],
    vg_t: Float[jax.Array, "y x"],
    ucg_t: Float[jax.Array, "y x"],
    vcg_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    coriolis_factor_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    u_adv_t, v_adv_t = _advection(ucg_t, vcg_t, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, land_mask)

    u_imbalance = ucg_t + v_adv_t / coriolis_factor_t - ug_t
    v_imbalance = vcg_t - u_adv_t / coriolis_factor_t - vg_t

    return u_imbalance, v_imbalance


def _advection(
    u_t: Float[jax.Array, "y x"],
    v_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> tuple[Float[jax.Array, "y x"], Float[jax.Array, "y x"]]:
    u_adv = _u_advection(u_t, v_t, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, land_mask)
    v_adv = _v_advection(u_t, v_t, dx_e_t, dx_n_t, dy_e_t, dy_n_t, J_t, land_mask)

    return u_adv, v_adv


def _u_advection(
    u_t: Float[jax.Array, "y x"],
    v_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> Float[jax.Array, "y x"]:
    """
    Computes u * ∂u/∂x + v * ∂u/∂y
    """
    du_e_t, du_n_t = operators.horizontal_derivatives(
        u_t, dx_e=dx_e_t, dx_n=dx_n_t, dy_e=dy_e_t, dy_n=dy_n_t, J=J_t, land_mask=land_mask
    )

    u_adv = u_t * du_e_t + v_t * du_n_t

    return u_adv


def _v_advection(
    u_t: Float[jax.Array, "y x"],
    v_t: Float[jax.Array, "y x"],
    dx_e_t: Float[jax.Array, "y x"],
    dx_n_t: Float[jax.Array, "y x"],
    dy_e_t: Float[jax.Array, "y x"],
    dy_n_t: Float[jax.Array, "y x"],
    J_t: Float[jax.Array, "y x"],
    land_mask: Float[jax.Array, "y x"],
) -> Float[jax.Array, "y x"]:
    """
    Computes u * ∂v/∂x + v * ∂v/∂y
    """
    dv_e_t, dv_n_t = operators.horizontal_derivatives(
        v_t, dx_e=dx_e_t, dx_n=dx_n_t, dy_e=dy_e_t, dy_n=dy_n_t, J=J_t, land_mask=land_mask
    )

    v_adv = u_t * dv_e_t + v_t * dv_n_t

    return v_adv
