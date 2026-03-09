"""Core utilities, types, and internal functions for cyclogeostrophy computations."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float

from ..geostrophy import geostrophy
from ..utils import geometry, kinematics, operators, sanitize


# =============================================================================
# Types
# =============================================================================

class CyclogeostrophySetup(NamedTuple):
    """
    Contains precomputed values needed for cyclogeostrophic calculations.

    Attributes
    ----------
    is_land : Float[jax.Array, "lat lon"]
        Land mask where `True`/`1` indicates land
    ug_u : Float[jax.Array, "lat lon"]
        U component of geostrophic velocity on U grid
    vg_v : Float[jax.Array, "lat lon"]
        V component of geostrophic velocity on V grid
    lat_u : Float[jax.Array, "lat lon"]
        Latitudes of the U grid
    lon_u : Float[jax.Array, "lat lon"]
        Longitudes of the U grid
    lat_v : Float[jax.Array, "lat lon"]
        Latitudes of the V grid
    lon_v : Float[jax.Array, "lat lon"]
        Longitudes of the V grid
    dx_u : Float[jax.Array, "lat lon"]
        Spatial steps in meters along x on U grid
    dy_u : Float[jax.Array, "lat lon"]
        Spatial steps in meters along y on U grid
    dx_v : Float[jax.Array, "lat lon"]
        Spatial steps in meters along x on V grid
    dy_v : Float[jax.Array, "lat lon"]
        Spatial steps in meters along y on V grid
    coriolis_factor_u : Float[jax.Array, "lat lon"]
        Coriolis factor on U grid
    coriolis_factor_v : Float[jax.Array, "lat lon"]
        Coriolis factor on V grid
    coriolis_factor_t : Float[jax.Array, "lat lon"]
        Coriolis factor on T grid
    grid_angle_u : Float[jax.Array, "lat lon"]
        Grid rotation angle on U grid (radians, counterclockwise from east)
    grid_angle_v : Float[jax.Array, "lat lon"]
        Grid rotation angle on V grid (radians, counterclockwise from east)
    grid_angle_t : Float[jax.Array, "lat lon"]
        Grid rotation angle on T grid (radians, counterclockwise from east)
    """

    is_land: Float[jax.Array, "lat lon"]
    ug_u: Float[jax.Array, "lat lon"]
    vg_v: Float[jax.Array, "lat lon"]
    lat_u: Float[jax.Array, "lat lon"]
    lon_u: Float[jax.Array, "lat lon"]
    lat_v: Float[jax.Array, "lat lon"]
    lon_v: Float[jax.Array, "lat lon"]
    dx_u: Float[jax.Array, "lat lon"]
    dy_u: Float[jax.Array, "lat lon"]
    dx_v: Float[jax.Array, "lat lon"]
    dy_v: Float[jax.Array, "lat lon"]
    coriolis_factor_u: Float[jax.Array, "lat lon"]
    coriolis_factor_v: Float[jax.Array, "lat lon"]
    coriolis_factor_t: Float[jax.Array, "lat lon"]
    grid_angle_u: Float[jax.Array, "lat lon"]
    grid_angle_v: Float[jax.Array, "lat lon"]
    grid_angle_t: Float[jax.Array, "lat lon"]


class CyclogeostrophyResult(NamedTuple):
    """
    Result of cyclogeostrophic velocity computation.

    This NamedTuple provides named access to results, avoiding positional unpacking errors.
    All fields except ``ucg`` and ``vcg`` are optional and depend on the
    ``return_*`` flags passed to the computation function.

    Attributes
    ----------
    ucg : Float[jax.Array, "lat lon"]
        U component of cyclogeostrophic velocity on U grid
    vcg : Float[jax.Array, "lat lon"]
        V component of cyclogeostrophic velocity on V grid
    ug : Float[jax.Array, "lat lon"] | None
        U component of geostrophic velocity on U grid (if ``return_geos=True``)
    vg : Float[jax.Array, "lat lon"] | None
        V component of geostrophic velocity on V grid (if ``return_geos=True``)
    lat_u : Float[jax.Array, "lat lon"] | None
        Latitudes of U grid (if ``return_grids=True``)
    lon_u : Float[jax.Array, "lat lon"] | None
        Longitudes of U grid (if ``return_grids=True``)
    lat_v : Float[jax.Array, "lat lon"] | None
        Latitudes of V grid (if ``return_grids=True``)
    lon_v : Float[jax.Array, "lat lon"] | None
        Longitudes of V grid (if ``return_grids=True``)
    losses : Float[jax.Array, "n_it"] | None
        Cyclogeostrophic imbalance over iterations (if ``return_losses=True``)
    """

    ucg: Float[jax.Array, "lat lon"]
    vcg: Float[jax.Array, "lat lon"]
    ug: Float[jax.Array, "lat lon"] | None = None
    vg: Float[jax.Array, "lat lon"] | None = None
    lat_u: Float[jax.Array, "lat lon"] | None = None
    lon_u: Float[jax.Array, "lat lon"] | None = None
    lat_v: Float[jax.Array, "lat lon"] | None = None
    lon_v: Float[jax.Array, "lat lon"] | None = None
    losses: Float[jax.Array, "n_it"] | None = None


# =============================================================================
# Setup and Result Assembly
# =============================================================================

def setup_cyclogeostrophy(
    ssh_t: Float[jax.Array, "lat lon"],
    lat_t: Float[jax.Array, "lat lon"],
    lon_t: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"] = None
) -> CyclogeostrophySetup:
    """
    Computes all preliminary values needed for cyclogeostrophic calculations.

    This includes: land mask, geostrophic velocities, U/V grids, spatial steps,
    Coriolis factors, and grid rotation angles on all grids.

    Parameters
    ----------
    ssh_t : Float[jax.Array, "lat lon"]
        SSH field on T grid
    lat_t : Float[jax.Array, "lat lon"]
        Latitudes of T grid
    lon_t : Float[jax.Array, "lat lon"]
        Longitudes of T grid
    mask : Float[jax.Array, "lat lon"], optional
        Land mask where `1`/`True` is land. If None, inferred from ssh_t nan values.

    Returns
    -------
    CyclogeostrophySetup
        Named tuple containing all precomputed values
    """
    is_land = sanitize.init_land_mask(ssh_t, mask)

    geos_results = geostrophy(
        ssh_t, lat_t, lon_t, is_land, return_grids=True
    )

    ug_u = geos_results.ug
    vg_v = geos_results.vg
    lat_u = geos_results.lat_u
    lon_u = geos_results.lon_u
    lat_v = geos_results.lat_v
    lon_v = geos_results.lon_v

    dx_u, dy_u = geometry.compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = geometry.compute_spatial_step(lat_v, lon_v)

    coriolis_factor_u = geometry.compute_coriolis_factor(lat_u)
    coriolis_factor_v = geometry.compute_coriolis_factor(lat_v)
    coriolis_factor_t = geometry.compute_coriolis_factor(lat_t)

    # Compute grid rotation angles for curvilinear grid support
    grid_angle_u = geometry.compute_grid_angle(lat_u, lon_u)
    grid_angle_v = geometry.compute_grid_angle(lat_v, lon_v)
    grid_angle_t = geometry.compute_grid_angle(lat_t, lon_t)

    return CyclogeostrophySetup(
        is_land=is_land,
        ug_u=ug_u,
        vg_v=vg_v,
        lat_u=lat_u,
        lon_u=lon_u,
        lat_v=lat_v,
        lon_v=lon_v,
        dx_u=dx_u,
        dy_u=dy_u,
        dx_v=dx_v,
        dy_v=dy_v,
        coriolis_factor_u=coriolis_factor_u,
        coriolis_factor_v=coriolis_factor_v,
        coriolis_factor_t=coriolis_factor_t,
        grid_angle_u=grid_angle_u,
        grid_angle_v=grid_angle_v,
        grid_angle_t=grid_angle_t,
    )


def assemble_result(
    ucg_u: Float[jax.Array, "lat lon"],
    vcg_v: Float[jax.Array, "lat lon"],
    setup: CyclogeostrophySetup,
    return_geos: bool = False,
    return_grids: bool = True,
    return_losses: bool = False,
    losses: Float[jax.Array, "n_it"] = None,
) -> CyclogeostrophyResult:
    """
    Assembles the final result, sanitizing outputs and including optional fields.

    Parameters
    ----------
    ucg_u : Float[jax.Array, "lat lon"]
        U component of cyclogeostrophic velocity
    vcg_v : Float[jax.Array, "lat lon"]
        V component of cyclogeostrophic velocity
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
    # Extrapolate velocities to fill NaN cells within the valid ocean domain
    # This recovers edge cells that became NaN due to derivative computation
    ucg_u = operators.extrapolate_to_valid(ucg_u, setup.is_land)
    vcg_v = operators.extrapolate_to_valid(vcg_v, setup.is_land)

    # Handle masked data (set land cells to NaN)
    ucg_u = sanitize.sanitize_data(ucg_u, jnp.nan, setup.is_land)
    vcg_v = sanitize.sanitize_data(vcg_v, jnp.nan, setup.is_land)

    return CyclogeostrophyResult(
        ucg=ucg_u,
        vcg=vcg_v,
        ug=setup.ug_u if return_geos else None,
        vg=setup.vg_v if return_geos else None,
        lat_u=setup.lat_u if return_grids else None,
        lon_u=setup.lon_u if return_grids else None,
        lat_v=setup.lat_v if return_grids else None,
        lon_v=setup.lon_v if return_grids else None,
        losses=losses if return_losses else None,
    )


# =============================================================================
# Public API Functions
# =============================================================================

def cyclogeostrophic_loss(
    ug: Float[jax.Array, "lat lon"],
    vg: Float[jax.Array, "lat lon"],
    ucg: Float[jax.Array, "lat lon"],
    vcg: Float[jax.Array, "lat lon"],
    lat_t: Float[jax.Array, "lat lon"] = None,
    lon_t: Float[jax.Array, "lat lon"] = None,
    lat_u: Float[jax.Array, "lat lon"] = None,
    lon_u: Float[jax.Array, "lat lon"] = None,
    lat_v: Float[jax.Array, "lat lon"] = None,
    lon_v: Float[jax.Array, "lat lon"] = None,
    mask: Float[jax.Array, "lat lon"] = None,
    vel_on_uv: bool = True
) -> Float[jax.Array, ""]:
    """
    Computes the cyclogeostrophic imbalance loss from a geostrophic SSC velocity field and a cyclogeostrophic SSC velocity field.

    The velocity fields can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    ug : Float[jax.Array, "lat lon"]
        U component of the geostrophic SSC velocity field
    vg : Float[jax.Array, "lat lon"]
        V component of the geostrophic SSC velocity field
    ucg : Float[jax.Array, "lat lon"]
        U component of the cyclogeostrophic SSC velocity field
    vcg : Float[jax.Array, "lat lon"]
        V component of the cyclogeostrophic SSC velocity field
    lat_t : Float[jax.Array, "lat lon"], optional
        Latitudes of the T grid.
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        Defaults to `None`
    lon_t : Float[jax.Array, "lat lon"], optional
        Longitudes of the T grid.
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        Defaults to `None`
    lat_u : Float[jax.Array, "lat lon"], optional
        Latitudes of the U grid.
        Defaults to `None`
    lon_u : Float[jax.Array, "lat lon"], optional
        Longitudes of the U grid.
        Defaults to `None`
    lat_v : Float[jax.Array, "lat lon"], optional
        Latitudes of the V grid.
        Defaults to `None`
    lon_v : Float[jax.Array, "lat lon"], optional
        Longitudes of the V grid.
        Defaults to `None`
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).
        If not provided, inferred from ``ug`` `nan` values.
        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, ``ucg`` and ``vcg`` are on the U and V grids.
        If `False`, they are on the T grid.
        Defaults to `True`

    Returns
    -------
    loss : Float[jax.Array, ""]
        Cyclogeostrophic imbalance loss
    """
    if mask is None:
        mask = sanitize.init_land_mask(ug)

    if not vel_on_uv:
        ug = operators.interpolation(ug, mask, axis=1, padding="right")
        vg = operators.interpolation(vg, mask, axis=0, padding="right")
        ucg = operators.interpolation(ucg, mask, axis=1, padding="right")
        vcg = operators.interpolation(vcg, mask, axis=0, padding="right")

    if lat_u is None or lon_u is None or lat_v is None or lon_v is None:
        if lat_t is None or lon_t is None:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
        lat_u, lon_u, lat_v, lon_v = geometry.compute_uv_grids(lat_t, lon_t)

    dx_u, dy_u = geometry.compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = geometry.compute_spatial_step(lat_v, lon_v)
    coriolis_factor_u = geometry.compute_coriolis_factor(lat_u)
    coriolis_factor_v = geometry.compute_coriolis_factor(lat_v)

    return _cyclogeostrophic_loss(
        ug, vg, ucg, vcg,
        dx_u, dx_v, dy_u, dy_v,
        coriolis_factor_u, coriolis_factor_v,
        mask
    )


def cyclogeostrophic_imbalance(
    ug: Float[jax.Array, "lat lon"],
    vg: Float[jax.Array, "lat lon"],
    ucg: Float[jax.Array, "lat lon"],
    vcg: Float[jax.Array, "lat lon"],
    lat_t: Float[jax.Array, "lat lon"] = None,
    lon_t: Float[jax.Array, "lat lon"] = None,
    lat_u: Float[jax.Array, "lat lon"] = None,
    lon_u: Float[jax.Array, "lat lon"] = None,
    lat_v: Float[jax.Array, "lat lon"] = None,
    lon_v: Float[jax.Array, "lat lon"] = None,
    mask: Float[jax.Array, "lat lon"] = None,
    vel_on_uv: bool = True
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]]:
    """
    Computes the cyclogeostrophic imbalance of a 2d velocity field, on a C-grid (following NEMO convention).

    The velocity fields can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    ug : Float[jax.Array, "lat lon"]
        U component of the geostrophic velocity field
    vg : Float[jax.Array, "lat lon"]
        V component of the geostrophic velocity field
    ucg : Float[jax.Array, "lat lon"]
        U component of the cyclogeostrophic velocity field
    vcg : Float[jax.Array, "lat lon"]
        V component of the cyclogeostrophic velocity field
    lat_t : Float[jax.Array, "lat lon"], optional
        Latitudes of the T grid.
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        Defaults to `None`
    lon_t : Float[jax.Array, "lat lon"], optional
        Longitudes of the T grid.
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        Defaults to `None`
    lat_u : Float[jax.Array, "lat lon"], optional
        Latitudes of the U grid.
        Defaults to `None`
    lon_u : Float[jax.Array, "lat lon"], optional
        Longitudes of the U grid.
        Defaults to `None`
    lat_v : Float[jax.Array, "lat lon"], optional
        Latitudes of the V grid.
        Defaults to `None`
    lon_v : Float[jax.Array, "lat lon"], optional
        Longitudes of the V grid.
        Defaults to `None`
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).
        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, the velocity components are assumed to be located on the U and V grids.
        If `False`, they are on the T grid.
        Defaults to `True`

    Returns
    -------
    u_imbalance_u : Float[jax.Array, "lat lon"]
        U component of the cyclogeostrophic imbalance, on the U grid
    v_imbalance_v : Float[jax.Array, "lat lon"]
        V component of the cyclogeostrophic imbalance, on the V grid
    """
    if not vel_on_uv:
        ug_u = operators.interpolation(ug, mask, axis=1, padding="right")
        vg_v = operators.interpolation(vg, mask, axis=0, padding="right")
        ucg_u = operators.interpolation(ucg, mask, axis=1, padding="right")
        vcg_v = operators.interpolation(vcg, mask, axis=0, padding="right")
    else:
        ug_u = ug
        vg_v = vg
        ucg_u = ucg
        vcg_v = vcg

    if lat_u is None or lon_u is None or lat_v is None or lon_v is None:
        if lat_t is None or lon_t is None:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
        lat_u, lon_u, lat_v, lon_v = geometry.compute_uv_grids(lat_t, lon_t)

    dx_u, dy_u = geometry.compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = geometry.compute_spatial_step(lat_v, lon_v)
    coriolis_factor_u = geometry.compute_coriolis_factor(lat_u)
    coriolis_factor_v = geometry.compute_coriolis_factor(lat_v)

    return _cyclogeostrophic_imbalance(
        ug_u, vg_v, ucg_u, vcg_v,
        dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v, mask
    )


def radius_of_curvature(
    u: Float[jax.Array, "lat lon"],
    v: Float[jax.Array, "lat lon"],
    lat_t: Float[jax.Array, "lat lon"] = None,
    lon_t: Float[jax.Array, "lat lon"] = None,
    lat_u: Float[jax.Array, "lat lon"] = None,
    lon_u: Float[jax.Array, "lat lon"] = None,
    lat_v: Float[jax.Array, "lat lon"] = None,
    lon_v: Float[jax.Array, "lat lon"] = None,
    mask: Float[jax.Array, "lat lon"] = None,
    vel_on_uv: bool = True
) -> Float[jax.Array, "lat lon"]:
    """
    Computes the radius of curvature of a 2d velocity field, on a C-grid (following NEMO convention).

    The velocity field can be provided either on the U and V grids (``vel_on_uv=True``) or on the T grid (``vel_on_uv=False``).

    Parameters
    ----------
    u : Float[jax.Array, "lat lon"]
        U component of the velocity field
    v : Float[jax.Array, "lat lon"]
        V component of the velocity field
    lat_t : Float[jax.Array, "lat lon"], optional
        Latitudes of the T grid.
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        Defaults to `None`
    lon_t : Float[jax.Array, "lat lon"], optional
        Longitudes of the T grid.
        If ``lat_u``, ``lon_u``, ``lat_v``, and ``lon_v`` are not provided, ``lat_t`` and ``lon_t`` must be provided to compute them.
        Defaults to `None`
    lat_u : Float[jax.Array, "lat lon"], optional
        Latitudes of the U grid.
        Defaults to `None`
    lon_u : Float[jax.Array, "lat lon"], optional
        Longitudes of the U grid.
        Defaults to `None`
    lat_v : Float[jax.Array, "lat lon"], optional
        Latitudes of the V grid.
        Defaults to `None`
    lon_v : Float[jax.Array, "lat lon"], optional
        Longitudes of the V grid.
        Defaults to `None`
    mask : Float[jax.Array, "lat lon"], optional
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land).
        Defaults to `None`
    vel_on_uv : bool, optional
        If `True`, ``u`` and ``v`` are on the U and V grids.
        If `False`, they are on the T grid.
        Defaults to `True`

    Returns
    -------
    r : Float[jax.Array, "lat lon"]
        The radius of curvature of the velocity field
    """
    if lat_u is None or lon_u is None or lat_v is None or lon_v is None:
        if lat_t is None or lon_t is None:
            raise ValueError("Either lat_t and lon_t, or lat_u, lon_u, lat_v, and lon_v must be provided")
        lat_u, lon_u, lat_v, lon_v = geometry.compute_uv_grids(lat_t, lon_t)

    dx_u, dy_u = geometry.compute_spatial_step(lat_u, lon_u)
    dx_v, dy_v = geometry.compute_spatial_step(lat_v, lon_v)

    return _radius_of_curvature(u, v, dx_u, dx_v, dy_u, dy_v, mask, vel_on_uv)


# =============================================================================
# Internal Functions
# =============================================================================

def _cyclogeostrophic_loss(
    ug_u: Float[jax.Array, "lat lon"],
    vg_v: Float[jax.Array, "lat lon"],
    ucg_u: Float[jax.Array, "lat lon"],
    vcg_v: Float[jax.Array, "lat lon"],
    dx_u: Float[jax.Array, "lat lon"],
    dx_v: Float[jax.Array, "lat lon"],
    dy_u: Float[jax.Array, "lat lon"],
    dy_v: Float[jax.Array, "lat lon"],
    coriolis_factor_u: Float[jax.Array, "lat lon"],
    coriolis_factor_v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    grid_angle_u: Float[jax.Array, "lat lon"] = None,
    grid_angle_v: Float[jax.Array, "lat lon"] = None
) -> Float[jax.Array, ""]:
    u_imbalance, v_imbalance = _cyclogeostrophic_imbalance(
        ug_u, vg_v, ucg_u, vcg_v,
        dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
        mask, grid_angle_u, grid_angle_v
    )

    return jnp.nansum(u_imbalance ** 2 + v_imbalance ** 2)


def _cyclogeostrophic_imbalance(
    ug_u: Float[jax.Array, "lat lon"],
    vg_v: Float[jax.Array, "lat lon"],
    ucg_u: Float[jax.Array, "lat lon"],
    vcg_v: Float[jax.Array, "lat lon"],
    dx_u: Float[jax.Array, "lat lon"],
    dx_v: Float[jax.Array, "lat lon"],
    dy_u: Float[jax.Array, "lat lon"],
    dy_v: Float[jax.Array, "lat lon"],
    coriolis_factor_u: Float[jax.Array, "lat lon"],
    coriolis_factor_v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    grid_angle_u: Float[jax.Array, "lat lon"] = None,
    grid_angle_v: Float[jax.Array, "lat lon"] = None
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]]:
    u_adv_v, v_adv_u = _advection(ucg_u, vcg_v, dx_u, dx_v, dy_u, dy_v, mask, grid_angle_u, grid_angle_v)

    u_imbalance_u = ucg_u + v_adv_u / coriolis_factor_u - ug_u
    v_imbalance_v = vcg_v - u_adv_v / coriolis_factor_v - vg_v

    return u_imbalance_u, v_imbalance_v


def _advection(
    u_u: Float[jax.Array, "lat lon"],
    v_v: Float[jax.Array, "lat lon"],
    dx_u: Float[jax.Array, "lat lon"],
    dx_v: Float[jax.Array, "lat lon"],
    dy_u: Float[jax.Array, "lat lon"],
    dy_v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    grid_angle_u: Float[jax.Array, "lat lon"] = None,
    grid_angle_v: Float[jax.Array, "lat lon"] = None
) -> tuple[Float[jax.Array, "lat lon"], Float[jax.Array, "lat lon"]]:
    """
    Computes the advection terms of a 2d velocity field, on a C-grid, following NEMO convention.

    For curvilinear grids, gradients are rotated from grid coordinates to geographic coordinates
    before computing the advection terms.

    Parameters
    ----------
    u_u : Float[jax.Array, "lat lon"]
        U component of the velocity field (on the U grid)
    v_v : Float[jax.Array, "lat lon"]
        V component of the SSC velocity field (on the V grid)
    dx_u : Float[jax.Array, "lat lon"]
        Spatial steps on the U grid along `x`, in meters
    dy_u : Float[jax.Array, "lat lon"]
        Spatial steps on the U grid along `y`, in meters
    dx_v : Float[jax.Array, "lat lon"]
        Spatial steps on the V grid along `x`, in meters
    dy_v : Float[jax.Array, "lat lon"]
        Spatial steps on the V grid along `y`, in meters
    mask : Float[jax.Array, "lat lon"]
        Mask defining the marine area of the spatial domain; `1` or `True` stands for masked (i.e. land)
    grid_angle_u : Float[jax.Array, "lat lon"], optional
        Grid rotation angle on U grid (radians). If None, assumes rectilinear grid (no rotation).
    grid_angle_v : Float[jax.Array, "lat lon"], optional
        Grid rotation angle on V grid (radians). If None, assumes rectilinear grid (no rotation).

    Returns
    -------
    u_adv_v : Float[jax.Array, "lat lon"]
        U component of the advection term, on the V grid
    v_adv_u : Float[jax.Array, "lat lon"]
        V component of the advection term, on the U grid
    """
    u_adv_v = _u_advection_v(u_u, v_v, dx_v, dy_v, mask, grid_angle_v)
    v_adv_u = _v_advection_u(u_u, v_v, dx_u, dy_u, mask, grid_angle_u)

    return u_adv_v, v_adv_u


def _u_advection_v(
    u_u: Float[jax.Array, "lat lon"],
    v_v: Float[jax.Array, "lat lon"],
    dx_u: Float[jax.Array, "lat lon"],
    dy_u: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    grid_angle_v: Float[jax.Array, "lat lon"] = None
) -> Float[jax.Array, "lat lon"]:
    """
    Computes u * ∂u/∂x + v * ∂u/∂y at V points in geographic coordinates.
    """
    # Grid-coordinate gradients
    dudi_t = operators.derivative(u_u, dx_u, mask, axis=1, padding="left")   # (U(i), U(i+1)) -> T(i+1)
    dudi_v = operators.interpolation(dudi_t, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)

    dudj_f = operators.derivative(u_u, dy_u, mask, axis=0, padding="right")  # (U(j), U(j+1)) -> F(j)
    dudj_v = operators.interpolation(dudj_f, mask, axis=1, padding="left")   # (F(i), F(i+1)) -> V(i+1)

    # Rotate to geographic coordinates if grid angle provided
    if grid_angle_v is not None:
        dudx_v, dudy_v = operators.rotate_to_geographic(dudi_v, dudj_v, grid_angle_v)
    else:
        dudx_v, dudy_v = dudi_v, dudj_v

    u_t = operators.interpolation(u_u, mask, axis=1, padding="left")   # (U(i), U(i+1)) -> T(i+1)
    u_v = operators.interpolation(u_t, mask, axis=0, padding="right")  # (T(j), T(j+1)) -> V(j)

    u_adv_v = u_v * dudx_v + v_v * dudy_v  # V(j)

    return u_adv_v


def _v_advection_u(
    u_u: Float[jax.Array, "lat lon"],
    v_v: Float[jax.Array, "lat lon"],
    dx_v: Float[jax.Array, "lat lon"],
    dy_v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    grid_angle_u: Float[jax.Array, "lat lon"] = None
) -> Float[jax.Array, "lat lon"]:
    """
    Computes u * ∂v/∂x + v * ∂v/∂y at U points in geographic coordinates.
    """
    # Grid-coordinate gradients
    dvdi_f = operators.derivative(v_v, dx_v, mask, axis=1, padding="right")  # (V(i), V(i+1)) -> F(i)
    dvdi_u = operators.interpolation(dvdi_f, mask, axis=0, padding="left")   # (F(j), F(j+1)) -> U(j+1)

    dvdj_t = operators.derivative(v_v, dy_v, mask, axis=0, padding="left")   # (V(j), V(j+1)) -> T(j+1)
    dvdj_u = operators.interpolation(dvdj_t, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)

    # Rotate to geographic coordinates if grid angle provided
    if grid_angle_u is not None:
        dvdx_u, dvdy_u = operators.rotate_to_geographic(dvdi_u, dvdj_u, grid_angle_u)
    else:
        dvdx_u, dvdy_u = dvdi_u, dvdj_u

    v_t = operators.interpolation(v_v, mask, axis=0, padding="left")   # (V(j), V(j+1)) -> T(j+1)
    v_u = operators.interpolation(v_t, mask, axis=1, padding="right")  # (T(i), T(i+1)) -> U(i)

    v_adv_u = u_u * dvdx_u + v_u * dvdy_u  # U(i)

    return v_adv_u


def _radius_of_curvature(
    u: Float[jax.Array, "lat lon"],
    v: Float[jax.Array, "lat lon"],
    dx_u: Float[jax.Array, "lat lon"],
    dx_v: Float[jax.Array, "lat lon"],
    dy_u: Float[jax.Array, "lat lon"],
    dy_v: Float[jax.Array, "lat lon"],
    mask: Float[jax.Array, "lat lon"],
    vel_on_uv: bool,
    grid_angle_t: Float[jax.Array, "lat lon"] = None
) -> Float[jax.Array, "lat lon"]:
    if not vel_on_uv:
        u_t = u
        v_t = v
        u_u = operators.interpolation(u, mask, axis=1, padding="right")
        v_v = operators.interpolation(v, mask, axis=0, padding="right")
    else:
        u_t = operators.interpolation(u, mask, axis=1, padding="left")
        v_t = operators.interpolation(v, mask, axis=0, padding="left")
        u_u = u
        v_v = v

    V_t = kinematics.magnitude(u_t, v_t, vel_on_uv=False)

    # Derivatives along grid axes
    du_di_t = operators.derivative(u_u, dx_u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)
    du_dj_f = operators.derivative(u_u, dy_u, mask, axis=0, padding="right")  # (U(j), U(j+1)) -> F(j)

    dv_di_f = operators.derivative(v_v, dx_v, mask, axis=1, padding="right")  # (V(i), V(i+1)) -> F(i)
    dv_dj_t = operators.derivative(v_v, dy_v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)

    # Interpolate to T grid
    du_dj_v = operators.interpolation(du_dj_f, mask, axis=1, padding="left")  # (F(i), F(i+1)) -> V(i+1)
    du_dj_t = operators.interpolation(du_dj_v, mask, axis=0, padding="left")  # (V(j), V(j+1)) -> T(j+1)
    dv_di_u = operators.interpolation(dv_di_f, mask, axis=0, padding="left")  # (F(j), F(j+1)) -> U(j+1)
    dv_di_t = operators.interpolation(dv_di_u, mask, axis=1, padding="left")  # (U(i), U(i+1)) -> T(i+1)

    # Rotate to geographic coordinates if grid angle is provided
    if grid_angle_t is not None:
        du_dx_t, du_dy_t = operators.rotate_to_geographic(du_di_t, du_dj_t, grid_angle_t)
        dv_dx_t, dv_dy_t = operators.rotate_to_geographic(dv_di_t, dv_dj_t, grid_angle_t)
    else:
        du_dx_t = du_di_t
        du_dy_t = du_dj_t
        dv_dx_t = dv_di_t
        dv_dy_t = dv_dj_t

    numerator = V_t ** 3
    denominator = u_t ** 2 * dv_dx_t - v_t ** 2 * du_dy_t - u_t * v_t * (du_dx_t - dv_dy_t)
    r = numerator / denominator

    return r
