import numpy as np

import jaxparrow.cyclogeostrophy as cyclo
import jaxparrow.geostrophy as geos
import jaxparrow.tools.geometry as geom


def simulate_gaussian_eddy(r0, dxy, eta0, latitude):
    l0 = r0 * 1.5
    xy = np.arange(0, l0, dxy)
    xy = np.concatenate((-xy[::-1][:-1], xy))
    X, Y = np.meshgrid(xy, xy)
    R = np.hypot(X, Y)
    dXY = np.ones_like(X) * dxy
    coriolis_factor = np.ones_like(R) * geom.compute_coriolis_factor(latitude)

    ssh = simulate_gaussian_ssh(r0, eta0, R)
    azim_geos = simulate_gaussian_azim_geos(r0, ssh, R, coriolis_factor)
    azim_cyclo = simulate_gaussian_azim_cyclo(azim_geos, R, coriolis_factor)

    return X, Y, R, dXY, coriolis_factor, ssh, azim_geos, azim_cyclo


def simulate_gaussian_ssh(r0, eta0, R):
    return eta0 * np.exp(-(R / r0)**2)


def simulate_gaussian_azim_geos(r0, ssh, R, coriolis_factor):
    return - 2 * ssh * geos.GRAVITY / (coriolis_factor * r0) * R / r0


def simulate_gaussian_azim_cyclo(azim_geos, R, coriolis_factor):
    azim_cyclo = 2 * azim_geos / (1 + np.sqrt(1 + 4 * azim_geos / (coriolis_factor * R)))
    return np.nan_to_num(azim_cyclo)


def reinterpolate(f, axis):
    f = np.copy(f)
    if axis == 0:
        f[1:, :] = f[:-1, :]
        f[0, :] = - f[-1, :]  # axis-symmetric
    elif axis == 1:
        f[:, 1:] = f[:, :-1]
        f[:, 0] = - f[:, -1]  # axis-symmetric
    return f


def compute_azimuthal_magnitude(u_component, v_component):
    return np.sqrt(u_component**2 + v_component**2)


def compute_rmse(azim, azim_est):
    return np.sqrt(np.mean((azim - azim_est)**2))


def compute_mape(azim, azim_est):
    ape = np.abs((azim - azim_est) / azim)
    return np.mean(np.nan_to_num(ape, nan=0, posinf=0, neginf=0))


def compute_cor(azim, azim_est):
    return np.corrcoef(azim.flatten(), azim_est.flatten())[0, 1]


def test_velocities():
    _, __, ___, dXY, coriolis_factor, ssh, azim_geos, azim_cyclo = simulate_gaussian_eddy(50e3, 3e3, .1, 36)
    u_geos, v_geos, _ = test_geostrophy(ssh, dXY, dXY, coriolis_factor, coriolis_factor, azim_geos)
    test_cyclogeostrophy_penven(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, azim_cyclo)
    test_cyclogeostrophy_ioannou(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, azim_cyclo)
    test_cyclogeostrophy_var(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, azim_cyclo)


def test_geostrophy(ssh, dx, dy, coriolis_factor_u, coriolis_factor_v, azim_geos):
    u_geos, v_geos = geos.geostrophy(ssh, dx, dy, coriolis_factor_u, coriolis_factor_v)
    azim_geos_est = compute_azimuthal_magnitude(reinterpolate(u_geos, axis=0), reinterpolate(v_geos, axis=1))
    geos_rmse = compute_rmse(azim_geos, azim_geos_est)
    return u_geos, v_geos, geos_rmse


def test_cyclogeostrophy_penven(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                azim_cyclo):
    u_cyclo, v_cyclo = cyclo.iterative(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v)
    azim_cyclo_est = compute_azimuthal_magnitude(u_cyclo, v_cyclo)
    cyclo_rmse = compute_rmse(azim_cyclo, azim_cyclo_est)
    return cyclo_rmse


def test_cyclogeostrophy_ioannou(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                azim_cyclo):
    u_cyclo, v_cyclo = cyclo.iterative(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v)
    azim_cyclo_est = compute_azimuthal_magnitude(u_cyclo, v_cyclo)
    cyclo_rmse = compute_rmse(azim_cyclo, azim_cyclo_est)
    return cyclo_rmse


def test_cyclogeostrophy_var(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                             azim_cyclo):
    u_cyclo, v_cyclo = cyclo.variational(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v)
    azim_cyclo_est = compute_azimuthal_magnitude(u_cyclo, v_cyclo)
    cyclo_rmse = compute_rmse(azim_cyclo, azim_cyclo_est)
    return cyclo_rmse
