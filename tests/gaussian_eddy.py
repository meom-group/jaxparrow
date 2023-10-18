import numpy as np

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
    u_geos, v_geos = simulate_gaussian_geos(r0, X, Y, ssh, coriolis_factor)
    u_cyclo, v_cyclo = simulate_gaussian_cyclo(r0, u_geos, v_geos, coriolis_factor)

    return X, Y, R, dXY, coriolis_factor, ssh, u_geos, v_geos, u_cyclo, v_cyclo


def simulate_gaussian_ssh(r0, eta0, R):
    return eta0 * np.exp(-(R / r0)**2)


def simulate_gaussian_geos(r0, X, Y, ssh, coriolis_factor):
    u_geos = 2 * geom.GRAVITY * Y * ssh / (coriolis_factor * r0**2)
    v_geos = - 2 * geom.GRAVITY * X * ssh / (coriolis_factor * r0**2)
    return u_geos, v_geos


def simulate_gaussian_cyclo(r0, u_geos, v_geos, coriolis_factor):
    azim_geos = compute_azimuthal_magnitude(u_geos, v_geos)
    azim_cyclo = 2 * azim_geos / (1 + np.sqrt(1 + 4 * azim_geos / (coriolis_factor * r0)))
    u_cyclo = u_geos * r0 * coriolis_factor / (r0 * coriolis_factor - azim_cyclo**2)
    v_cyclo = v_geos * r0 * coriolis_factor / (r0 * coriolis_factor - azim_cyclo**2)
    return np.nan_to_num(u_cyclo, nan=0, posinf=0, neginf=0), np.nan_to_num(v_cyclo, nan=0, posinf=0, neginf=0)


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


def compute_rmse(vel, vel_est):
    return np.sqrt(np.mean((vel - vel_est)**2))


def compute_mape(vel, vel_est):
    ape = np.abs((vel - vel_est) / vel)
    return np.mean(np.nan_to_num(ape, nan=0, posinf=0, neginf=0))


def compute_cor(vel, vel_est):
    return np.corrcoef(vel.flatten(), vel_est.flatten())[0, 1]
