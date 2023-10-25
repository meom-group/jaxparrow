import numpy as np

from jaxparrow.tools import tools


def simulate_gaussian_eddy(r0: float, dxy: float, eta0: float, latitude: int) \
        -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            np.ndarray]:
    l0 = r0 * 1.5
    xy = np.arange(0, l0, dxy)
    xy = np.concatenate((-xy[::-1][:-1], xy))
    X, Y = np.meshgrid(xy, xy)
    R = np.hypot(X, Y)
    dXY = np.ones_like(X) * dxy
    coriolis_factor = np.ones_like(R) * tools.compute_coriolis_factor(latitude)

    ssh = simulate_gaussian_ssh(r0, eta0, R)
    u_geos, v_geos = simulate_gaussian_geos(r0, X, Y, ssh, coriolis_factor)
    u_cyclo, v_cyclo = simulate_gaussian_cyclo(r0, np.arctan2(Y, X), u_geos, v_geos, coriolis_factor)

    return X, Y, R, dXY, coriolis_factor, ssh, u_geos, v_geos, u_cyclo, v_cyclo


def simulate_gaussian_ssh(r0: float, eta0: float, R: np.ndarray) -> np.ndarray:
    return eta0 * np.exp(-(R / r0)**2)


def simulate_gaussian_geos(r0: float, X: np.ndarray, Y: np.ndarray, ssh: np.ndarray, coriolis_factor: np.ndarray) \
        -> [np.ndarray, np.ndarray]:
    def f():
        return 2 * tools.GRAVITY * ssh / (coriolis_factor * r0 ** 2)
    u_geos = Y * f()
    v_geos = - X * f()
    return u_geos, v_geos


def simulate_gaussian_cyclo(r0: float, theta: np.ndarray, u_geos: np.ndarray, v_geos: np.ndarray,
                            coriolis_factor: np.ndarray) -> [np.ndarray, np.ndarray]:
    def f():
        return azim_cyclo**2 / (r0 * coriolis_factor)
    azim_geos = compute_azimuthal_magnitude(u_geos, v_geos)
    azim_cyclo = 2 * azim_geos / (1 + np.sqrt(1 + 4 * azim_geos / (coriolis_factor * r0)))
    u_cyclo = u_geos + np.sin(theta) * f()
    v_cyclo = v_geos - np.cos(theta) * f()
    return u_cyclo, v_cyclo


def reinterpolate(f: np.ndarray, axis: int) -> np.ndarray:
    f = np.copy(f)
    if axis == 0:
        f[1:, :] = f[:-1, :]
        f[0, :] = - f[-1, :]  # axis-symmetric
    elif axis == 1:
        f[:, 1:] = f[:, :-1]
        f[:, 0] = - f[:, -1]  # axis-symmetric
    return f


def compute_azimuthal_magnitude(u_component: np.ndarray, v_component: np.ndarray) -> np.ndarray:
    return np.sqrt(u_component**2 + v_component**2)


def compute_rmse(vel: np.ndarray, vel_est: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((vel - vel_est)**2))


def compute_mape(vel: np.ndarray, vel_est: np.ndarray) -> np.ndarray:
    ape = np.abs((vel - vel_est) / vel)
    return np.mean(np.nan_to_num(ape, nan=0, posinf=0, neginf=0))


def compute_cor(vel: np.ndarray, vel_est: np.ndarray) -> np.ndarray:
    return np.corrcoef(vel.flatten(), vel_est.flatten())[0, 1]
