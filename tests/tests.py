import jaxparrow.cyclogeostrophy as cyclo
import jaxparrow.geostrophy as geos

import gaussian_eddy as ge


def test_velocities():
    _, __, ___, dXY, coriolis_factor, ssh, u_geos, v_geos, u_cyclo, v_cyclo = (
        ge.simulate_gaussian_eddy(50e3, 3e3, .1, 36))
    u_geos_cyclo, v_geos_cyclo, _ = test_geostrophy(ssh, dXY, dXY, coriolis_factor, coriolis_factor, u_geos, v_geos)
    test_cyclogeostrophy_penven(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, u_cyclo, v_cyclo)
    test_cyclogeostrophy_ioannou(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, u_cyclo, v_cyclo)
    test_cyclogeostrophy_var(u_geos, v_geos, dXY, dXY, dXY, dXY, coriolis_factor, coriolis_factor, u_cyclo, v_cyclo)


def test_geostrophy(ssh, dx, dy, coriolis_factor_u, coriolis_factor_v, u_geos, v_geos):
    u_geos_est, v_geos_est = geos.geostrophy(ssh, dx, dy, coriolis_factor_u, coriolis_factor_v)
    u_geos_est, v_geos_est = ge.reinterpolate(u_geos_est, axis=0), ge.reinterpolate(v_geos_est, axis=1)
    geos_rmse = ge.compute_rmse(u_geos, u_geos_est) + ge.compute_rmse(v_geos, v_geos_est)
    return u_geos_est, v_geos_est, geos_rmse


def test_cyclogeostrophy_penven(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                u_cyclo, v_cyclo):
    u_cyclo_est, v_cyclo_est = cyclo.iterative(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v,
                                               coriolis_factor_u, coriolis_factor_v)
    cyclo_rmse = ge.compute_rmse(u_cyclo, u_cyclo_est) + ge.compute_rmse(v_cyclo, v_cyclo_est)
    return u_cyclo_est, v_cyclo_est, cyclo_rmse


def test_cyclogeostrophy_ioannou(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                                 u_cyclo, v_cyclo):
    u_cyclo_est, v_cyclo_est = cyclo.iterative(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v,
                                               coriolis_factor_u, coriolis_factor_v)
    cyclo_rmse = ge.compute_rmse(u_cyclo, u_cyclo_est) + ge.compute_rmse(v_cyclo, v_cyclo_est)
    return u_cyclo_est, v_cyclo_est, cyclo_rmse


def test_cyclogeostrophy_var(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v, coriolis_factor_u, coriolis_factor_v,
                             u_cyclo, v_cyclo):
    u_cyclo_est, v_cyclo_est = cyclo.variational(u_geos, v_geos, dx_u, dx_v, dy_u, dy_v,
                                                 coriolis_factor_u, coriolis_factor_v)
    cyclo_rmse = ge.compute_rmse(u_cyclo, u_cyclo_est) + ge.compute_rmse(v_cyclo, v_cyclo_est)
    return u_cyclo_est, v_cyclo_est, cyclo_rmse
