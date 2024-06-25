import optax

from jaxparrow.cyclogeostrophy import _iterative, _variational, LR_VAR
from jaxparrow.geostrophy import _geostrophy
from jaxparrow.tools.operators import interpolation

import gaussian_eddy as ge


class TestVelocities:
    _, _, _, dXY, coriolis_factor, ssh, u_geos, v_geos, u_cyclo, v_cyclo, mask = ge.simulate_gaussian_eddy(
        50e3, 3e3, .1, 36
    )

    def test_geostrophy(self):
        u_geos_est, v_geos_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor, self.mask)
        u_geos_est_t = interpolation(u_geos_est, self.mask, axis=1, padding="left")
        v_geos_est_t = interpolation(v_geos_est, self.mask, axis=0, padding="left")
        geos_rmse = self.compute_rmse(self.u_geos, self.v_geos, u_geos_est_t, v_geos_est_t)  # around 0.0003
        assert geos_rmse < .0005

    def test_cyclogeostrophy_penven(self):
        u_geos_est, v_geos_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor, self.mask)
        u_cyclo_est, v_cyclo_est, _ = _iterative(u_geos_est, v_geos_est,
                                                 self.dXY, self.dXY, self.dXY, self.dXY,
                                                 self.coriolis_factor, self.coriolis_factor, self.mask,
                                                 20, 0.01, False, 3, False)
        u_cyclo_est_t = interpolation(u_cyclo_est, self.mask, axis=1, padding="left")
        v_cyclo_est_t = interpolation(v_cyclo_est, self.mask, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .002
        assert cyclo_rmse < .003

    def test_cyclogeostrophy_ioannou(self):
        u_geos_est, v_geos_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor, self.mask)
        u_cyclo_est, v_cyclo_est, _ = _iterative(u_geos_est, v_geos_est,
                                                 self.dXY, self.dXY, self.dXY, self.dXY,
                                                 self.coriolis_factor, self.coriolis_factor, self.mask,
                                                 20, 0.01, True, 3, False)
        u_cyclo_est_t = interpolation(u_cyclo_est, self.mask, axis=1, padding="left")
        v_cyclo_est_t = interpolation(v_cyclo_est, self.mask, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .002
        assert cyclo_rmse < .003

    def test_cyclogeostrophy_variational(self):
        u_geos_est, v_geos_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor, self.mask)
        u_cyclo_est, v_cyclo_est, _ = _variational(u_geos_est, v_geos_est,
                                                   self.dXY, self.dXY, self.dXY, self.dXY,
                                                   self.coriolis_factor, self.coriolis_factor, self.mask,
                                                   1000, optax.sgd(learning_rate=LR_VAR))
        u_cyclo_est_t = interpolation(u_cyclo_est, self.mask, axis=1, padding="left")
        v_cyclo_est_t = interpolation(v_cyclo_est, self.mask, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .002
        assert cyclo_rmse < .003

    @staticmethod
    def compute_rmse(u, v, u_est, v_est) -> float:
        return float(ge.compute_rmse(u, u_est) + ge.compute_rmse(v, v_est)) / 2
