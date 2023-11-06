import numpy as np

from jaxparrow import cyclogeostrophy
from jaxparrow import geostrophy

import gaussian_eddy as ge


class TestVelocities:
    _, _, _, dXY, coriolis_factor, ssh, u_geos, v_geos, u_cyclo, v_cyclo = ge.simulate_gaussian_eddy(50e3, 3e3, .1, 36)

    def test_geostrophy(self):
        u_geos_est, v_geos_est = geostrophy(self.ssh, self.dXY, self.dXY,
                                            self.coriolis_factor, self.coriolis_factor)
        u_geos_est, v_geos_est = ge.reinterpolate(u_geos_est, axis=0), ge.reinterpolate(v_geos_est, axis=1)
        geos_rmse = self.compute_rmse(self.u_geos, self.v_geos, u_geos_est, v_geos_est)  # around .003
        assert geos_rmse <= .01

    def test_cyclogeostrophy_penven(self):
        u_cyclo_est, v_cyclo_est = cyclogeostrophy(self.u_geos, self.v_geos, self.dXY, self.dXY,
                                                   self.dXY, self.dXY, self.coriolis_factor, self.coriolis_factor,
                                                   method="iterative")
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est, v_cyclo_est)  # around .003
        assert cyclo_rmse <= .01

    def test_cyclogeostrophy_ioannou(self):
        u_cyclo_est, v_cyclo_est = cyclogeostrophy(self.u_geos, self.v_geos, self.dXY, self.dXY,
                                                   self.dXY, self.dXY, self.coriolis_factor, self.coriolis_factor,
                                                   method="iterative", use_res_filter=True)
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est, v_cyclo_est)  # around .003
        assert cyclo_rmse <= .01

    def test_cyclogeostrophy_variational(self):
        u_cyclo_est, v_cyclo_est = cyclogeostrophy(self.u_geos, self.v_geos, self.dXY, self.dXY,
                                                   self.dXY, self.dXY, self.coriolis_factor, self.coriolis_factor,
                                                   method="variational")
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est, v_cyclo_est)  # around .003
        assert cyclo_rmse <= .01

    @staticmethod
    def compute_rmse(u: np.ndarray, v: np.ndarray, u_est: np.ndarray, v_est: np.ndarray) -> float:
        return float(ge.compute_rmse(u, u_est) + ge.compute_rmse(v, v_est)) / 2
