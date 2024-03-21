from jaxparrow.cyclogeostrophy import _iterative, _variational
from jaxparrow.geostrophy import _geostrophy
from jaxparrow.tools.operators import interpolation
from jaxparrow.tools.sanitize import init_mask

import gaussian_eddy as ge


class TestVelocities:
    _, _, _, dXY, coriolis_factor, ssh, u_geos, v_geos, u_cyclo, v_cyclo = ge.simulate_gaussian_eddy(50e3, 3e3, .1, 36)

    def test_geostrophy(self):
        u_geos_est, v_geos_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor)
        u_geos_est_t = interpolation(u_geos_est, axis=1, padding="left")
        v_geos_est_t = interpolation(v_geos_est, axis=0, padding="left")
        geos_rmse = self.compute_rmse(self.u_geos, self.v_geos, u_geos_est_t, v_geos_est_t)  # around 0.0008
        assert geos_rmse < .001

    def test_cyclogeostrophy_penven(self):
        mask = init_mask(self.u_geos)
        u_geos_u = interpolation(self.u_geos, axis=1, padding="right")
        v_geos_v = interpolation(self.v_geos, axis=0, padding="right")
        u_cyclo_est, v_cyclo_est, _ = _iterative(u_geos_u, v_geos_v,
                                                 self.dXY, self.dXY, self.dXY, self.dXY,
                                                 self.coriolis_factor, self.coriolis_factor, mask,
                                                 20, 0.01, "same", False, 3, False)
        u_cyclo_est_t = interpolation(u_cyclo_est, axis=1, padding="left")
        v_cyclo_est_t = interpolation(v_cyclo_est, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .0035
        assert cyclo_rmse < .004

    def test_cyclogeostrophy_ioannou(self):
        mask = init_mask(self.u_geos)
        u_geos_u = interpolation(self.u_geos, axis=1, padding="right")
        v_geos_v = interpolation(self.v_geos, axis=0, padding="right")
        u_cyclo_est, v_cyclo_est, _ = _iterative(u_geos_u, v_geos_v,
                                                 self.dXY, self.dXY, self.dXY, self.dXY,
                                                 self.coriolis_factor, self.coriolis_factor, mask,
                                                 20, 0.01, "same", True, 3, False)
        u_cyclo_est_t = interpolation(u_cyclo_est, axis=1, padding="left")
        v_cyclo_est_t = interpolation(v_cyclo_est, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .0035
        assert cyclo_rmse < .004

    def test_cyclogeostrophy_variational(self):
        mask = init_mask(self.u_geos)
        u_geos_u = interpolation(self.u_geos, axis=1, padding="right")
        v_geos_v = interpolation(self.v_geos, axis=0, padding="right")
        u_cyclo_est, v_cyclo_est, _ = _variational(u_geos_u, v_geos_v,
                                                   self.dXY, self.dXY, self.dXY, self.dXY,
                                                   self.coriolis_factor, self.coriolis_factor, mask,
                                                   20, "sgd", None, False)
        u_cyclo_est_t = interpolation(u_cyclo_est, axis=1, padding="left")
        v_cyclo_est_t = interpolation(v_cyclo_est, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .0035
        assert cyclo_rmse < .004

    @staticmethod
    def compute_rmse(u, v, u_est, v_est) -> float:
        return float(ge.compute_rmse(u, u_est) + ge.compute_rmse(v, v_est)) / 2
