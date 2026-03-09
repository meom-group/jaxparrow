import jax.numpy as jnp
import optax

from jaxparrow.cyclogeostrophy._fixed_point import _fixed_point
from jaxparrow.cyclogeostrophy._minimization_based import _minimization_based
from jaxparrow.geostrophy import _geostrophy
from jaxparrow.utils.operators import interpolation

import gaussian_eddy as ge


class TestVelocities:
    _, _, _, dXY, coriolis_factor, ssh, ug, vg, ucg, vcg, mask = ge.simulate_gaussian_eddy(
        50e3, 5e3, -.2, 36
    )
    grid_angle = jnp.zeros_like(ssh)

    def test_geostrophy(self):
        ug_est, vg_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor, self.grid_angle, self.mask)
        ug_est_t = interpolation(ug_est, self.mask, axis=1, padding="left")
        vg_est_t = interpolation(vg_est, self.mask, axis=0, padding="left")
        geos_rmse = self.compute_rmse(self.ug, self.vg, ug_est_t, vg_est_t)  # around 0.002
        assert geos_rmse < .003

    def test_cyclogeostrophy_penven(self):
        ug_est, vg_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor, self.grid_angle, self.mask)
        ucg_est, vcg_est, _ = _fixed_point(ug_est, vg_est,
                                                   self.dXY, self.dXY, self.dXY, self.dXY,
                                                   self.coriolis_factor, self.coriolis_factor, 
                                                   self.grid_angle, self.grid_angle, 
                                                   self.mask,
                                                   20, 0.01, False)
        ucg_est_t = interpolation(ucg_est, self.mask, axis=1, padding="left")
        vcg_est_t = interpolation(vcg_est, self.mask, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.ucg, self.vcg, ucg_est_t, vcg_est_t)  # around .002
        assert cyclo_rmse < .003

    def test_cyclogeostrophy_ioannou(self):
        ug_est, vg_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor, self.grid_angle, self.mask)
        ucg_est, vcg_est, _ = _fixed_point(ug_est, vg_est,
                                                   self.dXY, self.dXY, self.dXY, self.dXY,
                                                   self.coriolis_factor, self.coriolis_factor, 
                                                   self.grid_angle, self.grid_angle, 
                                                   self.mask,
                                                   20, 0.01, False)
        ucg_est_t = interpolation(ucg_est, self.mask, axis=1, padding="left")
        vcg_est_t = interpolation(vcg_est, self.mask, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.ucg, self.vcg, ucg_est_t, vcg_est_t)  # around .002
        assert cyclo_rmse < .003

    def test_cyclogeostrophy_minimization(self):
        ug_est, vg_est = _geostrophy(self.ssh, self.dXY, self.dXY, self.coriolis_factor, self.grid_angle, self.mask)
        ucg_est, vcg_est, _ = _minimization_based(ug_est, vg_est,
                                                          self.dXY, self.dXY, self.dXY, self.dXY,
                                                          self.coriolis_factor, self.coriolis_factor, 
                                                          self.grid_angle, self.grid_angle, 
                                                          self.mask,
                                                          1000, optax.sgd(learning_rate=0.005))
        ucg_est_t = interpolation(ucg_est, self.mask, axis=1, padding="left")
        vcg_est_t = interpolation(vcg_est, self.mask, axis=0, padding="left")
        cyclo_rmse = self.compute_rmse(self.ucg, self.vcg, ucg_est_t, vcg_est_t)  # around .002
        assert cyclo_rmse < .003

    @staticmethod
    def compute_rmse(u, v, u_est, v_est) -> float:
        return float(ge.compute_rmse(u, u_est) + ge.compute_rmse(v, v_est)) / 2
