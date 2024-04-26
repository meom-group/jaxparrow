import optax

from jaxparrow.cyclogeostrophy import _iterative, _variational, LR_VAR
from jaxparrow.geostrophy import _geostrophy
from jaxparrow.tools.operators import interpolation
from jaxparrow.tools.sanitize import init_mask

import gaussian_eddy as ge


class TestVelocities:
    _, _, _, stencil_weights, coriolis_factor, ssh, u_geos, v_geos, u_cyclo, v_cyclo = ge.simulate_gaussian_eddy(
        50e3, 3e3, .1, 36
    )

    def test_geostrophy(self):
        u_geos_est, v_geos_est = _geostrophy(self.ssh, self.stencil_weights, self.coriolis_factor)
        u_geos_est_t = interpolation(u_geos_est, axis=1, pad_left=True)
        v_geos_est_t = interpolation(v_geos_est, axis=0, pad_left=True)
        geos_rmse = self.compute_rmse(self.u_geos, self.v_geos, u_geos_est_t, v_geos_est_t)  # around 0.0008
        assert geos_rmse < .001

    def test_cyclogeostrophy_penven(self):
        mask = init_mask(self.u_geos)
        u_geos_u = interpolation(self.u_geos, axis=1, pad_left=False)
        v_geos_v = interpolation(self.v_geos, axis=0, pad_left=False)
        u_cyclo_est, v_cyclo_est, _ = _iterative(u_geos_u, v_geos_v,
                                                 self.stencil_weights, self.stencil_weights,
                                                 self.coriolis_factor, self.coriolis_factor, mask,
                                                 20, 0.01, False, 3, False)
        u_cyclo_est_t = interpolation(u_cyclo_est, axis=1, pad_left=True)
        v_cyclo_est_t = interpolation(v_cyclo_est, axis=0, pad_left=True)
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .0035
        assert cyclo_rmse < .004

    def test_cyclogeostrophy_ioannou(self):
        mask = init_mask(self.u_geos)
        u_geos_u = interpolation(self.u_geos, axis=1, pad_left=False)
        v_geos_v = interpolation(self.v_geos, axis=0, pad_left=False)
        u_cyclo_est, v_cyclo_est, _ = _iterative(u_geos_u, v_geos_v,
                                                 self.stencil_weights, self.stencil_weights,
                                                 self.coriolis_factor, self.coriolis_factor, mask,
                                                 20, 0.01, True, 3, False)
        u_cyclo_est_t = interpolation(u_cyclo_est, axis=1, pad_left=True)
        v_cyclo_est_t = interpolation(v_cyclo_est, axis=0, pad_left=True)
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .0035
        assert cyclo_rmse < .004

    def test_cyclogeostrophy_variational(self):
        mask = init_mask(self.u_geos)
        u_geos_u = interpolation(self.u_geos, axis=1, pad_left=False)
        v_geos_v = interpolation(self.v_geos, axis=0, pad_left=False)
        optim = optax.sgd(learning_rate=LR_VAR)
        u_cyclo_est, v_cyclo_est, _ = _variational(u_geos_u, v_geos_v,
                                                   self.stencil_weights, self.stencil_weights,
                                                   self.coriolis_factor, self.coriolis_factor, mask,
                                                   20, optim, False)
        u_cyclo_est_t = interpolation(u_cyclo_est, axis=1, pad_left=True)
        v_cyclo_est_t = interpolation(v_cyclo_est, axis=0, pad_left=True)
        cyclo_rmse = self.compute_rmse(self.u_cyclo, self.v_cyclo, u_cyclo_est_t, v_cyclo_est_t)  # around .0035
        assert cyclo_rmse < .004

    @staticmethod
    def compute_rmse(u, v, u_est, v_est) -> float:
        return float(ge.compute_rmse(u, u_est) + ge.compute_rmse(v, v_est)) / 2
