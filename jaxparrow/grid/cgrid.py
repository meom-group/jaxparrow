from typing import Tuple, Union

import numpy as np

from jaxparrow.grid import variable
from jaxparrow import geostrophy as geos, cyclogeostrophy as cyclo


class CGrid:
    """Class representing the C-grid over which flows will be computed"""

    def __init__(self, ssh_lat: np.ndarray, ssh_lon: np.ndarray,
                 u_lat: np.ndarray, u_lon: np.ndarray,
                 v_lat: np.ndarray, v_lon: np.ndarray,
                 ssh_value: np.ndarray,
                 u_value: np.ndarray = None, v_value: np.ndarray = None,
                 ssh_mask: np.ndarray = None, u_mask: np.ndarray = None, v_mask: np.ndarray = None,
                 u_geos_value: np.ndarray = None, v_geos_value: np.ndarray = None):
        """Constructor

        :param ssh_lat: the Sea Surface Height (SSH) latitude grid
        :type ssh_lat: np.ndarray
        :param ssh_lon: the SSH longitude grid
        :type ssh_lon: np.ndarray
        :param u_lat: the U reference velocity latitude grid
        :type u_lat: np.ndarray
        :param u_lon: the U reference velocity longitude grid
        :type u_lon: np.ndarray
        :param v_lat: the V reference velocity latitude grid
        :type v_lat: np.ndarray
        :param v_lon: the V reference velocity longitude grid
        :type v_lon: np.ndarray
        :param ssh_value: the SSH value grid
        :type ssh_value: np.ndarray
        :param u_value: the U reference velocity value grid, defaults to None
        :type u_value: np.ndarray, optional
        :param v_value: the V reference velocity value grid, defaults to None
        :type v_value: np.ndarray, optional
        :param ssh_mask: the SSH mask grid, defaults to None
        :type ssh_mask: np.ndarray, optional
        :param u_mask: the U velocity mask grid, defaults to None
        :type u_mask: np.ndarray, optional
        :param v_mask: the V velocity mask grid, defaults to None
        :type v_mask: np.ndarray, optional
        :param u_geos_value: the U geostrophic velocity value grid, defaults to None
        :type u_geos_value: np.ndarray, optional
        :param v_geos_value: the V geostrophic velocity value grid, defaults to None
        :type v_geos_value: np.ndarray, optional
        """

        #: Sea Surface Height (SSH) field
        self.__ssh = variable.Variable(ssh_lat, ssh_lon, ssh_value, ssh_mask)
        #: U reference velocity field
        self.__u = variable.Variable(u_lat, u_lon, u_value, u_mask)
        #: V reference velocity field
        self.__v = variable.Variable(v_lat, v_lon, v_value, v_mask)

        #: U geostrophic velocity field
        self.__u_geos = variable.Variable(u_lat, u_lon, u_geos_value, u_mask)
        #: V geostrophic velocity field
        self.__v_geos = variable.Variable(v_lat, v_lon, v_geos_value, v_mask)

        #: U cyclogeostrophic velocity field
        self.__u_cyclo = variable.Variable(u_lat, u_lon, None, u_mask)
        #: V cyclogeostrophic velocity field
        self.__v_cyclo = variable.Variable(v_lat, v_lon, None, v_mask)

    # Getter methods

    def get_ssh(self) -> variable.Variable:
        """Getter for SSH field

        :returns: SSH field
        :rtype: variable.Variable
        """
        return self.__ssh
    
    def get_u(self) -> variable.Variable:
        """Getter for U reference velocity field

        :returns: U reference velocity field
        :rtype: variable.Variable
        """
        return self.__u

    def get_v(self) -> variable.Variable:
        """Getter for V reference velocity field

        :returns: V reference velocity field
        :rtype: variable.Variable
        """
        return self.__v

    def get_u_geos(self) -> variable.Variable:
        """Getter for U geostrophic velocity field

        :returns: U geostrophic velocity field
        :rtype: variable.Variable
        """
        return self.__u_geos

    def get_v_geos(self) -> variable.Variable:
        """Getter for V geostrophic velocity field

        :returns: V geostrophic velocity field
        :rtype: variable.Variable
        """
        return self.__v_geos

    def get_u_cyclo(self) -> variable.Variable:
        """Getter for U cylogeostrophic velocity field

        :returns: U cylogeostrophic velocity field
        :rtype: variable.Variable
        """
        return self.__u_cyclo

    def get_v_cyclo(self) -> variable.Variable:
        """Getter for V cylogeostrophic velocity field

        :returns: V cylogeostrophic velocity field
        :rtype: variable.Variable
        """
        return self.__v_cyclo

    # Setter methods

    def set_ssh(self, ssh: variable.Variable):
        """Setter for SSH field

        :param ssh: SSH field to set
        :type: variable.Field
        """
        self.__ssh = ssh

    def set_u(self, u: variable.Variable):
        """Setter for U reference velocity field

        :param u: U reference velocity field to set
        :type u: variable.Variable
        """
        self.__u = u

    def set_v(self, v: variable.Variable):
        """Setter for V reference velocity field

        :param v: V reference velocity field to set
        :type v: variable.Variable
        """
        self.__v = v

    def set_u_geos(self, u_geos: variable.Variable):
        """Setter for U geostrophic velocity field

        :param u_geos: U geostrophic velocity field to set
        :type u_geos: variable.Variable
        """
        self.__u_geos = u_geos

    def set_v_geos(self, v_geos: variable.Variable):
        """Setter for V geostrophic velocity field

        :param v_geos: V geostrophic velocity field to set
        :type v_geos: variable.Variable
        """
        self.__v_geos = v_geos

    def set_u_cyclo(self, u_cyclo: variable.Variable):
        """Setter for U cyclogeostrophic velocity field

        :param: U cyclogeostrophic velocity field to set
        :type u_cyclo: variable.Variable
        """
        self.__u_cyclo = u_cyclo

    def set_v_cyclo(self, v_cyclo: variable.Variable):
        """Setter for V cyclogeostrophic velocity field

        :param v_cyclo: V cyclogeostrophic velocity field to set
        :type v_cyclo: variable.Variable
        """
        self.__v_cyclo = v_cyclo

    # Methods

    def compute_geostrophy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the geostrophic velocities from SSH values and derivatives, and U and V coriolis factor

        :returns: U and V geostrophic velocity fields
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        u_geos, v_geos = geos.geostrophy(self.__ssh.get_value(), self.__ssh.get_dx(), self.__ssh.get_dy(),
                                         self.__u_geos.get_coriolis_factor(), self.__v_geos.get_coriolis_factor())

        self.__u_geos.set_value(u_geos)
        self.__v_geos.set_value(v_geos)

        return u_geos, v_geos

    def compute_cyclogeostrophy(self, variational: bool = True, n_it: int = None, lr: float = cyclo.LR_VAR,
                                eps: float = cyclo.EPSILON_IT) \
            -> Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]:
        """Computes the cyclogeostrophic velocities using either the variational or the iterative approach

        :param variational: flag indicating which approach to use, defaults to True
        :type variational: bool, optional
        :param n_it: maximum number of iterations, defaults to None
        :type n_it: int, optional
        :param lr: learning rate for the variational method, defaults to cyclo.LR_VAR
        :type lr: float, optional
        :param eps: residual tolerance for the iterative method, defaults to cyclo.EPSILON_IT
        :type eps: float, optional

        :returns: U and V cyclogeostrophic velocity fields
        :rtype: Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]
        """
        if self.__u_geos.get_value() is None or self.__v_geos.get_value() is None:
            self.compute_geostrophy()
        
        if variational:
            u_cyclo, v_cyclo = self._compute_cyclogeostrophy_variational(n_it, lr)
        else:
            u_cyclo, v_cyclo = self._compute_cyclogeostrophy_iterative(n_it, eps)

        self.__u_cyclo.set_value(u_cyclo)
        self.__v_cyclo.set_value(v_cyclo)

        return u_cyclo, v_cyclo

    def _compute_cyclogeostrophy_variational(self, n_it: Union[int, None], lr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the cyclogeostrophic velocities using the variational formulation approach from geostrophic 
        velocities, their derivatives, and their coriolis factor

        :param n_it: maximum number of iterations, defaults to cyclo.N_IT_VAR if set to None
        :type n_it: Union[int, None]

        :returns: U and V cyclogeostrophic velocity fields
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if n_it is None:
            n_it = cyclo.N_IT_VAR

        u, v = cyclo._variational(self.__u_geos.get_value(), self.__v_geos.get_value(),
                                  self.__u_geos.get_dx(), self.__v_geos.get_dx(),
                                  self.__u_geos.get_dy(), self.__v_geos.get_dy(),
                                  self.__u_cyclo.get_coriolis_factor(), self.__v_cyclo.get_coriolis_factor(),
                                  n_it=n_it, lr=lr)
        return u, v

    def _compute_cyclogeostrophy_iterative(self, n_it: Union[int, None], eps: float) \
            -> Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]:
        """Computes the cyclogeostrophic velocities using the iterative approach from geostrophic velocities, and their 
        coriolis factor

        :param n_it: maximum number of iterations, defaults to cyclo.N_IT_IT if set to None
        :type n_it: Union[int, None]
        :param eps: residual tolerance
        :type eps: float

        :returns: U and V cyclogeostrophic velocity fields
        :rtype: Tuple[Union[np.ndarray, np.ma.MaskedArray], Union[np.ndarray, np.ma.MaskedArray]]
        """
        if n_it is None:
            n_it = cyclo.N_IT_IT

        u, v = cyclo._iterative(self.__u_geos.get_value(), self.__v_geos.get_value(),
                                self.__u_geos.get_dx(), self.__v_geos.get_dx(),
                                self.__u_geos.get_dy(), self.__v_geos.get_dy(),
                                self.__u_cyclo.get_coriolis_factor(), self.__v_cyclo.get_coriolis_factor(),
                                n_it=n_it, res_eps=eps)
        return u, v
