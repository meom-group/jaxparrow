from typing import Union

import numpy as np
import numpy.ma as ma

from jaxparrow.tools import geometry as geo


class Variable:
    """Class representing a geophysical variable on a C-grid"""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, value: Union[np.ndarray, None], mask: Union[np.ndarray, None]):
        """Constructor

        :param lat: the latitude grid to use
        :type lat: np.ndarray
        :param lon: the longitude grid to set
        :type: np.ndarray
        :param value: the value grid to use
        :type value: Union[np.ndarray, None]
        :param mask: the mask grid to set
        :type: Union[np.ndarray, None]
        """

        #: the latitude grid
        self.__lat = lat
        #: the longitude grid
        self.__lon = lon
        #: the value grid (can be equal to `None` if unknown)
        self.__value = value
        #: the mask grid (can be equal to `None` if irrelevant)
        self.__mask = 1 - mask

        self.__apply_mask()

        #: the Coriolis factor grid
        self.__coriolis_factor = geo.compute_coriolis_factor(self.__lat)
        dx, dy = geo.compute_spatial_step(self.__lat, self.__lon)
        #: the spatial step along x grid
        self.__dx = dx
        #: the spatial step along y grid
        self.__dy = dy

    # Getter methods

    def get_lat(self) -> Union[np.ndarray, np.ma.MaskedArray]:
        """Getter for the latitude grid

        :returns: the latitude grid
        :rtype: Union[np.ndarray, np.ma.MaskedArray]
        """
        return self.__lat

    def get_lon(self) -> Union[np.ndarray, np.ma.MaskedArray]:
        """Getter for the longitude grid

        :returns: the longitude grid
        :rtype: Union[np.ndarray, np.ma.MaskedArray]
        """
        return self.__lon

    def get_value(self) -> Union[np.ndarray, np.ma.MaskedArray, None]:
        """Getter for the value grid

        :returns: the value grid
        :rtype: Union[np.ndarray, np.ma.MaskedArray, None]
        """
        return self.__value

    def get_mask(self) -> Union[np.ndarray, None]:
        """Getter for the mask grid

        :returns: the mask grid
        :rtype: Union[np.ndarray, None]
        """
        return self.__mask

    def get_coriolis_factor(self) -> Union[np.ndarray, np.ma.MaskedArray]:
        """Getter for the Coriolis factor grid

        :returns: the Coriolis factor grid
        :rtype: Union[np.ndarray, np.ma.MaskedArray]
        """
        return self.__coriolis_factor

    def get_dx(self) -> np.ndarray:
        """Getter for the spatial step along x grid

        :returns: the spatial step along x grid
        :rtype: np.ndarray
        """
        return self.__dx

    def get_dy(self) -> np.ndarray:
        """Getter for the spatial step along y grid

        :returns: the spatial step along y grid
        :rtype: np.ndarray
        """
        return self.__dy

    # Setter methods

    def set_value(self, value: Union[np.ndarray, np.ma.MaskedArray]):
        """Setter for the value grid

        :param value: the value grid to set
        :type value: Union[np.ndarray, np.ma.MaskedArray]
        """
        if isinstance(value, np.ndarray):
            value = ma.masked_array(value, self.__mask)

        self.__value = value

    def set_lat(self, lat: np.ndarray):
        """Setter for the latitude grid

        :param lat: the latitude grid to set
        :type lat: np.ndarray
        """
        self.__lat = lat

    def set_lon(self, lon: np.ndarray):
        """Setter for the longitude grid

        :param lon: the longitude grid to set
        :type lon: np.ndarray
        """
        self.__lon = lon

    def set_mask(self, mask: np.ndarray):
        """Setter for the mask grid.
        Sets as `self.__mask = 1 - mask`

        :param mask: the mask grid to set
        :type mask: np.ndarray
        """
        self.__mask = 1 - mask

    def set_coriolis_factor(self, coriolis_factor: np.ndarray):
        """Setter for the Coriolis factor grid

        :param coriolis_factor: Coriolis factor grid to set
        :type coriolis_factor: np.ndarray
        """
        self.__coriolis_factor = coriolis_factor

    def set_dx(self, dx: np.ndarray):
        """Setter for the spatial step along x grid

        :param dx: spatial step along x grid to set
        :type dx: np.ndarray
        """
        self.__dx = dx

    def set_dy(self, dy: np.ndarray):
        """Setter for the spatial step along y grid

        :param dy: spatial step along y grid to set
        :type dy: np.ndarray
        """
        self.__dy = dy

    # Methods

    def __apply_mask(self):
        """Masks the data that are not in the domain of the measurements"""
        if self.__mask is None:
            return

        if self.__value is not None:
            self.__value = ma.masked_array(self.__value, self.__mask)
        self.__lon = ma.masked_array(self.__lon, self.__mask)
        self.__lat = ma.masked_array(self.__lat, self.__mask)
