#!python
# cython: language_level=3
"""
    This script contains classes for an antenna array

    This file can be imported as a module and contains the following
    class:

    * AntennaArray

    ----------
    AntArray - Antenna Array Analysis Module
    Copyright (C) 2018 - 2019  Zhengyu Peng
    E-mail: zpeng.me@gmail.com
    Website: https://zpeng.me

    `                      `
    -:.                  -#:
    -//:.              -###:
    -////:.          -#####:
    -/:.://:.      -###++##:
    ..   `://:-  -###+. :##:
           `:/+####+.   :##:
    .::::::::/+###.     :##:
    .////-----+##:    `:###:
     `-//:.   :##:  `:###/.
       `-//:. :##:`:###/.
         `-//:+######/.
           `-/+####/.
             `+##+.
              :##:
              :##:
              :##:
              :##:
              :##:
               .+:

"""

import numpy as np


class AntennaArray:
    """
    A class defines basic parameters of an antenna array

    ...

    Attributes
    ----------
    x : 1-d array
        Locations of the antenna elements on x-axis
        (Normalized to wavelength)
    y : 1-d array
        Locations of the antenna elements on y-axis
        (Normalized to wavelength)
    """

    def __init__(self, x, y=0):
        """
        Parameters
        ----------
        x : 1-d array
            Locations of the antenna elements on x-axis
            (Normalized to wavelength)
        y : 1-d array, optional
            Locations of the antenna elements on y-axis
            (Normalized to wavelength), (default is 0)
        """
        self.x = x
        self.y = y

    def get_pattern(self, u,
                    v,
                    weight=None,
                    polar=True):
        """
        Calculate the array factor

        Parameters
        ----------
        u : 1-D array
            If `polar` is `True': angles for calculation (deg)
            If `polar` is `False': directional cosines,
            $u=sin(\theta)cos(\phi)$
        v : 1-D array
            If `polar` is `True': angles for calculation (deg)
            If `polar` is `False': directional cosines,
            $v=sin(\theta)sin(\phi)$
        weight : 1-D array (complex), optional
            Weightings for array elements (default is None)
        polar : boolean
            Indicate the types of `u` and `v`

        Returns
        -------
        array_factor : 1-D array
            Array pattern in linear scale
        u : 1-D array
            Directional cosines
        v : 1-D array
            Directional cosines
        """

        size = len(self.x)

        if polar:
            theta_grid, phi_grid = np.meshgrid(u, v)
            u_grid = np.sin(theta_grid / 180 * np.pi) * \
                np.cos(phi_grid/180*np.pi)
            v_grid = np.sin(theta_grid / 180 * np.pi) * \
                np.sin(phi_grid/180*np.pi)
        else:
            u_grid, v_grid = np.meshgrid(u, v)

        if weight is None:
            weight = np.ones(size)

        AF = np.zeros(np.shape(u_grid), dtype=complex)

        for idx in range(0, size):
            AF = AF + \
                np.exp(1j * 2 * np.pi *
                       (self.x[idx]*u_grid + self.y[idx]*v_grid)) * weight[idx]

        return {'u': u_grid,
                'v': v_grid,
                'array_factor': AF}
