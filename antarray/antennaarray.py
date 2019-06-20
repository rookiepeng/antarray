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

    def get_pattern(self, azimuth,
                    elevation,
                    weight=None):
        """
        Calculate the array factor

        Parameters
        ----------
        azimuth : 1-D array
            Azimuth angles (deg)
        elevation : 1-D array
            Elevation angles (deg)
        weight : 1-D array (complex), optional
            Weightings for array elements (default is None)

        Returns
        -------
        array_factor : 1-D array
            Array pattern in linear scale
        azimuth : 1-D array
            Azimuth angles
        elevation : 1-D array
            Elevation angles
        """

        size = len(self.x)

        azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation)
        u_grid = np.sin(azimuth_grid / 180 * np.pi)
        v_grid = np.sin(elevation_grid/180*np.pi)

        if weight is None:
            weight = np.ones(size)

        AF = np.zeros(np.shape(u_grid), dtype=complex)

        for idx in range(0, size):
            AF = AF + \
                np.exp(-1j * 2 * np.pi *
                       (self.x[idx]*u_grid + self.y[idx]*v_grid)) * weight[idx]

        return {'array_factor': np.transpose(AF)}
