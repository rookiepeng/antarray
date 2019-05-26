#!python
#cython: language_level=3
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
