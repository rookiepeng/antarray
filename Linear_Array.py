"""
    This script contains classes for antenna arrays

    This script requires that `numpy` be installed within the Python
    environment you are running this script in.

    This file can be imported as a module and contains the following
    class:

    * Linear_Array

    ----------
    Antarray - Antenna Array Analysis Module
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
from antarray import Antenna_Array
from scipy.signal import chebwin, hamming, hann
from taylor_win import taylor


class Linear_Array(Antenna_Array):
    def __init__(self, size, spacing=0.5):
        self.size = size
        self.spacing = spacing
        self.window_dict = {
            'square': self.square_win,
            'chebyshev': self.chebyshev_win,
            'taylor': self.taylor_win,
            'hamming': self.hamming_win,
            'hanning': self.hann_win
        }
        Antenna_Array.__init__(self, x=np.arange(
            0, size, 1)*spacing)

    def get_pattern(self, theta, beam_loc=0, window='square', sll=-60, nbar=4):
        weight = np.exp(-1j * 2 * np.pi * self.x * np.sin(
            beam_loc / 180 * np.pi))
        * self.window_dict[window](self.size, sll, nbar)

        weight = weight / np.sum(np.abs(weight))

        theta_grid, array_geometry_grid = np.meshgrid(
            theta, self.x)
        A = np.exp(1j * 2 * np.pi * array_geometry_grid * np.sin(
            theta_grid / 180 * np.pi))

        AF = 20 * np.log10(np.abs(np.matmul(weight, A)) + 0.00001)

        return AF

    def update_parameters(self, **kwargs):
        keys = ['size', 'spacing']
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in keys)
        self.__init__(self.size, self.spacing)

    def square_win(self, *args, **kwargs):
        return 1

    def chebyshev_win(self, array_size, sll, *args, **kwargs):
        return chebwin(array_size, at=sll)

    def taylor_win(self, array_size, sll, nbar):
        return taylor(array_size, nbar, -sll)

    def hamming_win(self, array_size, *args, **kwargs):
        return hamming(array_size)

    def hann_win(self, array_size, *args, **kwargs):
        return hann(array_size)
