#!python
# cython: language_level=3
"""
    This script contains classes for a linear array

    This script requires that `numpy` and `scipy` be installed within
    the Python environment you are running this script in.

    This file can be imported as a module and contains the following
    class:

    * LinearArray

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
from scipy import signal
from antarray import AntennaArray


class LinearArray(AntennaArray):
    """
    A class defines basic parameters of a linear array.
    Inheritance of AntennaArray

    ...

    Attributes
    ----------
    size : int
        Total size of the linear array
    spacing : float
        Spacing between antenna elements
        (Normalized to wavelength)
    """

    def __init__(self, size, spacing=0.5):
        """
        Parameters
        ----------
        size : int
            Total size of the linear array
        spacing : float, optional
            Spacing between antenna elements
            (Normalized to wavelength), (default is 0.5)
        """

        self.size = size
        self.spacing = spacing
        self.window_dict = {
            'Square': self.square_win,
            'Chebyshev': self.chebyshev_win,
            'Taylor': self.taylor_win,
            'Hamming': self.hamming_win,
            'Hanning': self.hann_win
        }
        AntennaArray.__init__(self, x=np.arange(
            0, size, 1)*spacing, y=np.zeros(size))

    def update_parameters(self, **kwargs):
        """
        Update linear array parameters

        Parameters
        ----------
        size : int
            Total size of the linear array
        spacing : float
            Spacing between antenna elements
        """

        keys = ['size', 'spacing']
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in keys)
        self.__init__(self.size, self.spacing)

    def get_pattern(self, theta, beam_loc=0, window='Square', sll=-60, nbar=4):
        """
        Calculate the array factor

        Parameters
        ----------
        theta : 1-D array
            Angles for calculation (deg)
        beam_loc : float, optional
            Angle of the main beam (deg) (default is 0)
        window : str, optional
            Window type, supports `Square`, `Chebyshev`, `Taylor`, `Hamming`,
            and, `Hanning`
            (default is `Square`)
        sll : float, optional
            Desired peak sidelobe level in decibels (dB) relative to
            the mainlobe (default is -60)
        nbar : int, optional
            Number of nearly constant level sidelobes adjacent to the mainlobe
            (Only works with Taylor window) (default is 4)

        Returns
        -------
        AF : 1-D array
            Array pattern in decibels (dB)
        """

        weight = np.exp(-1j * 2 * np.pi * self.x * np.sin(
            beam_loc / 180 * np.pi)) * self.window_dict[window](
                self.size, sll, nbar)

        weight = weight / np.sum(np.abs(weight))

        theta_grid, array_geometry_grid = np.meshgrid(
            theta, self.x)
        A = np.exp(1j * 2 * np.pi * array_geometry_grid * np.sin(
            theta_grid / 180 * np.pi))

        AF = np.matmul(weight, A)

        return {'array_factor': AF,
                'weight': weight}

    def square_win(self, *args, **kwargs):
        return 1

    def chebyshev_win(self, array_size, sll, *args, **kwargs):
        return signal.chebwin(array_size, at=sll)

    def taylor_win(self, array_size, sll, nbar):
        return taylor(array_size, nbar, -sll)

    def hamming_win(self, array_size, *args, **kwargs):
        return signal.hamming(array_size)

    def hann_win(self, array_size, *args, **kwargs):
        return signal.hann(array_size)


def taylor(N, nbar=4, level=-30):
    """
    Return the Taylor window.
    The Taylor window allows for a selectable sidelobe suppression with a
    minimum broadening. This window is commonly used in radar processing [1].

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    nbar : int
        Number of nearly constant level sidelobes adjacent to the mainlobe
    level : float
        Desired peak sidelobe level in decibels (db) relative to the mainlobe

    Returns
    -------
    out : array
        The window, with the center value normalized to one (the value
        one appears only if the number of samples is odd).

    See Also
    --------
    kaiser, bartlett, blackman, hamming, hanning

    References
    -----
    .. [1] W. Carrara, R. Goodman, and R. Majewski "Spotlight Synthetic
               Aperture Radar: Signal Processing Algorithms" Pages 512-513,
               July 1995.
    """
    B = 10**(-level / 20)
    A = np.log(B + np.sqrt(B**2 - 1)) / np.pi
    s2 = nbar**2 / (A**2 + (nbar - 0.5)**2)
    ma = np.arange(1, nbar)

    def calc_Fm(m):
        numer = (-1)**(m + 1) * np.prod(1 - m**2 / s2 / (A**2 + (ma - 0.5)**2))
        denom = 2 * np.prod([1 - m**2 / j**2 for j in ma if j != m])
        return numer / denom

    calc_Fm_vec = np.vectorize(calc_Fm)
    Fm = calc_Fm_vec(ma)

    def W(n):
        return 2 * np.dot(Fm, np.cos(2 * np.pi * ma *
                                     (n - N / 2 + 1 / 2) / N)) + 1

    W_vec = np.vectorize(W)
    w = W_vec(range(N))

    # normalize (Note that this is not described in the original text [1])
    scale = 1.0 / W((N - 1) / 2)
    w *= scale
    return w
