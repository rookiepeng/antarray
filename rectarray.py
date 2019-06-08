#!python
# cython: language_level=3
"""
    This script contains classes for a rectangular array

    This script requires that `numpy` and `scipy` be installed within
    the Python environment you are running this script in.

    This file can be imported as a module and contains the following
    class:

    * RectArray

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


class RectArray(AntennaArray):
    """
    A class defines basic parameters of a rectangular array.
    Inheritance of AntennaArray

    ...

    Attributes
    ----------
    sizex : int
        Size of the rectangular array on x-axis
    sizey : int
        Size of the rectangular array on y-axis
    spacingx : float
        Spacing between antenna elements on x-axis
        (Normalized to wavelength)
    spacingy : float
        Spacing between antenna elements on y-axis
        (Normalized to wavelength)
    """

    def __init__(self, sizex, sizey=1, spacingx=0.5, spacingy=0.5):
        """
        Parameters
        ----------
        sizex : int
            Size of the rectangular array on x-axis
        sizey : int, optional
            Size of the rectangular array on y-axis (default is 1)
        spacingx : float, optional
            Spacing between antenna elements on x-axis
            (Normalized to wavelength), (default is 0.5)
        spacingy : float, optional
            Spacing between antenna elements on y-axis
            (Normalized to wavelength), (default is 0.5)
        """

        self.sizex = sizex
        self.sizey = sizey
        self.spacingx = spacingx
        self.spacingy = spacingy
        self.window_dict = {
            'Square': self.square_win,
            'Chebyshev': self.chebyshev_win,
            'Taylor': self.taylor_win,
            'Hamming': self.hamming_win,
            'Hanning': self.hann_win
        }
        AntennaArray.__init__(self, x=np.arange(
            0, sizex, 1)*spacingx, y=np.arange(
            0, sizey, 1)*spacingy)

    def update_parameters(self, **kwargs):
        """
        Update linear array parameters

        Parameters
        ----------
        sizex : int, optional
            Size of the rectangular array on x-axis
        sizey : int, optional
            Size of the rectangular array on y-axis
        spacingx : float, optional
            Spacing between antenna elements on x-axis
            (Normalized to wavelength)
        spacingy : float, optional
            Spacing between antenna elements on y-axis
            (Normalized to wavelength)
        """

        keys = ['sizex', 'sizey', 'spacingx', 'spacingy']
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in keys)
        self.__init__(self.sizex, self.sizey, self.spacingx, self.spacingy)

    def get_pattern(self,
                    nfft_az=512,
                    nfft_el=512,
                    beam_az=0,
                    beam_el=0,
                    windowx='Square',
                    sllx=-60,
                    nbarx=4,
                    windowy='Square',
                    slly=-60,
                    nbary=4,
                    plot_az=None,
                    plot_el=None):
        """
        Calculate the array factor

        Parameters
        ----------
        nfft_az : int, optional
            FFT points for azimuth beamforming.
            Azimuth is the plane of x. (default is 512)
        nfft_el : int, optional
            FFT points for elevation beamforming.
            Elevation is the plane of y. (default is 512)
        beam_az : float, optional
            Angle of the main beam (deg) on azimuth. (default is 0)
        beam_el : float, optional
            Angle of the main beam (deg) on elevation. (default is 0)
        windowx : str, optional
            Window type along x axis, supports `Square`, `Chebyshev`,
            `Taylor`, `Hamming`, and, `Hanning`
            (default is `Square`)
        sllx : float, optional
            Desired peak sidelobe level in decibels (dB) relative to
            the mainlobe for window along x axis. Only valid with
            Chebyshev window and Taylor window. (default is -60)
        nbarx : int, optional
            Number of nearly constant level sidelobes adjacent to the mainlobe
            along x axis. Only works with Taylor window. (default is 4)
        windowy : str, optional
            Window type along y axis, supports `Square`, `Chebyshev`,
            `Taylor`, `Hamming`, and, `Hanning`
            (default is `Square`)
        slly : float, optional
            Desired peak sidelobe level in decibels (dB) relative to
            the mainlobe for window along y axis. Only valid with
            Chebyshev window and Taylor window. (default is -60)
        nbary : int, optional
            Number of nearly constant level sidelobes adjacent to the mainlobe
            along y axis. Only works with Taylor window. (default is 4)
        plot_az : float, optional
            If `nfft_az == 1`, `plot_az` indicates the azimuth angle of the
            returned elevation pattern. (default `plot_az = beam_az`)
        plot_el : float, optional
            If `nfft_el == 1`, `plot_el` indicates the elevation angle of the
            returned azimuth pattern. (default `plot_el = beam_el`)

        Returns
        -------
        dict(
            'array_factor' : 1-D array or 2-D array
                Antenna array pattern.
            'weight' : 2-D array
                Weights for array elements
            'x' : 1-D array
                x axis locations of antenna array elements
            'y' : 1-D array
                y axis locations of antenna array elements
            'azimuth' : 1-D array
                Corresponded azimuth angles for `array_factor
            'elevation' : 1-D array
                Corresponded elevation angles for `array_factor
        )
        """
        y_grid, x_grid = np.meshgrid(self.y, self.x)

        xy = np.ones((self.sizex, self.sizey), dtype=complex)

        window = np.matmul(np.transpose(np.array([self.window_dict[windowx](
            self.sizex, sllx, nbarx)])), np.array([self.window_dict[windowy](
                self.sizey, slly, nbary)]))

        weight = np.exp(1j * 2 * np.pi * (x_grid * np.sin(
            beam_az / 180 * np.pi) + y_grid * np.sin(
            beam_el/180*np.pi)))*window

        weight = weight / np.sum(np.abs(weight))

        tilex = int(np.ceil(self.spacingx-0.5))*2+1
        k_az = 0.5*np.linspace(-tilex, tilex, nfft_az*tilex)/self.spacingx
        tiley = int(np.ceil(self.spacingy-0.5))*2+1
        k_el = 0.5*np.linspace(-tiley, tiley, nfft_el*tiley)/self.spacingy

        if nfft_el <= 1 and nfft_az > 1:
            A = np.fft.fftshift(np.fft.fft(xy*weight, nfft_az, axis=0), axes=0)
            if plot_el is None:
                plot_weight = np.array(
                    [np.exp(-1j * 2 * np.pi * self.y * np.sin(
                        beam_el / 180 * np.pi))])
                elevation = np.array(beam_el)
            else:
                plot_weight = np.array(
                    [np.exp(-1j * 2 * np.pi * self.y * np.sin(
                        plot_el / 180 * np.pi))])
                elevation = np.array(plot_el)

            AF = np.matmul(A, np.transpose(plot_weight))[:, 0]
            AF = np.tile(AF, tilex)
            AF = AF[np.where(np.logical_and(
                k_az >= -1, k_az <= 1))[0]]
            k_az = k_az[np.where(np.logical_and(
                k_az >= -1, k_az <= 1))[0]]
            azimuth = np.arcsin(k_az)/np.pi*180

        elif nfft_az <= 1 and nfft_el > 1:
            A = np.fft.fftshift(np.fft.fft(xy*weight, nfft_el, axis=1), axes=1)
            if plot_az is None:
                plot_weight = np.array(
                    [np.exp(-1j * 2 * np.pi * self.x * np.sin(
                        beam_az / 180 * np.pi))])
                azimuth = np.array(beam_az)
            else:
                plot_weight = np.array(
                    [np.exp(-1j * 2 * np.pi * self.x * np.sin(
                        plot_az / 180 * np.pi))])
                azimuth = np.array(plot_az)

            AF = np.matmul(np.transpose(A), np.transpose(plot_weight))[:, 0]
            AF = np.tile(AF, tiley)
            AF = AF[np.where(np.logical_and(
                k_el >= -1, k_el <= 1))[0]]
            k_el = k_el[np.where(np.logical_and(
                k_el >= -1, k_el <= 1))[0]]
            elevation = np.arcsin(k_el)/np.pi*180

        elif nfft_el > 1 and nfft_az > 1:
            AF = np.fft.fftshift(np.fft.fft2(xy*weight, (nfft_az, nfft_el)))
            AF = np.tile(AF, (tilex, 1))
            AF = np.tile(AF, (1, tiley))
            AF = AF[np.where(np.logical_and(
                k_az >= -1, k_az <= 1))[0], :]
            AF = AF[:, np.where(np.logical_and(
                k_el >= -1, k_el <= 1))[0]]
            k_az = k_az[np.where(np.logical_and(
                k_az >= -1, k_az <= 1))[0]]
            k_el = k_el[np.where(np.logical_and(
                k_el >= -1, k_el <= 1))[0]]
            azimuth = np.arcsin(k_az)/np.pi*180
            elevation = np.arcsin(k_el)/np.pi*180

        return {
            'array_factor': AF,
            'weight': weight,
            'x': x_grid,
            'y': y_grid,
            'azimuth': azimuth,
            'elevation': elevation}

    def square_win(self, array_size, *args, **kwargs):
        return np.ones(array_size)

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
