#!python
# cython: language_level=3
"""
    This script contains classes for a rectangular array

    This script requires that `numpy` and `scipy` be installed within
    the Python environment you are running this script in.

    This file can be imported as a module and contains the following
    class:

    * UniformRectangularArray

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
from arraybeam import AntennaArray


class UniformRectangularArray(AntennaArray):
    """
    A class defines basic parameters of a rectangular array.
    Inheritance of AntennaArray

    ...

    Attributes
    ----------
    sizex : int
        Size of the rectangular array on x-axis (azimuth direction)
    sizey : int
        Size of the rectangular array on y-axis (elevation direction)
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
            Size of the rectangular array on x-axis (azimuth direction)
        sizey : int, optional
            Size of the rectangular array on y-axis (elevation direction) (default is 1)
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
        self.x_array = np.arange(0, self.sizex, 1)*self.spacingx
        self.y_array = np.arange(0, self.sizey, 1)*self.spacingy

        AntennaArray.__init__(self, x=np.tile(
            self.x_array, self.sizey), y=np.repeat(self.y_array, self.sizex))

    def update_parameters(self, **kwargs):
        """
        Update rectangular array parameters

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_weights(self, beam_az, beam_el, weight_x, weight_y):
        """Build the 2-D complex weight matrix (steering + taper).

        Parameters
        ----------
        beam_az, beam_el : float
            Steering angles (deg).
        weight_x : 1-D array or None
            Amplitude taper along x (azimuth). None → uniform.
        weight_y : 1-D array or None
            Amplitude taper along y (elevation). None → uniform.

        Returns
        -------
        weight : 2-D complex array, shape (sizex, sizey)
            Normalised complex element weights.
        x_grid, y_grid : 2-D arrays
            Meshgrid of element positions.
        """
        y_grid, x_grid = np.meshgrid(self.y_array, self.x_array)

        if weight_x is None:
            weight_x = np.ones(self.sizex)
        if weight_y is None:
            weight_y = np.ones(self.sizey)

        window = np.outer(weight_x, weight_y)

        weight = np.exp(1j * 2 * np.pi * (
            x_grid * np.sin(np.radians(beam_az)) +
            y_grid * np.sin(np.radians(beam_el)))) * window
        weight = weight / np.sum(np.abs(weight))

        return weight, x_grid, y_grid

    @staticmethod
    def _tile_count(spacing):
        """Number of tiles needed to cover the full k-space."""
        return int(np.ceil(spacing - 0.5)) * 2 + 1

    @staticmethod
    def _k_axis(nfft, tile, spacing):
        """Build a tiled k-space axis."""
        return 0.5 * np.linspace(
            -tile, tile, nfft * tile, endpoint=False) / spacing

    @staticmethod
    def _steer(positions, angle_deg):
        """Steering vector for a set of positions at a given angle."""
        return np.exp(-1j * 2 * np.pi * positions *
                      np.sin(np.radians(angle_deg)))

    @staticmethod
    def _clip_visible(k):
        """Clip k-space to the visible region |k| ≤ 1 → angles (deg)."""
        mask = (k >= -1) & (k <= 1)
        return mask, np.degrees(np.arcsin(k[mask]))

    # ------------------------------------------------------------------
    # Public pattern methods
    # ------------------------------------------------------------------

    def get_pattern_2d(self, nfft_az=128, nfft_el=128,
                       beam_az=0, beam_el=0,
                       weight_x=None, weight_y=None):
        """Compute the full 2-D array factor over azimuth and elevation.

        Coordinate System
        -----------------
        - x-axis → Azimuth (horizontal plane)
        - y-axis → Elevation (vertical plane)

        Parameters
        ----------
        nfft_az : int, optional
            FFT size for the azimuth (x) dimension. (default is 128)
        nfft_el : int, optional
            FFT size for the elevation (y) dimension. (default is 128)
        beam_az : float, optional
            Steering angle in azimuth (deg). (default is 0)
        beam_el : float, optional
            Steering angle in elevation (deg). (default is 0)
        weight_x : 1-D array, optional
            Amplitude taper along x-axis (length == sizex).
            None → uniform. (default is None)
        weight_y : 1-D array, optional
            Amplitude taper along y-axis (length == sizey).
            None → uniform. (default is None)

        Returns
        -------
        dict
            'array_factor' : 2-D array – pattern in linear scale
            'x'            : 1-D array – element x-positions (λ)
            'y'            : 1-D array – element y-positions (λ)
            'weight'       : 1-D array – flattened complex weights
            'azimuth'      : 1-D array – azimuth angles (deg)
            'elevation'    : 1-D array – elevation angles (deg)
        """
        weight, _, _ = self._compute_weights(
            beam_az, beam_el, weight_x, weight_y)

        xy = np.ones((self.sizex, self.sizey), dtype=complex)

        tilex = self._tile_count(self.spacingx)
        tiley = self._tile_count(self.spacingy)
        k_az = self._k_axis(nfft_az, tilex, self.spacingx)
        k_el = self._k_axis(nfft_el, tiley, self.spacingy)

        AF = np.fft.fftshift(np.fft.fft2(
            xy * weight, (nfft_az, nfft_el)))
        AF = np.tile(AF, (tilex, tiley))

        az_mask, azimuth = self._clip_visible(k_az)
        el_mask, elevation = self._clip_visible(k_el)
        AF = AF[np.ix_(az_mask, el_mask)]

        return {
            'array_factor': AF,
            'x': self.x,
            'y': self.y,
            'weight': weight.ravel(order='F'),
            'azimuth': azimuth,
            'elevation': elevation,
        }

    def get_pattern_az(self, nfft=512,
                       beam_az=0, beam_el=0,
                       weight_x=None, weight_y=None,
                       cut_el=None):
        """Compute a 1-D azimuth cut of the array factor.

        The pattern is computed as a function of azimuth at a single
        fixed elevation angle.

        Parameters
        ----------
        nfft : int, optional
            FFT size for azimuth. (default is 512)
        beam_az : float, optional
            Steering angle in azimuth (deg). (default is 0)
        beam_el : float, optional
            Steering angle in elevation (deg). (default is 0)
        weight_x : 1-D array, optional
            Amplitude taper along x-axis (length == sizex).
            None → uniform. (default is None)
        weight_y : 1-D array, optional
            Amplitude taper along y-axis (length == sizey).
            None → uniform. (default is None)
        cut_el : float, optional
            Fixed elevation angle (deg) at which the azimuth cut is taken.
            If None, defaults to beam_el. (default is None)

        Returns
        -------
        dict
            'array_factor' : 1-D array – pattern in linear scale
            'raw_fft'      : 2-D array – intermediate FFT matrix (nfft × sizey)
            'x'            : 1-D array – element x-positions (λ)
            'y'            : 1-D array – element y-positions (λ)
            'weight'       : 1-D array – flattened complex weights
            'azimuth'      : 1-D array – azimuth angles (deg)
            'elevation'    : float     – the fixed elevation angle (deg)
        """
        weight, _, _ = self._compute_weights(
            beam_az, beam_el, weight_x, weight_y)

        xy = np.ones((self.sizex, self.sizey), dtype=complex)

        tilex = self._tile_count(self.spacingx)
        k_az = self._k_axis(nfft, tilex, self.spacingx)

        A = np.fft.fftshift(np.fft.fft(
            xy * weight, nfft, axis=0), axes=0)

        el_angle = beam_el if cut_el is None else cut_el
        AF = A @ self._steer(self.y_array, el_angle)
        AF = np.tile(AF, tilex)

        az_mask, azimuth = self._clip_visible(k_az)
        AF = AF[az_mask]

        return {
            'array_factor': AF,
            'raw_fft': A,
            'x': self.x,
            'y': self.y,
            'weight': weight.ravel(order='F'),
            'azimuth': azimuth,
            'elevation': np.array(el_angle),
        }

    def get_pattern_el(self, nfft=512,
                       beam_az=0, beam_el=0,
                       weight_x=None, weight_y=None,
                       cut_az=None):
        """Compute a 1-D elevation cut of the array factor.

        The pattern is computed as a function of elevation at a single
        fixed azimuth angle.

        Parameters
        ----------
        nfft : int, optional
            FFT size for elevation. (default is 512)
        beam_az : float, optional
            Steering angle in azimuth (deg). (default is 0)
        beam_el : float, optional
            Steering angle in elevation (deg). (default is 0)
        weight_x : 1-D array, optional
            Amplitude taper along x-axis (length == sizex).
            None → uniform. (default is None)
        weight_y : 1-D array, optional
            Amplitude taper along y-axis (length == sizey).
            None → uniform. (default is None)
        cut_az : float, optional
            Fixed azimuth angle (deg) at which the elevation cut is taken.
            If None, defaults to beam_az. (default is None)

        Returns
        -------
        dict
            'array_factor' : 1-D array – pattern in linear scale
            'raw_fft'      : 2-D array – intermediate FFT matrix (sizex × nfft)
            'x'            : 1-D array – element x-positions (λ)
            'y'            : 1-D array – element y-positions (λ)
            'weight'       : 1-D array – flattened complex weights
            'azimuth'      : float     – the fixed azimuth angle (deg)
            'elevation'    : 1-D array – elevation angles (deg)
        """
        weight, _, _ = self._compute_weights(
            beam_az, beam_el, weight_x, weight_y)

        xy = np.ones((self.sizex, self.sizey), dtype=complex)

        tiley = self._tile_count(self.spacingy)
        k_el = self._k_axis(nfft, tiley, self.spacingy)

        A = np.fft.fftshift(np.fft.fft(
            xy * weight, nfft, axis=1), axes=1)

        az_angle = beam_az if cut_az is None else cut_az
        AF = A.T @ self._steer(self.x_array, az_angle)
        AF = np.tile(AF, tiley)

        el_mask, elevation = self._clip_visible(k_el)
        AF = AF[el_mask]

        return {
            'array_factor': AF,
            'raw_fft': A,
            'x': self.x,
            'y': self.y,
            'weight': weight.ravel(order='F'),
            'azimuth': np.array(az_angle),
            'elevation': elevation,
        }
