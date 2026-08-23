"""
Uniform rectangular (planar) array with FFT-accelerated pattern methods.

Coordinate system
-----------------
- x-axis -> azimuth (horizontal plane)
- y-axis -> elevation (vertical plane)

See :mod:`arraybeam.antenna_array` for the sign and sine-space conventions
shared by the whole package.

Copyright (C) 2018-2026 Zhengyu Peng <zpeng.me@gmail.com>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .antenna_array import _TWO_PI_J, AntennaArray

__all__ = ["UniformRectangularArray"]


class UniformRectangularArray(AntennaArray):
    """A uniformly spaced rectangular array.

    Parameters
    ----------
    sizex : int
        Number of elements along the x-axis (azimuth direction).
    sizey : int, optional
        Number of elements along the y-axis (elevation direction).
        (default is 1)
    spacingx, spacingy : float, optional
        Element spacing along each axis, normalised to the wavelength.
        (default is 0.5)

    Attributes
    ----------
    sizex, sizey : int
        Element counts along each axis.
    spacingx, spacingy : float
        Element spacings along each axis.
    x_array, y_array : ndarray of float
        The 1-D generating position vectors for each axis.
    x, y : ndarray of float
        Flattened per-element positions, x varying fastest.
    """

    _PARAMETERS = ("sizex", "sizey", "spacingx", "spacingy")

    def __init__(
        self,
        sizex: int,
        sizey: int = 1,
        spacingx: float = 0.5,
        spacingy: float = 0.5,
    ) -> None:
        self._configure(sizex, sizey, spacingx, spacingy)

    # ------------------------------------------------------------------
    # Validation and geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_size(value, name: str) -> int:
        """Check that an element count is a whole number of at least one."""
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer; got {value!r}")
        if isinstance(value, (float, np.floating)):
            if not float(value).is_integer():
                raise TypeError(f"{name} must be a whole number; got {value!r}")
            value = int(value)
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer; got {value!r}")
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be >= 1; got {value}")
        return value

    @staticmethod
    def _validate_spacing(value, name: str) -> float:
        """Check that a spacing is finite and strictly positive."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be a number; got {value!r}") from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number; got {value}")
        return value

    def _configure(self, sizex, sizey, spacingx, spacingy) -> None:
        """Validate parameters and rebuild the element geometry."""
        self.sizex = self._validate_size(sizex, "sizex")
        self.sizey = self._validate_size(sizey, "sizey")
        self.spacingx = self._validate_spacing(spacingx, "spacingx")
        self.spacingy = self._validate_spacing(spacingy, "spacingy")

        self.x_array = np.arange(self.sizex) * self.spacingx
        self.y_array = np.arange(self.sizey) * self.spacingy

        super().__init__(
            x=np.tile(self.x_array, self.sizey),
            y=np.repeat(self.y_array, self.sizex),
        )

    def update_parameters(self, **kwargs) -> None:
        """Update array parameters in place and rebuild the geometry.

        Parameters
        ----------
        sizex, sizey : int, optional
            New element counts.
        spacingx, spacingy : float, optional
            New element spacings, normalised to the wavelength.

        Raises
        ------
        TypeError
            If an unrecognised parameter name is given.  Unknown names are
            rejected rather than ignored so that typos cannot silently
            become no-ops.
        """
        # Bind to this class explicitly: subclasses expose their own
        # parameter names and forward the translated ones here.
        names = UniformRectangularArray._PARAMETERS
        unknown = set(kwargs) - set(names)
        if unknown:
            raise TypeError(
                f"unknown parameter(s): {', '.join(sorted(unknown))}; "
                f"expected any of {', '.join(names)}"
            )
        self._configure(**{k: kwargs.get(k, getattr(self, k)) for k in names})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _element_weights(
        self, beam_az, beam_el, weight_x, weight_y
    ) -> NDArray[np.complex128]:
        """Build the normalised 2-D complex weight matrix (steering + taper).

        Returns an array of shape ``(sizex, sizey)``.  Flattening it with
        ``order='F'`` yields the per-element ordering of ``self.x``/``self.y``
        and matches the convention of
        :meth:`~arraybeam.antenna_array.AntennaArray.steering_weights`.
        """
        taper_x = self._validate_taper(weight_x, self.sizex, "weight_x")
        taper_y = self._validate_taper(weight_y, self.sizey, "weight_y")

        phase = np.exp(
            _TWO_PI_J
            * (
                self.x_array[:, None] * np.sin(np.radians(beam_az))
                + self.y_array[None, :] * np.sin(np.radians(beam_el))
            )
        )
        return self._normalise(phase * np.outer(taper_x, taper_y))

    @staticmethod
    def _tile_count(spacing: float) -> int:
        """Number of tiles needed to cover the full k-space."""
        return int(np.ceil(spacing - 0.5)) * 2 + 1

    @staticmethod
    def _k_axis(nfft: int, tile: int, spacing: float) -> NDArray[np.float64]:
        """Build a tiled k-space axis."""
        return 0.5 * np.linspace(-tile, tile, nfft * tile, endpoint=False) / spacing

    @staticmethod
    def _steer(positions, angle_deg):
        """Steering vector for a set of positions at a given angle."""
        return np.exp(-_TWO_PI_J * positions * np.sin(np.radians(angle_deg)))

    @staticmethod
    def _clip_visible(k):
        """Clip k-space to the visible region ``|k| <= 1`` -> angles (deg)."""
        mask = (k >= -1) & (k <= 1)
        return mask, np.degrees(np.arcsin(k[mask]))

    # ------------------------------------------------------------------
    # Public pattern methods
    # ------------------------------------------------------------------

    def get_pattern_2d(
        self,
        nfft_az: int = 128,
        nfft_el: int = 128,
        beam_az: float = 0,
        beam_el: float = 0,
        weight_x: ArrayLike | None = None,
        weight_y: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Compute the full 2-D array factor over azimuth and elevation.

        The pattern is sampled on the regular sine-space grid produced by
        the FFT, so the returned ``azimuth`` and ``elevation`` axes are
        non-uniform in degrees.  Use
        :meth:`~arraybeam.antenna_array.AntennaArray.get_pattern` when a
        specific angle grid is required.

        Parameters
        ----------
        nfft_az, nfft_el : int, optional
            FFT sizes for the azimuth (x) and elevation (y) dimensions.
            (default is 128)
        beam_az, beam_el : float, optional
            Steering angles (deg). (default is 0)
        weight_x : array_like, optional
            Amplitude taper along the x-axis (length ``sizex``).
            ``None`` gives uniform illumination.
        weight_y : array_like, optional
            Amplitude taper along the y-axis (length ``sizey``).
            ``None`` gives uniform illumination.

        Returns
        -------
        dict
            'array_factor' : 2-D ndarray - pattern in linear scale
            'x', 'y'       : 1-D ndarray - element positions (wavelengths)
            'weight'       : 1-D ndarray - flattened complex weights
            'azimuth'      : 1-D ndarray - azimuth angles (deg)
            'elevation'    : 1-D ndarray - elevation angles (deg)
        """
        weight = self._element_weights(beam_az, beam_el, weight_x, weight_y)

        tilex = self._tile_count(self.spacingx)
        tiley = self._tile_count(self.spacingy)
        k_az = self._k_axis(nfft_az, tilex, self.spacingx)
        k_el = self._k_axis(nfft_el, tiley, self.spacingy)

        array_factor = np.fft.fftshift(np.fft.fft2(weight, (nfft_az, nfft_el)))
        array_factor = np.tile(array_factor, (tilex, tiley))

        az_mask, azimuth = self._clip_visible(k_az)
        el_mask, elevation = self._clip_visible(k_el)
        array_factor = array_factor[np.ix_(az_mask, el_mask)]

        return {
            "array_factor": array_factor,
            "x": self.x,
            "y": self.y,
            "weight": weight.ravel(order="F"),
            "azimuth": azimuth,
            "elevation": elevation,
        }

    def get_pattern_az(
        self,
        nfft: int = 512,
        beam_az: float = 0,
        beam_el: float = 0,
        weight_x: ArrayLike | None = None,
        weight_y: ArrayLike | None = None,
        cut_el: float | None = None,
    ) -> dict[str, Any]:
        """Compute a 1-D azimuth cut of the array factor.

        The pattern is computed as a function of azimuth at a single fixed
        elevation angle.

        Parameters
        ----------
        nfft : int, optional
            FFT size for azimuth. (default is 512)
        beam_az, beam_el : float, optional
            Steering angles (deg). (default is 0)
        weight_x : array_like, optional
            Amplitude taper along the x-axis (length ``sizex``).
        weight_y : array_like, optional
            Amplitude taper along the y-axis (length ``sizey``).
        cut_el : float, optional
            Fixed elevation angle (deg) at which the cut is taken.
            ``None`` (the default) uses ``beam_el``.

        Returns
        -------
        dict
            'array_factor' : 1-D ndarray - pattern in linear scale
            'raw_fft'      : 2-D ndarray - the intermediate FFT matrix
                             (``nfft`` x ``sizey``); an implementation
                             detail exposed so callers can take further
                             cuts cheaply
            'x', 'y'       : 1-D ndarray - element positions (wavelengths)
            'weight'       : 1-D ndarray - flattened complex weights
            'azimuth'      : 1-D ndarray - azimuth angles (deg)
            'elevation'    : 0-d ndarray - the fixed elevation angle (deg)
        """
        weight = self._element_weights(beam_az, beam_el, weight_x, weight_y)

        tilex = self._tile_count(self.spacingx)
        k_az = self._k_axis(nfft, tilex, self.spacingx)

        raw_fft = np.fft.fftshift(np.fft.fft(weight, nfft, axis=0), axes=0)

        el_angle = beam_el if cut_el is None else cut_el
        array_factor = raw_fft @ self._steer(self.y_array, el_angle)
        array_factor = np.tile(array_factor, tilex)

        az_mask, azimuth = self._clip_visible(k_az)
        array_factor = array_factor[az_mask]

        return {
            "array_factor": array_factor,
            "raw_fft": raw_fft,
            "x": self.x,
            "y": self.y,
            "weight": weight.ravel(order="F"),
            "azimuth": azimuth,
            "elevation": np.asarray(el_angle, dtype=float),
        }

    def get_pattern_el(
        self,
        nfft: int = 512,
        beam_az: float = 0,
        beam_el: float = 0,
        weight_x: ArrayLike | None = None,
        weight_y: ArrayLike | None = None,
        cut_az: float | None = None,
    ) -> dict[str, Any]:
        """Compute a 1-D elevation cut of the array factor.

        The pattern is computed as a function of elevation at a single fixed
        azimuth angle.

        Parameters
        ----------
        nfft : int, optional
            FFT size for elevation. (default is 512)
        beam_az, beam_el : float, optional
            Steering angles (deg). (default is 0)
        weight_x : array_like, optional
            Amplitude taper along the x-axis (length ``sizex``).
        weight_y : array_like, optional
            Amplitude taper along the y-axis (length ``sizey``).
        cut_az : float, optional
            Fixed azimuth angle (deg) at which the cut is taken.
            ``None`` (the default) uses ``beam_az``.

        Returns
        -------
        dict
            'array_factor' : 1-D ndarray - pattern in linear scale
            'raw_fft'      : 2-D ndarray - the intermediate FFT matrix
                             (``sizex`` x ``nfft``); an implementation
                             detail exposed so callers can take further
                             cuts cheaply
            'x', 'y'       : 1-D ndarray - element positions (wavelengths)
            'weight'       : 1-D ndarray - flattened complex weights
            'azimuth'      : 0-d ndarray - the fixed azimuth angle (deg)
            'elevation'    : 1-D ndarray - elevation angles (deg)
        """
        weight = self._element_weights(beam_az, beam_el, weight_x, weight_y)

        tiley = self._tile_count(self.spacingy)
        k_el = self._k_axis(nfft, tiley, self.spacingy)

        raw_fft = np.fft.fftshift(np.fft.fft(weight, nfft, axis=1), axes=1)

        az_angle = beam_az if cut_az is None else cut_az
        array_factor = raw_fft.T @ self._steer(self.x_array, az_angle)
        array_factor = np.tile(array_factor, tiley)

        el_mask, elevation = self._clip_visible(k_el)
        array_factor = array_factor[el_mask]

        return {
            "array_factor": array_factor,
            "raw_fft": raw_fft,
            "x": self.x,
            "y": self.y,
            "weight": weight.ravel(order="F"),
            "azimuth": np.asarray(az_angle, dtype=float),
            "elevation": elevation,
        }
