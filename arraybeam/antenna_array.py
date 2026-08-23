"""
Core geometry and array-factor computation for :mod:`arraybeam`.

Conventions
-----------
All element positions and spacings are normalised to the wavelength.
Angles are given in degrees.

The array is analysed in *separable sine space*::

    u = sin(azimuth)        v = sin(elevation)

which is the usual radar convention for a planar array lying in the x-y
plane.  Note that this is **not** the true spherical direction cosine
(``u = cos(elevation) * sin(azimuth)``): azimuth and elevation are treated
as independent cuts.  A consequence is that the region ``u**2 + v**2 > 1``
is not physically visible even though the transforms still produce values
there.

The array factor and the steering weights use a single, fixed sign
convention throughout the package::

    AF(u, v) = sum_i  w_i * exp(-j*2*pi*(x_i*u + y_i*v))

    w_i(beam) = exp(+j*2*pi*(x_i*sin(beam_az) + y_i*sin(beam_el))) * taper_i

with the weights normalised so that ``sum(|w|) == 1``.

Copyright (C) 2018-2026 Zhengyu Peng <zpeng.me@gmail.com>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["AntennaArray"]

_TWO_PI_J = 2j * np.pi


class AntennaArray:
    """An antenna array with arbitrary element positions.

    Parameters
    ----------
    x : array_like
        Element positions along the x-axis, normalised to the wavelength.
    y : array_like, optional
        Element positions along the y-axis, normalised to the wavelength.
        A scalar is broadcast to every element.  ``None`` (the default)
        places all elements on the x-axis.

    Attributes
    ----------
    x, y : ndarray of float
        Element positions, always 1-D and of equal length.
    """

    def __init__(self, x: ArrayLike, y: ArrayLike | None = None) -> None:
        self.x, self.y = self._validate_positions(x, y)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_positions(x, y):
        """Coerce and check element positions, returning matched 1-D arrays."""
        x_arr = np.asarray(x, dtype=float).ravel()
        if x_arr.size == 0:
            raise ValueError("x must contain at least one element position")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x must contain only finite values")

        if y is None:
            return x_arr, np.zeros_like(x_arr)

        y_arr = np.asarray(y, dtype=float).ravel()
        if y_arr.size == 1 and x_arr.size != 1:
            y_arr = np.full_like(x_arr, y_arr.item())
        if y_arr.size != x_arr.size:
            raise ValueError(
                f"x and y must have the same length; got {x_arr.size} and {y_arr.size}"
            )
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y must contain only finite values")
        return x_arr, y_arr

    @staticmethod
    def _validate_weight(weight, expected: int, name: str = "weight"):
        """Check a complex weight vector against the expected element count."""
        arr = np.asarray(weight)
        if arr.ndim != 1 or arr.size != expected:
            raise ValueError(
                f"{name} must be a 1-D array of length {expected}; "
                f"got shape {arr.shape}"
            )
        return arr.astype(complex, copy=False)

    @classmethod
    def _validate_taper(cls, taper, expected: int, name: str = "taper"):
        """Return a validated amplitude taper, or uniform weighting if None."""
        if taper is None:
            return np.ones(expected, dtype=complex)
        return cls._validate_weight(taper, expected, name)

    @staticmethod
    def _normalise(weight):
        """Scale weights so that ``sum(|w|) == 1``."""
        total = np.sum(np.abs(weight))
        if total == 0:
            raise ValueError("weights sum to zero and cannot be normalised")
        return weight / total

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def num_elements(self) -> int:
        """Total number of elements in the array."""
        return int(self.x.size)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def steering_weights(
        self,
        beam_az: float = 0.0,
        beam_el: float = 0.0,
        taper: ArrayLike | None = None,
    ) -> NDArray[np.complex128]:
        """Complex element weights steering the main beam to a given direction.

        Parameters
        ----------
        beam_az, beam_el : float, optional
            Steering angles in azimuth and elevation (deg). (default is 0)
        taper : array_like, optional
            Per-element amplitude taper of length ``num_elements``.
            ``None`` (the default) gives uniform illumination.

        Returns
        -------
        ndarray of complex
            Weights normalised so that ``sum(|w|) == 1``.
        """
        taper_arr = self._validate_taper(taper, self.num_elements, "taper")
        phase = np.exp(
            _TWO_PI_J
            * (
                self.x * np.sin(np.radians(beam_az))
                + self.y * np.sin(np.radians(beam_el))
            )
        )
        return self._normalise(phase * taper_arr)

    # ------------------------------------------------------------------
    # Pattern
    # ------------------------------------------------------------------

    def get_pattern(
        self,
        azimuth: ArrayLike,
        elevation: ArrayLike = 0.0,
        *,
        beam_az: float = 0.0,
        beam_el: float = 0.0,
        taper: ArrayLike | None = None,
        weight: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Compute the array factor by direct summation over element positions.

        This works for any array in the package because it depends only on
        ``x``, ``y`` and the element weights.  The angle grids are arbitrary,
        so this is the method to use for non-uniform sampling; the uniform
        subclasses additionally offer FFT-accelerated methods on a regular
        grid.

        Parameters
        ----------
        azimuth : array_like
            Azimuth angles (deg).  A scalar collapses that output axis.
        elevation : array_like, optional
            Elevation angles (deg).  A scalar collapses that output axis,
            so the default returns a 1-D azimuth cut at boresight.
        beam_az, beam_el : float, optional
            Steering angles (deg) used to build the weights. (default is 0)
        taper : array_like, optional
            Per-element amplitude taper of length ``num_elements``.
            Ignored when ``weight`` is given.
        weight : array_like, optional
            Fully specified complex element weights of length
            ``num_elements``.  Overrides ``beam_az``, ``beam_el`` and
            ``taper``, and is used verbatim without renormalisation.

        Returns
        -------
        dict
            'array_factor' : ndarray  - pattern in linear scale, shape
                             ``(len(azimuth), len(elevation))`` before any
                             scalar axis is collapsed
            'x', 'y'       : ndarray  - element positions (wavelengths)
            'weight'       : ndarray  - complex weights that were applied
            'azimuth'      : ndarray  - the azimuth angles (deg)
            'elevation'    : ndarray  - the elevation angles (deg)
        """
        az = np.asarray(azimuth, dtype=float)
        el = np.asarray(elevation, dtype=float)
        az_is_scalar = az.ndim == 0
        el_is_scalar = el.ndim == 0

        if weight is None:
            weights = self.steering_weights(beam_az, beam_el, taper)
        else:
            weights = self._validate_weight(weight, self.num_elements, "weight")

        u = np.sin(np.radians(np.atleast_1d(az)))
        v = np.sin(np.radians(np.atleast_1d(el)))

        # The kernel exp(-j2pi(x*u + y*v)) is separable because u depends only
        # on azimuth and v only on elevation.  Factoring it costs
        # N*(n_az + n_el) exponentials instead of N*n_az*n_el, and needs no
        # large intermediate.
        e_x = np.exp(-_TWO_PI_J * np.outer(self.x, u))  # (N, n_az)
        e_y = np.exp(-_TWO_PI_J * np.outer(self.y, v))  # (N, n_el)
        array_factor = (weights[:, None] * e_x).T @ e_y  # (n_az, n_el)

        if el_is_scalar:
            array_factor = array_factor[:, 0]
        if az_is_scalar:
            array_factor = array_factor[0]

        return {
            "array_factor": array_factor,
            "x": self.x,
            "y": self.y,
            "weight": weights,
            "azimuth": az,
            "elevation": el,
        }
