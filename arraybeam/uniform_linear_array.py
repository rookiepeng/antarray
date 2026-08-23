"""
Uniform linear array.

A linear array is the degenerate rectangular array with a single row, so
this class carries no pattern maths of its own: it is a convenience
constructor plus ``size``/``spacing`` aliases over
:class:`~arraybeam.uniform_rectangular_array.UniformRectangularArray`.
Keeping a single implementation of the array factor is what guarantees
that a linear array and the equivalent 1-row rectangular array return
identical patterns and identical weights.

Copyright (C) 2018-2026 Zhengyu Peng <zpeng.me@gmail.com>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .uniform_rectangular_array import UniformRectangularArray

__all__ = ["UniformLinearArray"]


class UniformLinearArray(UniformRectangularArray):
    """A uniformly spaced linear array along the x-axis.

    Parameters
    ----------
    size : int
        Number of elements in the array.
    spacing : float, optional
        Element spacing, normalised to the wavelength. (default is 0.5)

    Attributes
    ----------
    size : int
        Number of elements (read-only alias of ``sizex``).
    spacing : float
        Element spacing (read-only alias of ``spacingx``).

    Notes
    -----
    Patterns come from the inherited methods.  Use
    :meth:`~arraybeam.antenna_array.AntennaArray.get_pattern` for an
    arbitrary azimuth grid::

        ula.get_pattern(azimuth, beam_az=30, taper=np.hanning(ula.size))

    or :meth:`UniformRectangularArray.get_pattern_az
    <arraybeam.uniform_rectangular_array.UniformRectangularArray.get_pattern_az>`
    for the FFT-sampled grid::

        ula.get_pattern_az(nfft=1024, beam_az=30, weight_x=np.hanning(ula.size))
    """

    _PARAMETERS = ("size", "spacing")
    _ALIASES = {"size": "sizex", "spacing": "spacingx"}

    def __init__(self, size: int, spacing: float = 0.5) -> None:
        super().__init__(sizex=size, sizey=1, spacingx=spacing, spacingy=spacing)

    @property
    def size(self) -> int:
        """Number of elements in the array."""
        return self.sizex

    @property
    def spacing(self) -> float:
        """Element spacing, normalised to the wavelength."""
        return self.spacingx

    def update_parameters(self, **kwargs) -> None:
        """Update the array size and/or spacing in place.

        Parameters
        ----------
        size : int, optional
            New number of elements.
        spacing : float, optional
            New element spacing, normalised to the wavelength.

        Raises
        ------
        TypeError
            If an unrecognised parameter name is given.
        """
        unknown = set(kwargs) - set(self._PARAMETERS)
        if unknown:
            raise TypeError(
                f"unknown parameter(s): {', '.join(sorted(unknown))}; "
                f"expected any of {', '.join(self._PARAMETERS)}"
            )
        mapped = {self._ALIASES[k]: v for k, v in kwargs.items()}
        # The single row must keep spacingy == spacingx so that the array
        # stays the exact degenerate case of the rectangular array.
        if "spacingx" in mapped:
            mapped["spacingy"] = mapped["spacingx"]
        super().update_parameters(**mapped)
