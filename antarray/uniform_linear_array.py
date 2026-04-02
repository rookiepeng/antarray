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
from .uniform_rectangular_array import UniformRectangularArray


class UniformLinearArray(UniformRectangularArray):
    """
    A class defines basic parameters of a linear array.
    Inheritance of UniformRectangularArray (special case with sizey=1)

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
        # Initialize as 1D rectangular array (1 row)
        UniformRectangularArray.__init__(self, sizex=size, sizey=1, spacingx=spacing, spacingy=0.5)

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

    def get_pattern(self, azimuth=None, nfft=512, beam_az=0, weight=None):
        """
        Calculate the array factor for a linear array

        Parameters
        ----------
        azimuth : 1-D array, optional
            Specific azimuth angles for calculation (deg). If provided, the array
            factor will be interpolated to these angles. Takes priority over nfft.
            (default is None)
        nfft : int, optional
            Number of FFT points for beamforming. Used when azimuth is None.
            (default is 512)
        beam_az : float, optional
            Steering angle in azimuth (deg). (default is 0)
        weight : 1-D array, optional
            Amplitude tapering weights for array elements.
            Array length must equal size. If None, uniform weighting is used.
            (default is None)

        Returns
        -------
        dict(
            'array_factor' : 1-D array
                Array pattern in linear scale
            'weight' : 1-D array
                Complex weights applied to array elements (includes steering + taper)
            'azimuth' : 1-D array
                Azimuth angles (deg) corresponding to array_factor
        )
        """
        # Use inherited UniformRectangularArray.get_pattern_az for FFT method
        if azimuth is not None:
            # Direct array factor calculation (non-FFT method)
            # Apply amplitude weights (default to uniform if not provided)
            if weight is None:
                weight_applied = np.ones(self.size)
            else:
                weight_applied = weight
            
            # Apply beam steering phase
            element_weights = weight_applied * np.exp(
                -1j * 2 * np.pi * self.x * np.sin(np.radians(beam_az))
            )
            
            # Normalize weights
            element_weights = element_weights / np.sum(np.abs(element_weights))
            
            # Calculate array factor using direct summation
            # AF(azimuth) = sum_i weight[i] * exp(j * 2*pi * x[i] * sin(azimuth))
            azimuth_grid, x_grid = np.meshgrid(np.radians(azimuth), self.x)
            steering_matrix = np.exp(1j * 2 * np.pi * x_grid * np.sin(azimuth_grid))
            AF = element_weights @ steering_matrix
            
            return {
                'array_factor': AF,
                'weight': element_weights,
                'azimuth': azimuth,
                'raw_fft': None
            }
        else:
            # Use FFT method for efficiency
            result = self.get_pattern_az(
                nfft=nfft,
                beam_az=beam_az,
                beam_el=0,
                weight_x=weight,
                weight_y=None,
                cut_el=0,
            )
            
            return {
                'array_factor': result['array_factor'],
                'weight': result['weight'],
                'azimuth': result['azimuth'],
                'raw_fft': result['raw_fft'].flatten()  # Flatten to 1D since it's a linear array
            }
