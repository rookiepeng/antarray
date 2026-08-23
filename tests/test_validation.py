"""Input validation.

Regression tests for parameters that used to be accepted silently and
produce NaNs or empty arrays instead of raising.
"""

import numpy as np
import pytest

from arraybeam import AntennaArray, UniformLinearArray, UniformRectangularArray

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, -4])
def test_size_below_one_raises(size):
    """UniformLinearArray(size=0) used to build an empty array and return
    an all-NaN pattern."""
    with pytest.raises(ValueError, match=">= 1"):
        UniformLinearArray(size=size)


@pytest.mark.parametrize("spacing", [0, -0.5])
def test_non_positive_spacing_raises(spacing):
    """spacing=0 used to divide by zero inside _k_axis and yield NaNs."""
    with pytest.raises(ValueError, match="finite positive"):
        UniformLinearArray(size=8, spacing=spacing)


def test_non_finite_spacing_raises():
    with pytest.raises(ValueError, match="finite positive"):
        UniformRectangularArray(sizex=8, spacingx=np.inf)


@pytest.mark.parametrize("size", [4.5, "8", None])
def test_non_integer_size_raises(size):
    with pytest.raises(TypeError):
        UniformRectangularArray(sizex=size)


def test_integral_float_size_is_accepted():
    assert UniformRectangularArray(sizex=8.0).sizex == 8


def test_empty_positions_raise():
    with pytest.raises(ValueError, match="at least one"):
        AntennaArray(x=np.array([]))


def test_mismatched_position_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        AntennaArray(x=np.zeros(4), y=np.zeros(3))


def test_non_finite_positions_raise():
    with pytest.raises(ValueError, match="finite"):
        AntennaArray(x=np.array([0.0, np.nan, 1.0]))


# ---------------------------------------------------------------------------
# Weights and tapers
# ---------------------------------------------------------------------------


def test_wrong_length_weight_names_both_lengths():
    arr = UniformLinearArray(size=8)
    with pytest.raises(ValueError, match="length 8"):
        arr.get_pattern(azimuth=np.array([0.0]), weight=np.ones(7))


def test_wrong_length_taper_raises():
    arr = UniformLinearArray(size=8)
    with pytest.raises(ValueError, match="taper"):
        arr.get_pattern(azimuth=np.array([0.0]), taper=np.ones(7))


def test_wrong_length_weight_x_names_the_parameter():
    arr = UniformRectangularArray(sizex=8, sizey=2)
    with pytest.raises(ValueError, match="weight_x"):
        arr.get_pattern_az(weight_x=np.ones(7))


def test_wrong_length_weight_y_names_the_parameter():
    arr = UniformRectangularArray(sizex=8, sizey=2)
    with pytest.raises(ValueError, match="weight_y"):
        arr.get_pattern_az(weight_y=np.ones(7))


def test_all_zero_weights_raise():
    arr = UniformLinearArray(size=8)
    with pytest.raises(ValueError, match="sum to zero"):
        arr.steering_weights(taper=np.zeros(8))


# ---------------------------------------------------------------------------
# update_parameters
# ---------------------------------------------------------------------------


def test_unknown_parameter_raises_on_rect():
    """Typos used to be silently discarded, leaving the array unchanged."""
    arr = UniformRectangularArray(sizex=8)
    with pytest.raises(TypeError, match="spacings"):
        arr.update_parameters(spacings=1.0)


def test_unknown_parameter_raises_on_linear():
    arr = UniformLinearArray(size=8)
    with pytest.raises(TypeError, match="sizex"):
        arr.update_parameters(sizex=4)


def test_update_parameters_validates_new_values():
    arr = UniformLinearArray(size=8)
    with pytest.raises(ValueError, match="finite positive"):
        arr.update_parameters(spacing=0)
    # the array must be left usable after a rejected update
    assert arr.size == 8
    assert arr.spacing == 0.5


def test_linear_update_keeps_row_spacing_consistent():
    arr = UniformLinearArray(size=8)
    arr.update_parameters(spacing=1.25)
    assert arr.spacingx == 1.25
    assert arr.spacingy == 1.25
    assert arr.sizey == 1


def test_non_finite_y_positions_raise():
    with pytest.raises(ValueError, match="y must contain only finite"):
        AntennaArray(x=np.zeros(3), y=np.array([0.0, np.inf, 1.0]))


def test_bool_size_raises():
    """bool is a subclass of int, so it needs rejecting explicitly."""
    with pytest.raises(TypeError, match="must be an integer"):
        UniformRectangularArray(sizex=True)


def test_non_numeric_spacing_raises():
    with pytest.raises(TypeError, match="must be a number"):
        UniformRectangularArray(sizex=8, spacingx="half")
