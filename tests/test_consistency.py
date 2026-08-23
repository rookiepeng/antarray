"""Cross-checks between the classes, and regression tests for the defects
that the per-class suites could not see.

Each class was internally self-consistent before these tests existed, which
is exactly why a conflicting weight convention between two of them went
unnoticed.  The tests here compare the classes against *each other* and
against known array physics.
"""

import numpy as np
import numpy.testing as npt
import pytest

from arraybeam import AntennaArray, UniformLinearArray, UniformRectangularArray

AZ = np.arange(-90, 90.01, 0.02)


# ---------------------------------------------------------------------------
# FFT path vs direct summation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beam_az", [0, 20, -35])
def test_fft_az_matches_direct_summation(beam_az):
    """get_pattern_az (FFT) and get_pattern (direct) must agree when handed
    the same weights, on the same angles."""
    arr = UniformRectangularArray(sizex=8, sizey=4)
    fft = arr.get_pattern_az(nfft=512, beam_az=beam_az, cut_el=0)
    direct = arr.get_pattern(
        azimuth=fft["azimuth"], elevation=0.0, weight=fft["weight"]
    )
    npt.assert_allclose(direct["array_factor"], fft["array_factor"], atol=1e-12)


@pytest.mark.parametrize("beam_el", [0, 15])
def test_fft_el_matches_direct_summation(beam_el):
    arr = UniformRectangularArray(sizex=4, sizey=8)
    fft = arr.get_pattern_el(nfft=512, beam_el=beam_el, cut_az=0)
    direct = arr.get_pattern(
        azimuth=0.0, elevation=fft["elevation"], weight=fft["weight"]
    )
    npt.assert_allclose(direct["array_factor"], fft["array_factor"], atol=1e-12)


def test_fft_2d_matches_direct_summation():
    arr = UniformRectangularArray(sizex=8, sizey=4)
    fft = arr.get_pattern_2d(nfft_az=64, nfft_el=64, beam_az=10, beam_el=-5)
    direct = arr.get_pattern(
        azimuth=fft["azimuth"], elevation=fft["elevation"], weight=fft["weight"]
    )
    npt.assert_allclose(direct["array_factor"], fft["array_factor"], atol=1e-12)


# ---------------------------------------------------------------------------
# Weight convention  (regression for the ULA/URA conjugate mismatch)
# ---------------------------------------------------------------------------


def _weight_producers(beam_az):
    """Every public route that hands a caller a weight vector."""
    ula = UniformLinearArray(size=16)
    ura = UniformRectangularArray(sizex=16, sizey=1)
    return [
        (
            "ULA.get_pattern",
            ula,
            ula.get_pattern(azimuth=AZ, beam_az=beam_az)["weight"],
        ),
        ("ULA.steering_weights", ula, ula.steering_weights(beam_az=beam_az)),
        ("ULA.get_pattern_az", ula, ula.get_pattern_az(beam_az=beam_az)["weight"]),
        ("URA.get_pattern_az", ura, ura.get_pattern_az(beam_az=beam_az)["weight"]),
        ("URA.get_pattern_2d", ura, ura.get_pattern_2d(beam_az=beam_az)["weight"]),
    ]


@pytest.mark.parametrize("beam_az", [30, -25])
def test_all_weights_share_one_sign_convention(beam_az):
    """Weights from any route, fed back through the base class, must steer
    the beam to +beam_az.

    Before the convention was unified, UniformLinearArray's direct path
    returned the complex conjugate of every other route, so this same call
    peaked at -beam_az.
    """
    for name, arr, weight in _weight_producers(beam_az):
        base = AntennaArray(x=arr.x, y=arr.y)
        pattern = np.abs(
            base.get_pattern(azimuth=AZ, elevation=0.0, weight=weight)["array_factor"]
        )
        peak = AZ[np.argmax(pattern)]
        assert abs(peak - beam_az) < 0.05, (
            f"{name} steered to {peak:+.2f} deg, expected {beam_az:+d}"
        )


def test_all_weight_routes_are_identical():
    """The routes should not merely agree in sign - they should be equal."""
    reference = None
    for name, _, weight in _weight_producers(30):
        if reference is None:
            reference = weight
            continue
        npt.assert_allclose(
            weight, reference, atol=1e-12, err_msg=f"{name} differs from the reference"
        )


def test_weights_are_normalised():
    arr = UniformRectangularArray(sizex=8, sizey=4)
    npt.assert_almost_equal(
        np.sum(np.abs(arr.get_pattern_2d(beam_az=15)["weight"])), 1.0
    )


def test_weight_order_matches_element_order():
    """weight[i] must correspond to (x[i], y[i])."""
    arr = UniformRectangularArray(sizex=4, sizey=3, spacingx=0.5, spacingy=0.5)
    weight = arr.get_pattern_az(beam_az=25, beam_el=10)["weight"]
    npt.assert_allclose(
        weight, arr.steering_weights(beam_az=25, beam_el=10), atol=1e-12
    )


# ---------------------------------------------------------------------------
# UniformLinearArray is exactly the one-row UniformRectangularArray
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing", [0.5, 0.75, 1.0])
def test_linear_array_equals_single_row_rectangular(spacing):
    ula = UniformLinearArray(size=16, spacing=spacing)
    ura = UniformRectangularArray(sizex=16, sizey=1, spacingx=spacing, spacingy=spacing)

    npt.assert_array_equal(ula.x, ura.x)
    npt.assert_array_equal(ula.y, ura.y)

    lin = ula.get_pattern_az(nfft=512, beam_az=25)
    rect = ura.get_pattern_az(nfft=512, beam_az=25)
    npt.assert_allclose(lin["array_factor"], rect["array_factor"], atol=1e-12)
    npt.assert_allclose(lin["weight"], rect["weight"], atol=1e-12)
    npt.assert_allclose(lin["azimuth"], rect["azimuth"], atol=1e-12)


def test_linear_array_get_pattern_is_the_inherited_one():
    """The ULA override was removed; get_pattern must be the base method."""
    assert UniformLinearArray.get_pattern is AntennaArray.get_pattern
    assert UniformRectangularArray.get_pattern is AntennaArray.get_pattern


# ---------------------------------------------------------------------------
# Array physics
# ---------------------------------------------------------------------------


def test_uniform_peak_sidelobe_is_minus_13_dB():
    """A uniformly illuminated linear array has a -13.2 dB peak sidelobe."""
    arr = UniformLinearArray(size=16)
    pattern = np.abs(arr.get_pattern(azimuth=AZ)["array_factor"])
    sidelobes = pattern[np.abs(AZ) > 8]  # first null is at ~7.2 deg
    npt.assert_allclose(20 * np.log10(sidelobes.max()), -13.2, atol=0.2)


def test_taper_lowers_sidelobes():
    arr = UniformLinearArray(size=16)
    far = np.abs(AZ) > 25  # clear of both main lobes

    def peak_sll(taper):
        pattern = np.abs(arr.get_pattern(azimuth=AZ, taper=taper)["array_factor"])
        return 20 * np.log10(pattern[far].max())

    assert peak_sll(np.hanning(16)) < peak_sll(None) - 10


def test_grating_lobe_appears_at_predicted_angle():
    """With d = 1.0 wavelength and the beam at 30 deg, a grating lobe must
    appear at sin(theta) = sin(30) - 1, i.e. -30 deg."""
    arr = UniformLinearArray(size=16, spacing=1.0)
    pattern = np.abs(arr.get_pattern(azimuth=AZ, beam_az=30)["array_factor"])
    peaks = AZ[
        (pattern > 0.5)
        & (pattern >= np.roll(pattern, 1))
        & (pattern >= np.roll(pattern, -1))
    ]
    npt.assert_allclose(np.sort(peaks), [-30, 30], atol=0.05)


def test_no_grating_lobe_at_half_wavelength():
    arr = UniformLinearArray(size=16, spacing=0.5)
    pattern = np.abs(arr.get_pattern(azimuth=AZ, beam_az=30)["array_factor"])
    peaks = AZ[
        (pattern > 0.5)
        & (pattern >= np.roll(pattern, 1))
        & (pattern >= np.roll(pattern, -1))
    ]
    npt.assert_allclose(peaks, [30], atol=0.05)


@pytest.mark.parametrize("size", [16, 32, 64])
def test_beamwidth_matches_analytic_hpbw(size):
    """Broadside half-power beamwidth of a uniformly illuminated aperture
    of length L is approximately 0.886 * lambda / L radians."""
    spacing = 0.5
    arr = UniformLinearArray(size=size, spacing=spacing)
    pattern = np.abs(arr.get_pattern(azimuth=AZ)["array_factor"])
    half_power = AZ[pattern >= 1 / np.sqrt(2)]
    measured = half_power.max() - half_power.min()
    expected = np.degrees(0.886 / (size * spacing))
    npt.assert_allclose(measured, expected, atol=0.1)


# ---------------------------------------------------------------------------
# Scalar-axis collapsing
# ---------------------------------------------------------------------------


def test_scalar_elevation_gives_1d_cut():
    arr = UniformLinearArray(size=8)
    result = arr.get_pattern(azimuth=AZ, elevation=0.0)
    assert result["array_factor"].shape == (AZ.size,)


def test_scalar_azimuth_and_elevation_give_scalar():
    arr = UniformLinearArray(size=8)
    result = arr.get_pattern(azimuth=0.0, elevation=0.0)
    assert np.ndim(result["array_factor"]) == 0
    npt.assert_almost_equal(np.abs(result["array_factor"]), 1.0)


def test_array_axes_give_2d():
    arr = UniformRectangularArray(sizex=4, sizey=4)
    result = arr.get_pattern(
        azimuth=np.arange(-10, 11, 1.0), elevation=np.arange(-5, 6, 1.0)
    )
    assert result["array_factor"].shape == (21, 11)
