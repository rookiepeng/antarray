from arraybeam import UniformLinearArray
import numpy as np
import numpy.testing as npt


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_lineararray_default_positions():
    arr = UniformLinearArray(size=16)
    npt.assert_array_equal(arr.x, np.arange(0, 16) * 0.5)
    npt.assert_array_equal(arr.y, np.zeros(16))


def test_lineararray_custom_spacing():
    arr = UniformLinearArray(size=8, spacing=1.0)
    npt.assert_array_equal(arr.x, np.arange(0, 8) * 1.0)
    npt.assert_array_equal(arr.y, np.zeros(8))


def test_lineararray_stores_size_and_spacing():
    arr = UniformLinearArray(size=10, spacing=0.75)
    assert arr.size == 10
    assert arr.spacing == 0.75


# ---------------------------------------------------------------------------
# update_parameters
# ---------------------------------------------------------------------------

def test_lineararray_update_size():
    arr = UniformLinearArray(size=16)
    arr.update_parameters(size=32, spacing=1)
    npt.assert_array_equal(arr.x, np.arange(0, 32) * 1.0)
    npt.assert_array_equal(arr.y, np.zeros(32))


def test_lineararray_update_spacing_only():
    arr = UniformLinearArray(size=8)
    arr.update_parameters(spacing=0.25)
    npt.assert_array_almost_equal(arr.x, np.arange(0, 8) * 0.25)
    assert arr.size == 8


# ---------------------------------------------------------------------------
# get_pattern  –  direct (azimuth provided)
# ---------------------------------------------------------------------------

def test_lineararray_direct_return_keys():
    arr = UniformLinearArray(size=16)
    result = arr.get_pattern(azimuth=np.arange(-90, 90, 1))
    for key in ('array_factor', 'weight', 'azimuth'):
        assert key in result


def test_lineararray_direct_broadside_peak():
    arr = UniformLinearArray(size=16)
    theta = np.arange(-90, 90, 1)
    result = arr.get_pattern(azimuth=theta, beam_az=0)
    assert np.max(np.abs(result['array_factor'])) == 1
    assert theta[np.argmax(np.abs(result['array_factor']))] == 0


def test_lineararray_direct_positive_steering():
    arr = UniformLinearArray(size=32)
    arr.update_parameters(spacing=0.5)
    theta = np.arange(-90, 90, 1)
    result = arr.get_pattern(azimuth=theta, beam_az=10)
    assert np.max(np.abs(result['array_factor'])) == 1
    assert theta[np.argmax(np.abs(result['array_factor']))] == 10


def test_lineararray_direct_negative_steering():
    arr = UniformLinearArray(size=16)
    theta = np.arange(-90, 90, 1)
    result = arr.get_pattern(azimuth=theta, beam_az=-20)
    assert theta[np.argmax(np.abs(result['array_factor']))] == -20


def test_lineararray_direct_weight_length():
    """Returned weight vector length must equal array size."""
    arr = UniformLinearArray(size=16)
    result = arr.get_pattern(azimuth=np.arange(-90, 90, 1))
    assert len(result['weight']) == 16


def test_lineararray_direct_azimuth_passthrough():
    """Returned azimuth must equal the input azimuth array."""
    arr = UniformLinearArray(size=8)
    theta = np.array([-30.0, 0.0, 30.0])
    result = arr.get_pattern(azimuth=theta)
    npt.assert_array_equal(result['azimuth'], theta)


def test_lineararray_direct_custom_weight():
    """Custom amplitude taper is incorporated (result still normalised)."""
    arr = UniformLinearArray(size=8)
    taper = np.hanning(8)
    theta = np.arange(-90, 90, 1)
    result = arr.get_pattern(azimuth=theta, weight=taper)
    npt.assert_almost_equal(np.max(np.abs(result['array_factor'])), 1.0)


# ---------------------------------------------------------------------------
# get_pattern  –  FFT path (no azimuth)
# ---------------------------------------------------------------------------

def test_lineararray_fft_return_keys():
    arr = UniformLinearArray(size=16)
    result = arr.get_pattern(nfft=512)
    for key in ('array_factor', 'weight', 'azimuth', 'raw_fft'):
        assert key in result


def test_lineararray_fft_broadside_peak():
    arr = UniformLinearArray(size=16)
    result = arr.get_pattern(nfft=1024, beam_az=0)
    peak_az = result['azimuth'][np.argmax(np.abs(result['array_factor']))]
    assert np.abs(peak_az) < 0.5   # within half a degree of 0


def test_lineararray_fft_steered_peak():
    arr = UniformLinearArray(size=16)
    result = arr.get_pattern(nfft=2048, beam_az=30)
    peak_az = result['azimuth'][np.argmax(np.abs(result['array_factor']))]
    assert np.abs(peak_az - 30) < 0.5


def test_lineararray_fft_azimuth_range():
    arr = UniformLinearArray(size=8)
    result = arr.get_pattern(nfft=512)
    assert np.all(result['azimuth'] >= -90)
    assert np.all(result['azimuth'] <= 90)
