from arraybeam import UniformRectangularArray
import numpy as np
import numpy.testing as npt


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_rectarray_default_is_linear():
    """Default sizey=1 → flat 1-D element layout."""
    arr = UniformRectangularArray(sizex=16)
    npt.assert_array_equal(arr.x, np.arange(0, 16) * 0.5)
    npt.assert_array_equal(arr.y, np.zeros(16))
    assert arr.sizex == 16
    assert arr.sizey == 1


def test_rectarray_2d_positions():
    """sizex=4, sizey=3 → 12 elements with correct tiled positions."""
    arr = UniformRectangularArray(sizex=4, sizey=3, spacingx=0.5, spacingy=0.5)
    assert len(arr.x) == 12
    assert len(arr.y) == 12
    # x repeats the x_array for each row
    expected_x = np.tile(np.arange(0, 4) * 0.5, 3)
    npt.assert_array_equal(arr.x, expected_x)
    # y repeats each y value sizex times
    expected_y = np.repeat(np.arange(0, 3) * 0.5, 4)
    npt.assert_array_equal(arr.y, expected_y)


def test_rectarray_custom_spacing():
    arr = UniformRectangularArray(sizex=4, sizey=2, spacingx=1.0, spacingy=0.75)
    npt.assert_array_almost_equal(arr.x_array, np.arange(0, 4) * 1.0)
    npt.assert_array_almost_equal(arr.y_array, np.arange(0, 2) * 0.75)


# ---------------------------------------------------------------------------
# update_parameters
# ---------------------------------------------------------------------------

def test_rectarray_update_sizex():
    arr = UniformRectangularArray(sizex=8)
    arr.update_parameters(sizex=4)
    assert arr.sizex == 4
    assert len(arr.x) == 4


def test_rectarray_update_sizex_sizey():
    arr = UniformRectangularArray(sizex=8)
    arr.update_parameters(sizex=4, sizey=2)
    assert arr.sizex == 4
    assert arr.sizey == 2
    assert len(arr.x) == 8  # 4 * 2


def test_rectarray_update_spacing():
    arr = UniformRectangularArray(sizex=4)
    arr.update_parameters(spacingx=1.0)
    npt.assert_array_equal(arr.x, np.arange(0, 4) * 1.0)


# ---------------------------------------------------------------------------
# get_pattern_2d
# ---------------------------------------------------------------------------

def test_rectarray_2d_return_keys():
    arr = UniformRectangularArray(sizex=8)
    result = arr.get_pattern_2d()
    for key in ('array_factor', 'x', 'y', 'weight', 'azimuth', 'elevation'):
        assert key in result


def test_rectarray_2d_output_shape():
    arr = UniformRectangularArray(sizex=8, sizey=4)
    nfft_az, nfft_el = 64, 64
    result = arr.get_pattern_2d(nfft_az=nfft_az, nfft_el=nfft_el)
    az_len = len(result['azimuth'])
    el_len = len(result['elevation'])
    assert result['array_factor'].shape == (az_len, el_len)


def test_rectarray_2d_max_is_one():
    arr = UniformRectangularArray(sizex=16)
    result = arr.get_pattern_2d()
    npt.assert_almost_equal(np.max(np.abs(result['array_factor'])), 1.0)


def test_rectarray_2d_broadside_peak():
    """Peak azimuth and elevation both at 0° when beam_az=beam_el=0."""
    arr = UniformRectangularArray(sizex=16)
    result = arr.get_pattern_2d(nfft_az=256)
    peak_idx = np.unravel_index(
        np.argmax(np.abs(result['array_factor'])),
        result['array_factor'].shape)
    assert result['azimuth'][peak_idx[0]] == 0


def test_rectarray_2d_uniform_weight_normalisation():
    """Uniform weights for a 16-element array → each weight == 1/16."""
    sizex = 16
    arr = UniformRectangularArray(sizex=sizex)
    result = arr.get_pattern_2d(nfft_az=256)
    npt.assert_array_equal(result['weight'], np.ones(sizex, dtype=complex) / sizex)


def test_rectarray_2d_steered_weights():
    """beam_az=30, sizex=4, sizey=2 → weights match expected steering phases."""
    arr = UniformRectangularArray(sizex=4, sizey=2)
    result = arr.get_pattern_2d(beam_az=30)
    npt.assert_almost_equal(result['weight'], np.array(
        [0.125, 0.125j, -0.125, -0.125j, 0.125, 0.125j, -0.125, -0.125j]))


def test_rectarray_2d_azimuth_range():
    arr = UniformRectangularArray(sizex=8)
    result = arr.get_pattern_2d()
    assert np.all(result['azimuth'] >= -90)
    assert np.all(result['azimuth'] <= 90)
    assert np.all(result['elevation'] >= -90)
    assert np.all(result['elevation'] <= 90)


# ---------------------------------------------------------------------------
# get_pattern_az
# ---------------------------------------------------------------------------

def test_rectarray_az_return_keys():
    arr = UniformRectangularArray(sizex=8)
    result = arr.get_pattern_az()
    for key in ('array_factor', 'raw_fft', 'x', 'y', 'weight', 'azimuth', 'elevation'):
        assert key in result


def test_rectarray_az_is_1d():
    arr = UniformRectangularArray(sizex=8)
    result = arr.get_pattern_az(nfft=512)
    assert result['array_factor'].ndim == 1


def test_rectarray_az_broadside_peak():
    arr = UniformRectangularArray(sizex=16)
    result = arr.get_pattern_az(nfft=1024, beam_az=0)
    peak_az = result['azimuth'][np.argmax(np.abs(result['array_factor']))]
    assert np.abs(peak_az) < 0.5


def test_rectarray_az_steered_peak():
    arr = UniformRectangularArray(sizex=16)
    result = arr.get_pattern_az(nfft=2048, beam_az=30)
    peak_az = result['azimuth'][np.argmax(np.abs(result['array_factor']))]
    assert np.abs(peak_az - 30) < 0.5


def test_rectarray_az_max_is_one():
    arr = UniformRectangularArray(sizex=16)
    result = arr.get_pattern_az(nfft=512)
    npt.assert_almost_equal(np.max(np.abs(result['array_factor'])), 1.0)


def test_rectarray_az_cut_el_fixed():
    """'elevation' key in az-cut result should be a scalar."""
    arr = UniformRectangularArray(sizex=8)
    result = arr.get_pattern_az(beam_az=0, cut_el=10)
    assert np.ndim(result['elevation']) == 0
    assert float(result['elevation']) == 10


def test_rectarray_az_weight_taper():
    """Custom weight_x taper should change the weight vector."""
    arr = UniformRectangularArray(sizex=8)
    uniform_result = arr.get_pattern_az()
    taper = np.hanning(8)
    tapered_result = arr.get_pattern_az(weight_x=taper)
    assert not np.allclose(uniform_result['weight'], tapered_result['weight'])


# ---------------------------------------------------------------------------
# get_pattern_el
# ---------------------------------------------------------------------------

def test_rectarray_el_return_keys():
    arr = UniformRectangularArray(sizex=4, sizey=4)
    result = arr.get_pattern_el()
    for key in ('array_factor', 'raw_fft', 'x', 'y', 'weight', 'azimuth', 'elevation'):
        assert key in result


def test_rectarray_el_is_1d():
    arr = UniformRectangularArray(sizex=4, sizey=4)
    result = arr.get_pattern_el(nfft=512)
    assert result['array_factor'].ndim == 1


def test_rectarray_el_broadside_peak():
    arr = UniformRectangularArray(sizex=4, sizey=16)
    result = arr.get_pattern_el(nfft=1024, beam_el=0)
    peak_el = result['elevation'][np.argmax(np.abs(result['array_factor']))]
    assert np.abs(peak_el) < 0.5


def test_rectarray_el_steered_peak():
    arr = UniformRectangularArray(sizex=4, sizey=16)
    result = arr.get_pattern_el(nfft=2048, beam_el=20)
    peak_el = result['elevation'][np.argmax(np.abs(result['array_factor']))]
    assert np.abs(peak_el - 20) < 0.5


def test_rectarray_el_cut_az_fixed():
    """'azimuth' key in el-cut result should be a scalar."""
    arr = UniformRectangularArray(sizex=4, sizey=8)
    result = arr.get_pattern_el(beam_az=0, cut_az=15)
    assert np.ndim(result['azimuth']) == 0
    assert float(result['azimuth']) == 15
