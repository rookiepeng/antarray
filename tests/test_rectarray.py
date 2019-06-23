from antarray import RectArray
import numpy as np
import numpy.testing as npt


def test_rectarray():
    sizex = 16
    rect_array = RectArray(sizex=sizex)
    assert np.array_equal(rect_array.x, np.arange(0, sizex, 1)*0.5)
    assert np.array_equal(rect_array.y, np.zeros(sizex))

    nfft_az = 256
    pattern_data = rect_array.get_pattern(nfft_az=nfft_az)
    peak_idx = np.unravel_index(np.argmax(
        np.abs(pattern_data['array_factor'])),
        np.shape(pattern_data['array_factor']))

    assert np.max(np.abs(pattern_data['array_factor'])) == 1
    assert pattern_data['azimuth'][peak_idx[0]] == 0
    assert np.array_equal(pattern_data['weight'], np.ones(
        sizex, dtype=complex)/sizex)

    rect_array.update_parameters(sizex=4, sizey=2)
    pattern_data = rect_array.get_pattern(beam_az=30)
    npt.assert_almost_equal(pattern_data['weight'], np.array(
        [0.125, 0.125j, -0.125, -0.125j, 0.125, 0.125j, -0.125, -0.125j]))
