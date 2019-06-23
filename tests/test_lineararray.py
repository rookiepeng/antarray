from antarray import LinearArray
import numpy as np


def test_lineararray():
    print('#### Test LinearArray ####')
    lin_array = LinearArray(size=16)
    assert np.array_equal(lin_array.x, np.arange(0, 16, 1)*0.5)
    assert np.array_equal(lin_array.y, np.zeros(16))

    lin_array.update_parameters(size=32, spacing=1)
    assert np.array_equal(lin_array.x, np.arange(0, 32, 1)*1)
    assert np.array_equal(lin_array.y, np.zeros(32))

    theta = np.arange(-90, 90, 1)
    lin_array.update_parameters(spacing=0.5)
    pattern_data = lin_array.get_pattern(theta=theta, beam_loc=10)
    assert np.max(np.abs(pattern_data['array_factor'])) == 1
    assert theta[np.argmax(np.abs(pattern_data['array_factor']))] == 10
