from antarray import AntennaArray
import numpy as np


def test_antennaarray():
    print('#### Test AntennaArray ####')
    x = np.array([0,  0.5, 1, 1.5,  0, 0.5, 1, 1.5])
    y = np.array([0, 0,  0,  0,  0.5, 0.5, 0.5, 0.5])
    ant_array = AntennaArray(x=x, y=y)

    azimuth = np.arange(-90, 90, 1)
    elevation = np.arange(-90, 90, 1)
    weight = np.array([0.125, 0.125j, -0.125, -0.125j,
                       0.125, 0.125j, -0.125, -0.125j])
    pattern_data = ant_array.get_pattern(
        azimuth=azimuth, elevation=elevation, weight=weight)

    peak_idx = np.unravel_index(np.argmax(
        np.abs(pattern_data['array_factor'])),
        np.shape(pattern_data['array_factor']))

    assert np.max(np.abs(pattern_data['array_factor'])) == 1
    assert azimuth[peak_idx[0]] == 30
    assert elevation[peak_idx[1]] == 0
