from antarray import LinearArray
import numpy as np


def test_lineararray():
    lin_array = LinearArray(size=16)
    assert np.array_equal(lin_array.x, np.arange(0, 16, 1)*0.5)
    assert np.array_equal(lin_array.y, np.zeros(16))

    lin_array = LinearArray(size=32, spacing=1)
    assert np.array_equal(lin_array.x, np.arange(0, 32, 1)*1)
    assert np.array_equal(lin_array.y, np.zeros(32))
