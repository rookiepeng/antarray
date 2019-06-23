from antarray import RectArray
import numpy as np


def test_rectarray():
    rect_array = RectArray(sizex=16)
    assert np.array_equal(rect_array.x, np.arange(0, 16, 1)*0.5)
