from arraybeam import AntennaArray
import numpy as np
import numpy.testing as npt


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_4x2():
    """4-element x, 2-row y → 8-element 2-D array."""
    x = np.array([0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5])
    y = np.array([0, 0,   0, 0,   0.5, 0.5, 0.5, 0.5])
    return AntennaArray(x=x, y=y)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_antennaarray_stores_positions():
    x = np.array([0, 0.5, 1.0])
    y = np.array([0, 0,   0.5])
    arr = AntennaArray(x=x, y=y)
    npt.assert_array_equal(arr.x, x)
    npt.assert_array_equal(arr.y, y)


def test_antennaarray_default_y_is_zero():
    x = np.array([0, 0.5, 1.0])
    arr = AntennaArray(x=x)
    assert arr.y == 0


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_antennaarray_return_key():
    arr = _make_4x2()
    az = np.arange(-10, 11, 1)
    el = np.arange(-10, 11, 1)
    result = arr.get_pattern(azimuth=az, elevation=el)
    assert 'array_factor' in result


def test_antennaarray_output_shape():
    arr = _make_4x2()
    az = np.arange(-45, 46, 5)   # 19 points
    el = np.arange(-30, 31, 5)   # 13 points
    result = arr.get_pattern(azimuth=az, elevation=el)
    assert result['array_factor'].shape == (len(az), len(el))


# ---------------------------------------------------------------------------
# Pattern correctness
# ---------------------------------------------------------------------------

def test_antennaarray_broadside_peak():
    """Uniform normalised weights → peak azimuth at 0°.
    A purely x-axis array (y=0) has a flat response in elevation,
    so only the azimuth peak location is checked."""
    x = np.array([0, 0.5, 1.0, 1.5])
    y = np.zeros(4)
    arr = AntennaArray(x=x, y=y)
    az = np.arange(-90, 91, 1)
    el = np.arange(-90, 91, 1)
    w = np.ones(4, dtype=complex) / 4
    result = arr.get_pattern(azimuth=az, elevation=el, weight=w)
    AF = result['array_factor']
    # Collapse elevation axis to find peak azimuth
    az_profile = np.max(np.abs(AF), axis=1)
    assert az[np.argmax(az_profile)] == 0


def test_antennaarray_normalised_peak_is_one():
    """When sum(|w|) == 1 the maximum array factor magnitude equals 1."""
    arr = _make_4x2()
    az = np.arange(-90, 90, 1)
    el = np.arange(-90, 90, 1)
    w = np.array([0.125, 0.125j, -0.125, -0.125j,
                  0.125, 0.125j, -0.125, -0.125j])
    result = arr.get_pattern(azimuth=az, elevation=el, weight=w)
    assert np.max(np.abs(result['array_factor'])) == 1


def test_antennaarray_steered_peak():
    """Steered weights → peak at az=30, el=0."""
    arr = _make_4x2()
    azimuth = np.arange(-90, 90, 1)
    elevation = np.arange(-90, 90, 1)
    weight = np.array([0.125, 0.125j, -0.125, -0.125j,
                       0.125, 0.125j, -0.125, -0.125j])
    result = arr.get_pattern(azimuth=azimuth, elevation=elevation, weight=weight)
    peak_idx = np.unravel_index(
        np.argmax(np.abs(result['array_factor'])),
        result['array_factor'].shape)
    assert azimuth[peak_idx[0]] == 30
    assert elevation[peak_idx[1]] == 0


def test_antennaarray_default_weight_does_not_raise():
    """Calling get_pattern without weights should not raise."""
    arr = _make_4x2()
    az = np.arange(-10, 11, 5)
    el = np.arange(-10, 11, 5)
    result = arr.get_pattern(azimuth=az, elevation=el)
    assert result['array_factor'].shape == (len(az), len(el))


def test_antennaarray_single_element():
    """Single element array: |AF| == |weight| everywhere."""
    arr = AntennaArray(x=np.array([0.0]), y=np.array([0.0]))
    az = np.array([0.0, 30.0, -45.0])
    el = np.array([0.0, 15.0])
    w = np.array([0.5 + 0j])
    result = arr.get_pattern(azimuth=az, elevation=el, weight=w)
    npt.assert_allclose(np.abs(result['array_factor']), 0.5)
