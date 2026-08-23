# arraybeam

[![CI](https://github.com/rookiepeng/arraybeam/actions/workflows/ci.yml/badge.svg)](https://github.com/rookiepeng/arraybeam/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Antenna array analysis in Python.

## Features

- Arbitrary array geometries with direct array-factor summation
- Uniform linear array analysis
- Uniform rectangular (planar) array analysis
- FFT-accelerated 2-D patterns and azimuth/elevation cuts
- Beam steering and arbitrary amplitude tapering

## Installation

```bash
pip install arraybeam
```

Or, from a clone of this repository:

```bash
pip install -e ".[dev]"
```

Requires Python 3.9+ and numpy.

## Quick start

```python
import numpy as np
from arraybeam import UniformLinearArray

array = UniformLinearArray(size=16, spacing=0.5)

# Pattern on an explicit angle grid, beam steered to 30 degrees
azimuth = np.arange(-90, 90, 0.1)
result = array.get_pattern(azimuth, beam_az=30)

pattern_db = 20 * np.log10(np.abs(result['array_factor']))
```

Amplitude tapering uses any array you like — nothing is built in:

```python
result = array.get_pattern(azimuth, beam_az=30, taper=np.hanning(16))
```

A rectangular array, with the full 2-D pattern from the FFT path:

```python
from arraybeam import UniformRectangularArray

array = UniformRectangularArray(sizex=32, sizey=8, spacingx=0.5, spacingy=0.5)
result = array.get_pattern_2d(nfft_az=256, nfft_el=256, beam_az=40, beam_el=10)

pattern = result['array_factor']      # 2-D, indexed [azimuth, elevation]
azimuth = result['azimuth']           # degrees
elevation = result['elevation']       # degrees
```

And an arbitrary geometry:

```python
from arraybeam import AntennaArray

array = AntennaArray(x=np.random.uniform(0, 8, 64),
                     y=np.random.uniform(0, 8, 64))
result = array.get_pattern(azimuth=np.arange(-90, 90, 0.5),
                           elevation=np.arange(-90, 90, 0.5),
                           beam_az=15)
```

## API

| Class | Constructor | Pattern methods |
| --- | --- | --- |
| `AntennaArray` | `(x, y=None)` | `get_pattern`, `steering_weights` |
| `UniformRectangularArray` | `(sizex, sizey=1, spacingx=0.5, spacingy=0.5)` | the above, plus `get_pattern_2d`, `get_pattern_az`, `get_pattern_el` |
| `UniformLinearArray` | `(size, spacing=0.5)` | inherits everything from `UniformRectangularArray` |

`get_pattern` samples an arbitrary angle grid by direct summation and is
available on every class with the same signature. The `get_pattern_*`
methods are FFT-accelerated and sample the regular grid that the FFT
produces, so their angle axes are uniform in sine space rather than in
degrees.

Every pattern method returns a dict with `array_factor`, `x`, `y`,
`weight`, `azimuth` and `elevation`; the cut methods add `raw_fft`.

`UniformLinearArray` is the single-row special case of
`UniformRectangularArray` and carries no pattern maths of its own, so a
linear array and the equivalent 1-row rectangular array return bit-identical
results.

## Conventions

Positions and spacings are normalised to the wavelength; all angles are in
degrees.

Patterns are computed in **separable sine space** (`u = sin(azimuth)`,
`v = sin(elevation)`), the usual radar convention for a planar array in the
x-y plane. This is *not* the true spherical direction cosine
(`u = cos(elevation) * sin(azimuth)`): azimuth and elevation are treated as
independent cuts, and the region `u² + v² > 1` is not physically visible
even though the transforms still produce values there.

A single sign convention holds across the whole package:

```
AF(u, v) = Σᵢ wᵢ · exp(-j2π(xᵢu + yᵢv))

wᵢ(beam) = exp(+j2π(xᵢ·sin(beam_az) + yᵢ·sin(beam_el))) · taperᵢ
```

Weights are normalised so that `Σ|w| = 1`. Because every class shares this
convention, a `weight` vector returned by any method can be fed back into
any other method's `weight=` argument.

## Examples

- [Linear array](examples/linear-array.ipynb)
- [Rectangular array](examples/rectangular-planar-array.ipynb)
- [Arbitrary array](examples/arbitrary-array.ipynb)

Running them needs the plotting extras:

```bash
pip install -e ".[examples]"
```

## Development

```bash
pytest --cov=arraybeam
```

```bash
ruff check arraybeam tests
```

## License

MIT — see [LICENSE](LICENSE).
