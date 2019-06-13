#!python
#cython: language_level=3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension('antennaarray', ['./antarray/antennaarray.py']),
    Extension('lineararray', ['./antarray/lineararray.py']),
    Extension('rectarray', ['./antarray/rectarray.py']),
]
setup(name='antarray',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
