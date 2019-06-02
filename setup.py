#!python
#cython: language_level=3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension('antennaarray', ['antennaarray.py']),
    Extension('lineararray', ['lineararray.py']),
    Extension('rectarray', ['rectarray.py']),
]
setup(name='antarray',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
