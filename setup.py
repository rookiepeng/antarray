#cython: language_level=3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension('Antenna_Array', ['Antenna_Array.py']),
    Extension('Linear_Array', ['Linear_Array.py']),
]
setup(name='antarray',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)