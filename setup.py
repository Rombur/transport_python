#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
      numpy_include = numpy.get_include()
except AttributeError:
      numpy_include = numpy.get_numpy_include()

# cg extension module
_cg = Extension("_cg",
                     ["cg_wrap.cxx","cg.cc"],
                     include_dirs = [numpy_include],
                    )

# NumyTypemapTests setup
setup(name = "cg function",
      description = "cg with ssor.",
      author = "Bruno Turcksin",
      version = "1.0",
      ext_modules = [_cg]
     )
