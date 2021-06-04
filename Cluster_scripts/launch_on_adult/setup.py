from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    name = '_CART',
    ext_modules=[
        Extension('_CART',
                  sources=['_CART.pyx'],
                  extra_compile_args=['-O3'],
                  language='c++')
        ],
    include_dirs=[np.get_include()],
    cmdclass = {'build_ext': build_ext}
)
