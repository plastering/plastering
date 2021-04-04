"""Set up cython accelerator
"""
import platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


OPENMP_FLAG = '-openmp' if platform.system() == 'Windows'\
              else '-fopenmp'


EXT_MODULES = [
    Extension(
        "colocation.core.corr_score.c_score_func",
        ["colocation/core/corr_score/c_score_func.pyx"],
        extra_compile_args=[OPENMP_FLAG],
        extra_link_args=[OPENMP_FLAG],
    )
]

setup(
    name="colocation",
    ext_modules=cythonize(EXT_MODULES),  # accepts a glob pattern
)
