import numpy
from Cython.Build import cythonize
from setuptools import setup

# extensions = [
#     Extension(
#         "vertex_cover",
#         sources=["vc_cython.pyx"],
#         include_dirs=[numpy.get_include()],
#         # extra_compile_args=["-O3", "-std=c++20", "-D_hypot=hypot", "-stdlib=libc++"],
#         language="c++",
#     )
# ]
#
setup(
    name="Vertex cover functions",
    ext_modules=cythonize("vc_cython.pyx"),
    include_dirs=[numpy.get_include()],
    py_modules=[],
)
