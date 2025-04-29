import numpy

# from Cython.Build import cythonize
# from setuptools import setup
#
# # extensions = [
# #     Extension(
# #         "vertex_cover",
# #         sources=["vc_cython.pyx"],
# #         include_dirs=[numpy.get_include()],
# #         # extra_compile_args=["-O3", "-std=c++20", "-D_hypot=hypot", "-stdlib=libc++"],
# #         language="c++",
# #     )
# # ]
# #
# setup(
#     name="Vertex cover functions",
#     ext_modules=cythonize("vc_cython.pyx"),
#     include_dirs=[numpy.get_include()],
#     py_modules=[],
# )
from Cython.Build import cythonize
from setuptools import Extension, setup

# Ensure C++11 or later is used for std::hash, std::unordered_set etc.
cpp_args = ["-std=c++11"]

extensions = [
    Extension(
        "vc_cython",  # Name of the resulting Python module
        ["vc_cython.pyx"],  # List of Cython source files
        language="c++",  # Specify C++ mode
        extra_compile_args=cpp_args,
        extra_link_args=cpp_args,  # Linker might also need the C++ standard flag
    )
]

setup(
    name="vc_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},  # Use Python 3 syntax
    ),
    py_modules=[],
    include_dirs=[numpy.get_include()],
)
