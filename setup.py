# Copyright (c) Meta Platforms, Inc. and affiliates.

import setuptools
from Cython.Build import cythonize
import numpy
from setuptools import Extension
# from setuptools import setup
# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mise (efficient mesh extraction)
mise_module = setuptools.Extension(
    "av3d.lib.libmise.mise",
    sources=["av3d/lib/libmise/mise.pyx"],
)

# Gather all extension modules
ext_modules = [
    mise_module,
]

setuptools.setup(
    name="av3d",
    url="https://github.com/facebookresearch/av3d",
    description="AV3D",
    version="1.0.0",
    packages=setuptools.find_packages(),
    ext_modules=cythonize(ext_modules),
    # install_requires=INSTALL_REQUIREMENTS,
)