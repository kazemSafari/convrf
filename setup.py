#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='convrf',
    version='0.0.0',
    description='Implementation of Directional Receptive Field Convolutional Layer',
    author='Kazem Safari',
    author_email='mksafari@math.uh.edu',
    url='https://github.com/kazemSafari/DRF',
    packages=find_packages(exclude=[]),
    install_requires=['torch', 'numpy', 'pathlib', 'argparse', 'scipy']
)
