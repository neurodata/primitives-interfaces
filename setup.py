import os
import sys
from setuptools import setup

PACKAGE_NAME = 'jhu_primitives'
MINIMUM_PYTHON_VERSION = 3, 6
VERSION = '0.0.1a'

def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)

check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description='Python interfaces for TA1 primitives',
    author='Disa Mhembere, Eric Bridgeford',
    packages=[
              PACKAGE_NAME,
              'primitive_interfaces',
              'jhu_primitives.ase',
              'jhu_primitives.lse',
              'jhu_primitives.dimselect',
              'jhu_primitives.gclust',
              'jhu_primitives.nonpar',
              'jhu_primitives.numclust',
              'jhu_primitives.oocase',
              'jhu_primitives.ptr',
              'jhu_primitives.sgc',
              'jhu_primitives.sgm',
              'jhu_primitives.vnsgm',
              'jhu_primitives.utils',
              'jhu_primitives.wrapper',
              'jhu_primitives.core'
    ],
    install_requires=['typing', 'numpy', 'scipy',
        'python-igraph', 'rpy2', 'sklearn', 'jinja2'],
    url='https://github.com/neurodata/primitives-interfaces',
)
