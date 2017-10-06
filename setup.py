import os
import sys
from setuptools import setup

PACKAGE_NAME = 'jhu_primitives'
MINIMUM_PYTHON_VERSION = 3, 6
VERSION = '0.0.4'

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
    long_description='A library wrapping JHU\'s Python interfaces for the D3M program\'s TA1 primitives.',
    author='Disa Mhembere, Eric Bridgeford, Youngser Park, Heather G. Patsolic',
    author_email="disa@jhu.edu",
    packages=[
              PACKAGE_NAME,
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
    entry_points = {
        'd3m.primtiives': [
            'jhu_primitives.AdjacencySpectralEmbedding=jhu_primitives.ase:AdjacencySpectralEmbedding',
            'jhu_primitives.LaplacianSpectralEmbedding=jhu_primitives.lse:LaplacianSpectralEmbedding',
            'jhu_primitives.DimensionSelection=jhu_primitives.dimselect:DimensionSelection',
            'jhu_primitives.GaussianClustering=jhu_primitives.gclust:GaussianClustering',
            'jhu_primitives.NonParametricClusteirng=jhu_primitives.nonpar:NonParametricClustering',
            'jhu_primitives.NumberOfClusters=jhu_primitives.numclust:NumberOfClusters',
            'jhu_primitives.OutOfCoreAdjacencySpectralEmbedding=jhu_primitives.oocase:OutOfCoreAdjacencySpectralEmbedding',
            'jhu_primitives.PassToRanks=jhu_primitives.ptr:PassToRanks',
            'jhu_primitives.SpectralGraphClustering=jhu_primitives.sgc:SpectralGraphClustering',
            'jhu_primitives.SeededGraphMatching=jhu_primitives.sgm:SeededGraphMatching',
            'jhu_primitives.VertexNominationSeededGraphMatching=jhu_primitives.vnsgm:VertexNominationSeededGraphMatching'
            ]
    },
    package_data = {'': ['*.r', '*.R']},
    include_package_data = True,
    install_requires=['typing', 'numpy', 'scipy',
        'python-igraph', 'rpy2', 'sklearn', 'jinja2', 'primitive_interfaces'],
    url='https://github.com/neurodata/primitives-interfaces',
)
