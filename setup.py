import os
import sys
from setuptools import setup
from setuptools.command.install import install
from subprocess import check_output, call
from sys import platform

PACKAGE_NAME = 'jhu_primitives'
MINIMUM_PYTHON_VERSION = 3, 6
VERSION = '2019.1.21'

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
 #   cmdclass={'install': CustomInstallCommand},
    name=PACKAGE_NAME,
    version=VERSION,
    description='JHU Python interfaces for TA1 primitives for d3m',
    long_description='A library wrapping JHU\'s Python interfaces for the D3M program\'s TA1 primitives.',
    author='Disa Mhembere, Eric Bridgeford, Youngser Park, Heather G. Patsolic, Tyler M. Tomita, Jesse L. Patsolic, Hayden S. Helm, Joshua Agterberg, Bijan Varjavand',
    author_email="disa@jhu.edu",
    packages=[
              PACKAGE_NAME,
              'jhu_primitives.ase',
              'jhu_primitives.lcc',
              'jhu_primitives.lse',
              'jhu_primitives.gclass',
              'jhu_primitives.gclust',
              # 'jhu_primitives.oosase',
              # 'jhu_primitives.ooslse',
              'jhu_primitives.sgc',
              'jhu_primitives.sgm',
              'jhu_primitives.utils',
    ],
    entry_points = {
        'd3m.primitives': [
            'data_transformation.adjacency_spectral_embedding.JHU=jhu_primitives.ase:AdjacencySpectralEmbedding',
            'data_preprocessing.largest_connected_component.JHU=jhu_primitives.lcc:LargestConnectedComponent',
            'data_transformation.laplacian_spectral_embedding.JHU=jhu_primitives.lse:LaplacianSpectralEmbedding',
            'classification.gaussian_classification.JHU=jhu_primitives.gclass:GaussianClassification',
            'graph_clustering.gaussian_clustering.JHU=jhu_primitives.gclust:GaussianClustering',
            # 'data_transformation.out_of_sample_adjacency_spectral_embedding.JHU=jhu_primitives.oosase:OutOfSampleAdjacencySpectralEmbedding',
            # 'data_transformation.out_of_sample_laplacian_spectral_embedding.JHU=jhu_primitives.ooslse:OutOfSampleLaplacianSpectralEmbedding',
            'vertex_nomination.spectral_graph_clustering.JHU=jhu_primitives.sgc:SpectralGraphClustering',
            'graph_matching.seeded_graph_matching.JHU=jhu_primitives.sgm:SeededGraphMatching'
            ]
    },
    # package_data = {'': ['*.r', '*.R']},
    # include_package_data = True,
    install_requires=['typing', 'numpy', 'scipy','networkx',
                      'sklearn', 'jinja2', 'd3m', 'scipy',
                      'git+git://github.com/gatagat/lap.git',
                      'git+git://github.com/src-d/lapjv'],
    url='https://github.com/neurodata/primitives-interfaces',
    keywords = 'd3m_primitive'
)
