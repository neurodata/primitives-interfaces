import os
import sys
from setuptools import setup
from setuptools.command.install import install
from subprocess import check_output, call
from sys import platform

PACKAGE_NAME = 'jhu_primitives'
MINIMUM_PYTHON_VERSION = 3, 6
VERSION = '2018.7.10'


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

#from setuptools import setup
#from setuptools.command.install import install
#import subprocess
#import os

#class CustomInstallCommand(install):
#    """Custom install setup to help run shell commands (outside shell) before installation"""
#    def run(self):
#        dir_path = os.path.dirname(os.path.realpath(__file__))
#        #template_path = os.path.join(dir_path, 'install_r.sh')
#        #templatejs_path = os.path.join(dir_path, 'install_r.sh')
#        templatejs = subprocess.check_output([
#            './install_r.sh'
 #       ])
 #       install.run(self)



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
              #'jhu_primitives.adj_concat',
              'jhu_primitives.lcc',
              'jhu_primitives.lse',
              #'jhu_primitives.dimselect',
              'jhu_primitives.gclass',
              'jhu_primitives.gclust',
              #'jhu_primitives.nonpar',
              #'jhu_primitives.numclust',
              #'jhu_primitives.ptr',
              'jhu_primitives.oosase',
              'jhu_primitives.sgc',
              'jhu_primitives.sgm',
              'jhu_primitives.utils',
              #'jhu_primitives.vnsgm'
    ],
    entry_points = {
        'd3m.primitives': [
            'data_transformation.adjacency_spectral_embedding.JHU=jhu_primitives.ase:AdjacencySpectralEmbedding',
            #'jhu_primitives.AdjacencyMatrixConcatenator=jhu_primitives.adj_concat:AdjacencyMatrixConcatenator',
            'data_preprocessing.largest_connected_component.JHU=jhu_primitives.lcc:LargestConnectedComponent',
            'data_transformation.laplacian_spectral_embedding.JHU=jhu_primitives.lse:LaplacianSpectralEmbedding',
            'data_transformation.laplacian_spectral_embedding.JHU_out_of_sample=jhu_primitives.ooslse:OutOfSampleLaplacianSpectralEmbedding',
            #'jhu_primitives.DimensionSelection=jhu_primitives.dimselect:DimensionSelection',
            'classification.gaussian_classification.JHU=jhu_primitives.gclass:GaussianClassification',
            'graph_clustering.gaussian_clustering.JHU=jhu_primitives.gclust:GaussianClustering',
            #'jhu_primitives.NonParametricClustering=jhu_primitives.nonpar:NonParametricClustering',
            #'jhu_primitives.NumberOfClusters=jhu_primitives.numclust:NumberOfClusters',
            'data_transformation.adjacency_spectral_embedding.JHU_out_of_sample=jhu_primitives.oosase:OutOfSampleAdjacencySpectralEmbedding',
            #'jhu_primitives.PassToRanks=jhu_primitives.ptr:PassToRanks',
            'vertex_nomination.spectral_graph_clustering.JHU=jhu_primitives.sgc:SpectralGraphClustering',
            'graph_matching.seeded_graph_matching.JHU=jhu_primitives.sgm:SeededGraphMatching',
            #'jhu_primitives.VertexNominationSeededGraphMatching=jhu_primitives.vnsgm:VertexNominationSeededGraphMatching'
            ]
    },
    package_data = {'': ['*.r', '*.R']},
    include_package_data = True,
    install_requires=['typing', 'numpy', 'scipy','networkx',
                      'rpy2', 'sklearn', 'jinja2', 'd3m', 'scipy'],
    url='https://github.com/neurodata/primitives-interfaces',
    keywords = 'd3m_primitive'
)

"""
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
              'jhu_primitives.core',
              'jhu_primitives.monomial'
    ]"""
