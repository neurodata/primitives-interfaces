import os
import sys
from setuptools import setup
from setuptools.command.install import install
from subprocess import check_output, call
from sys import platform

PACKAGE_NAME = 'jhu_primitives'
MINIMUM_PYTHON_VERSION = 3, 6
VERSION = '2020.1.9'

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
    author='Hayden S. Helm, Joshua Agterberg, Bijan Varjavand, Disa Mhembere, Eric Bridgeford, Youngser Park, Heather G. Patsolic, Tyler M. Tomita, Jesse L. Patsolic',
    author_email="hhelm2@jhu.edu",
    packages=[
              PACKAGE_NAME,
              'jhu_primitives.ase',
              'jhu_primitives.gclass',
              'jhu_primitives.gclust',
              'jhu_primitives.load_graphs',
              'jhu_primitives.lcc',
              'jhu_primitives.lse',
              'jhu_primitives.link_pred_graph_reader',
              'jhu_primitives.link_pred_rc',
              # 'jhu_primitives.oosase',
              # 'jhu_primitives.ooslse',
              'jhu_primitives.sgc',
              'jhu_primitives.sgm',
              # 'jhu_primitives.sgvn',
              'jhu_primitives.utils',
    ],
    entry_points = {
        'd3m.primitives': [
            'data_transformation.adjacency_spectral_embedding.JHU=jhu_primitives.ase:AdjacencySpectralEmbedding',
            'data_transformation.load_graphs.JHU=jhu_primitives.load_graphs:LoadGraphs',
            'data_preprocessing.largest_connected_component.JHU=jhu_primitives.lcc:LargestConnectedComponent',
            'classification.gaussian_classification.JHU=jhu_primitives.gclass:GaussianClassification',
            'graph_clustering.gaussian_clustering.JHU=jhu_primitives.gclust:GaussianClustering',
            'data_transformation.laplacian_spectral_embedding.JHU=jhu_primitives.lse:LaplacianSpectralEmbedding',
            'link_prediction.data_conversion.JHU=jhu_primitives.link_pred_graph_reader:LinkPredictionGraphReader',
            'link_prediction.rank_classification.JHU=jhu_primitives.link_pred_rc:LinkPredictionRankClassifier',
            # 'data_transformation.out_of_sample_adjacency_spectral_embedding.JHU=jhu_primitives.oosase:OutOfSampleAdjacencySpectralEmbedding',
            # 'data_transformation.out_of_sample_laplacian_spectral_embedding.JHU=jhu_primitives.ooslse:OutOfSampleLaplacianSpectralEmbedding',
            'vertex_nomination.spectral_vertex_nomination.JHU=jhu_primitives.sgvn:SingleGraphVertexNomination',
            'vertex_nomination.spectral_graph_clustering.JHU=jhu_primitives.sgc:SpectralGraphClustering',
            'graph_matching.seeded_graph_matching.JHU=jhu_primitives.sgm:SeededGraphMatching'
            ]
    },
    # package_data = {'': ['*.r', '*.R']},
    # include_package_data = True,
    install_requires=['d3m', # jhu dependency
                      'typing', # jhu dependency
                      'scipy', # jhu dependency
                      # 'networkx', # jhu dependency
                      'numpy', # ==1.15.4', # jhu dependency'
                      # 'sklearn', # jhu dependency
                      'jinja2', # jhu dependency
                      'scipy', # jhu dependency
                      # 'lap',  # unnecessary jhu dependency
                      'cython', # jhu dependency,
                      'lapjv==1.2.0',
                      'graspy>=0.0.2',


                      # Begin d3m dependency
                      # 'pytypes==1.0b5', # d3m dependency
                      # 'frozendict==1.2', # d3m dependency
                      # 'numpy==1.15.4', # d3m dependency
                      # 'jsonschema==2.6.0', # d3m dependency
                      # 'requests==2.19.1', # d3m dependency
                      # 'strict-rfc3339==0.7', # d3m dependency
                      # 'rfc3987==1.3.8', # d3m dependency
                      # 'webcolors==1.8.1', # d3m dependency
                      # 'dateparser==0.7.0', # d3m dependency
                      # 'pandas==0.23.4', # d3m dependency
                      'networkx' # ==2.2', # d3m dependency
                      # 'typing-inspect==0.3.1', # d3m dependency
                      # 'GitPython==2.1.11', # d3m dependency
                      # 'jsonpath-ng==1.4.3', # d3m dependency
                      # 'custom-inherit==2.2.0', # d3m dependency
                      # 'PyYAML==3.13', # d3m dependency
                      # 'pycurl==7.43.0.2', # d3m dependency
                      # 'pyarrow==0.11.1', # d3m dependency
                      # 'gputil==1.3.0', # d3m dependency
                     ],
    url='https://github.com/neurodata/primitives-interfaces',
    dependency_links=['git+https://github.com/neurodata/graspy.git#egg=master'],
    keywords = 'd3m_primitive'
)
