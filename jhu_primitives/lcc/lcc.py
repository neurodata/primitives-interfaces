from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
import numpy
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
import igraph
import networkx


Inputs = container.ndarray
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    #dim = hyperparams.Hyperparameter[None](default=None)
    dim = None

def file_path_conversion(abs_file_path, uri="file"):
    local_drive, file_path = abs_file_path.split(':')[0], abs_file_path.split(':')[1]
    path_sep = file_path[0]
    file_path = file_path[1:]  # Remove initial separator
    if len(file_path) == 0:
        print("Invalid file path: len(file_path) == 0")
        return

    s = ""
    if path_sep == "/":
        s = file_path
    elif path_sep == "\\":
        splits = file_path.split("\\")
        data_folder = splits[-1]
        for i in splits:
            if i != "":
                s += "/" + i
    else:
        print("Unsupported path separator!")
        return

    if uri == "file":
        return "file://localhost" + s
    else:
        return local_drive + ":" + s

class LargestConnectedComponent(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '32fec24f-6861-4a4c-88f3-d4ec2bc1b486',
        'version': "0.1.0",
        'name': "jhu.lcc",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.LargestConnectedComponent',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['spectral clustering'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/lcc/lcc.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [
            {
            'type': 'UBUNTU',
            'package': 'r-base',
            'version': '3.4.2'
            },
            {
            'type': 'UBUNTU',
            'package': 'libxml2-dev',
            'version': '2.9.4'
            },
            {
            'type': 'UBUNTU',
            'package': 'libpcre3-dev',
            'version': '2.9.4'
            },
#            {
#            'type': 'UBUNTU',
#            'package': 'r-base-dev',
#            'version': '3.4.2'
#            },
#            {
#            'type': 'UBUNTU',
#            'package': 'r-recommended',
#            'version': '3.4.2'
#            },
            {
            'type': 'PIP',
            'package_uri': 'git+https://github.com/neurodata/primitives-interfaces.git@{git_commit}#egg=jhu_primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),),
            },
            {
            'type': 'PIP',
            'package': 'python_igraph',
            'version': '0.7.1'
            },
            {
            'type': 'PIP',
            'package': 'networkx',
            'version': '2.1'
            }
            ],
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris': [
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "GAUSSIAN_PROCESS"
        ],
        'primitive_family': "GRAPH_CLUSTERING"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Input: g: an n x n matrix, n x 2 edge list, a networkx Graph, or igraph Graph 
        Output: The largest connected component of g

        """

        g = inputs

        if type(g) == list:
            g = igraph.Graph(g)

        if type(g) == numpy.ndarray:
            if g.shape[0] == g.shape[1]: # n x n matrix
                g = networkx.Graph(g) # convert to networkx graph to be able to extract edge list 
            elif g.shape[1] == 2: # n x 2 matrix
                g = igraph.Graph(list(g))
            else:
                print("Neither n x n nor n x 2. Please submit a square matrix or edge list.")
                return
                
        if type(g) == networkx.classes.graph.Graph: # networkx graph
            g = igraph.Graph(list(g.edges)) # convert to igraph graph, find the clusters
            
        if type(g) == igraph.Graph: # igraph graph
            components = g.clusters()
            components_len = [len(components[i]) for i in range(len(components))] # find lengths of components (faster way?)
            largest_component = components[numpy.argmax(components_len)]
        else:
            print("Unsupported graph type")
            return

        result = numpy.array(largest_component)

        outputs = container.ndarray(result)

        return base.CallResult(outputs)