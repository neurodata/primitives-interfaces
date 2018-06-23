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


Inputs = container.Dataset
Outputs = container.List

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    #dim = hyperparams.Hyperparameter[None](default=None)
    dim = None

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
        'keywords': ['graphs', 'connected', 'largest connected component', 'graph','graph transformation','transformation'],
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
            {
            'type': 'PIP',
            'package_uri': 'git+https://github.com/neurodata/primitives-interfaces.git@{git_commit}#egg=jhu_primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),),
            },
            ],
        'algorithm_types': [
            "NONOVERLAPPING_COMMUNITY_DETECTION"
        ],
        'primitive_family': "GRAPH_CLUSTERING"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Input
            G: an n x n matrix or a networkx Graph 
        Return
            The largest connected component of g

        """

        G = inputs['1']

        #if type(G) == igraph.Graph:
        #    raise TypeError("Networkx graphs or n x n numpy arrays only")

        #if type(G) == numpy.ndarray:
        #    if G.ndim == 2:
        #        if G.shape[0] == G.shape[1]: # n x n matrix
        #            G = networkx.Graph(G)
        #        else:
        #            raise TypeError("Networkx graphs or n x n numpy arrays only") 
                
        if type(G) == networkx.classes.graph.Graph: # networkx graph
            g = igraph.Graph(list(G.edges)) # convert to igraph graph, find the clusters
        else:
            raise TypeError("Networkx graphs only")# or n x n numpy arrays only")
            
        components = g.clusters()
        components_len = [len(components[i]) for i in range(len(components))] # find lengths of components (faster way?)
        largest_component = components[numpy.argmax(components_len)]
        
        G_connected = G.subgraph(largest_component).copy()


        return base.CallResult(container.List([G_connected]))