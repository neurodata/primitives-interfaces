from networkx import Graph
import networkx as nx
import numpy as np
from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

Inputs = container.Dataset
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    #dim = hyperparams.Hyperparameter[None](default=None)
    dim = None

class AdjacencyMatrixConcatenator(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '85bbe5b4-3c36-4fe5-a7de-32fa42d550eb',
        'version': "0.1.0",
        'name': "jhu.adj_concat",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.AdjacencyMatrixConcatenator',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graphs', 'adjacency matrix', 'adjacency', 'graph','graph transformation','transformation'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/adj_concat/adj_concat.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu'
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
        'primitive_family': "DATA_TRANSFORMATION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        graph = inputs['0']
        csv = inputs['1']

        linktypes = np.array(csv['linkType'], dtype = 'int32')
        uniq_linktypes, n_i = np.unique(linktypes, return_counts = True)
        n_linktypes = len(uniq_linktypes)

        sources = np.array(csv['source_nodeID'], dtype = 'int32')
        targets = np.array(csv['target_nodeID'], dtype = 'int32')
        nodes = set(np.concatenate((sources, targets)))
        n_nodes = len(nodes)

        info = np.array(csv['linkExists'], dtype = 'int32')
        n_info = len(info)

        edge_counts = np.zeros(n_linktypes)
        for i in range(n_info):
            temp_link_type = linktypes[i]
            edge_counts[temp_link_type] += info[i]
            
        p_hats = edge_counts / n_i

        graphs = [p_hats[i] * np.ones(shape = (n_nodes, n_nodes)) for i in range(n_linktypes)] # set up a bunch of empty graphs

        for i in range(n_info):
            temp_link_type = int(linktypes[i])
            graphs[temp_link_type][sources[i], targets[i]] = info[i]
            graphs[temp_link_type][targets[i], sources[i]] = info[i]
            
        big_graph = np.zeros(shape = (n_nodes*int(n_linktypes), n_nodes*int(n_linktypes)))

        for i in range(n_linktypes):
            big_graph[i*n_nodes:(i + 1)*n_nodes, i*n_nodes:(i + 1)*n_nodes] = graphs[i]
            
        for i in range(n_linktypes):
            for j in range(i + 1, n_linktypes):
                big_graph[i*n_nodes: (i + 1)*n_nodes, j*n_nodes: (j + 1)*n_nodes] = (graphs[i] + graphs[j])/2
                big_graph[j*n_nodes: (j + 1)*n_nodes, i*n_nodes: (i + 1)*n_nodes] = (graphs[i] + graphs[j])/2

        return base.CallResult(container.List([container.ndarray(big_graph)]))