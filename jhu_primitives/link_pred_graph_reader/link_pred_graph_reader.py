from networkx import Graph
import networkx as nx
import numpy as np
from typing import Sequence, TypeVar, Union, Dict
import os

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

Inputs = container.Dataset
Outputs = container.List

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    #dim = hyperparams.Hyperparameter[None](default=None)
    dim = None
    # Add hyperparameter that controls how missing values are imputed

class LinkPredictionGraphReader(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '09f2eea8-667c-44b8-a955-6a153ba9ccc3',
        'version': "0.1.0",
        'name': "jhu.link_pred_graph_reader",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.LinkPredictionGraphReader',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'graph reader'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/link_pred_graph_reader/link_pred_graph_reader.py',
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
        # read in graph
        graph = inputs['0']
        csv = inputs['1']

        # make everything arrays, count things
        linktypes = np.array(csv['linkType'], dtype = 'int32')
        uniq_linktypes, n_i = np.unique(linktypes, return_counts = True)
        n_linktypes = len(uniq_linktypes)

        sources = np.array(csv['source_nodeID'], dtype = 'int32')
        targets = np.array(csv['target_nodeID'], dtype = 'int32')
        nodes = set(np.concatenate((sources, targets)))
        n_nodes = len(nodes)

        info = np.array(csv['linkExists'], dtype = 'int32')
        n_info = len(info)

        # find p_hats
        edge_counts = np.zeros(n_linktypes)

        info_triples = [np.zeros(shape = (n_i[i], 3)) for i in range(n_linktypes)]
        info_counter = np.zeros(n_linktypes)

        for i in range(n_info):
            temp_linktype = linktypes[i]
            index = np.where(uniq_linktypes == temp_linktype)[0][0]
            edge_counts[index] += info[i]
            info_triples[index][int(info_counter[index])] = np.array([sources[i], targets[i], info[i]])
            info_counter[index] += 1
            
        p_hats = edge_counts / n_i

        # trying to alleviate possible sampling biases...
        max_info_per_graph = (n_nodes**2 - n_nodes)/2 # (n_nodes choose 2) + n_nodes
        info_ratios = n_i / max_info_per_graph
        imputed_values = info_ratios*p_hats + (1 - info_ratios)*0.5

        # initialize matrices
        adjacency_matrices = [imputed_values[i] * np.ones(shape = (n_nodes, n_nodes)) for i in range(n_linktypes)]

        # update adjacency matrices
        for i in range(n_info):
            temp_linktype = linktypes[i]
            index = np.where(uniq_linktypes == temp_linktype)[0][0]
            adjacency_matrices[index][sources[i], targets[i]] = info[i]
            adjacency_matrices[index][targets[i], sources[i]] = info[i]

        graph_info_couple = [[] for i in range(n_linktypes)]

        for i in range(n_linktypes):
            info_triples[i] = container.ndarray(info_triples[i])
            adjacency_matrices[i] = container.ndarray(adjacency_matrices[i])
            graph_info_couple[i] = container.List([adjacency_matrices[i], info_triples[i]])

        return base.CallResult(container.List(graph_info_couple, contianer.ndarray(uniq_linktypes)))