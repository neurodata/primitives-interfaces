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

class LargestConnectedComponent(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Finds the largest connected component of a graph.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '32fec24f-6861-4a4c-88f3-d4ec2bc1b486',
        'version': "0.1.0",
        'name': "jhu.lcc",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_preprocessing.largest_connected_component.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'connected', 'largest connected component', 'graph','graph transformation','transformation'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/lcc/lcc.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu',
        },
        'description': 'Finds the largest connected component of a graph',
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [
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
            #"BREADTH_FIRST_SEARCH"
            "NONOVERLAPPING_COMMUNITY_DETECTION"
        ],
        'primitive_family': "DATA_PREPROCESSING",
        'preconditions': ['NO_MISSING_VALUES']
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
        try:
            G = inputs['0']
        except:
            edge_list = inputs['1'] # for edge lists
            V1_nodeIDs = np.array(edge_list.V1_nodeID.values).astype(int)
            V2_nodeIDs = np.array(edge_list.V2_nodeID.values).astype(int)
            edge_weights = np.array(edge_list.edge_weight.values).astype(float).astype(int)
            n_edges = len(V1_nodeIDs)

            unique_V1_nodeIDs = np.unique(V1_nodeIDs)
            unique_V2_nodeIDs = np.unique(V2_nodeIDs)

            concatenated_unique_IDs = np.concatenate((unique_V1_nodeIDs, unique_V2_nodeIDs))

            unique_all = np.unique(concatenated_unique_IDs)

            n_nodes = len(unique_all)

            G = nx.Graph()
            G.add_nodes_from(unique_all)

            for i in range(n_edges):
                G.add_edge(V1_nodeIDs[i], V2_nodeIDs[i], weight = edge_weights[i])

        csv = inputs['learningData']

        #if len(list(nx.get_node_attributes(G, 'nodeID').values())) == 0:
        #    nx.set_node_attributes(G,'nodeID',-1)
        #    for i in range(len(G)):
        #        G.node[i]['nodeID'] = i

        if len(csv) != 0:
            if len(list(nx.get_node_attributes(G, 'nodeID').values())) == 0:
                nx.set_node_attributes(G,'nodeID',-1)
                for i in range(len(G)):
                    G.node[i]['nodeID'] = i

            nodeIDs = list(nx.get_node_attributes(G, 'nodeID').values())
            nodeIDs = container.ndarray(np.array([int(i) for i in nodeIDs]))

            return base.CallResult(container.List([G.copy(), nodeIDs,csv]))

        if type(G) == np.ndarray:
            if G.ndim == 2:
                if G.shape[0] == G.shape[1]: # n x n matrix
                    G = Graph(G)
                else:
                    raise TypeError("Networkx graphs or n x n numpy arrays only")

        subgraphs = [G.subgraph(i).copy() for i in nx.connected_components(G)]

        G_connected = [[0]]
        for i in subgraphs:
            if len(i) > len(G_connected[0]):
                G_connected = [i]

        nodeIDs = list(nx.get_node_attributes(G_connected[0], 'nodeID').values())
        nodeIDs = container.ndarray(np.array([int(i) for i in nodeIDs]))

        return base.CallResult(container.List([G_connected[0].copy(), nodeIDs, csv]))
