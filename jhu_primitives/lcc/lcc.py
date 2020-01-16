import os
import sys
import json
from typing import Sequence, TypeVar, Union, Dict
import numpy as np
import networkx as nx
from networkx import Graph

from d3m import utils
from d3m import container
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

Inputs = container.List
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
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/lcc/lcc.py',
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
        np.random.seed(self.random_seed)
        print('lcc, baby!', file=sys.stderr)

        # unpack the data from the graph to list reader
        csv = inputs[0]
        G = inputs[1][0]
        nodeIDs = inputs[2]
        TASK = inputs[3]

        # split the data into connected components
        subgraphs = [G.subgraph(i).copy() for i in nx.connected_components(G)]

        # pick the largest connected component
        G_largest = [0]
        components = np.zeros(len(G), dtype=int)
        for i, connected_component in enumerate(subgraphs):
            # obtain indices associated with the node_ids in this component
            temp_indices = [i for i, x in enumerate(nodeIDs)
                            if x in [str(c) for c in list(connected_component)]]
            print(list(connected_component), file=sys.stderr)
            print(list(connected_component.nodes), file=sys.stderr)
            components[temp_indices] = i+1
            # check if the component is largest
            if len(connected_component) > len(G_largest):
                # if it is largest - flag as such
                G_largest = connected_component
                # and change the nodeIDs
                new_nodeIDs = nodeIDs[temp_indices]
        assert 1 == 0

        # for some problems the component needs to be specified in the dataframe
        # if TASK == "vertexClassification":
        #     csv['components'] = components[np.array(csv[NODEID], dtype=int)]
        if TASK == "communityDetection":
            csv['components'] = components

        # print(len(G_largest), file=sys.stderr)
        # print(len(nodeIDs), file=sys.stderr)
        # print(len(new_nodeIDs), file=sys.stderr)
        print("also print first 20 entries")
        # print(list(G_largest.nodes)[:20], file=sys.stderr)
        print(nodeIDs[:20], file=sys.stderr)
        print(new_nodeIDs[:20], file=sys.stderr)

        
        return base.CallResult(container.List([csv, [G_largest.copy()], new_nodeIDs]))
