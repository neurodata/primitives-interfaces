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
        #print('lcc, baby!', file=sys.stderr)        

        csv = inputs[0]
        G = inputs[1][0]
        nodeIDs = inputs[2]
        TASK = inputs[3]

        # print(len(G), file=sys.stderr)
        subgraphs = [G.subgraph(i).copy() for i in nx.connected_components(G)]
        
        components = np.zeros(len(G), dtype=int)
        for i, connected_component in enumerate(nx.connected_components(G)):
            #print(np.array(list(connected_component), dtype=int), file=sys.stderr)
            components[np.array(list(connected_component), dtype=int)] = i+1

        # NODEID = ""
        # for header in csv.columns:
        #     if "nodeID" in header:
        #         NODEID = header
        # nodeIDs = list(csv[NODEID].values)
        
        # if TASK == "vertexClassification":
        #     csv['components'] = components[np.array(csv[NODEID], dtype=int)]
        if TASK == "communityDetection":
            csv['components'] = components        
            
        G_connected = [0]
        for i in subgraphs:
            if len(i) > len(G_connected):
                G_connected = i

        return base.CallResult(container.List([csv, [G_connected.copy()], nodeIDs]))
