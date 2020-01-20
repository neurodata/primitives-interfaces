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
        print('lcc produce started', file=sys.stderr)

        # unpack the data from the graph to list reader
        learning_data, graphs_full_all, nodeIDs_full_all, task_type = inputs

        # initialize lists for connected components and associated nodeids
        graphs_largest_all = []
        nodeIDs_largest_all = []

        for graph_index in range(len(graphs_full_all)):
            # select the graph and node ids for the current graph
            graph_full = graphs_full_all[graph_index]
            nodeIDs_full = nodeIDs_full_all[graph_index]

            # split the current graph into connected components
            subgraphs = [graph_full.subgraph(i).copy()
                        for i in nx.connected_components(graph_full)]

            # pick the largest connected component of the current graph
            graph_largest = [0]
            components = np.zeros(len(graph_full), dtype=int) # only for CD
            for i, connected_component in enumerate(subgraphs):
                # obtain indices associated with the node_ids in this component
                temp_indices = [i for i, x in enumerate(nodeIDs_full)
                                if x in [str(c) for c in list(connected_component)]]
                components[temp_indices] = i
                # check if the component is largest
                if len(connected_component) > len(graph_largest):
                    # if it is largest - flag as such
                    graph_largest = connected_component.copy()
                    # and subselect the appropriate nodeIDs
                    nodeIDs_largest = nodeIDs_full[temp_indices]

            # append the largest_connected component and nodeIDs
            graphs_largest_all.append(graph_largest)
            nodeIDs_largest_all.append(nodeIDs_largest)

            # for communityDetection the component needs to be specified in
            # the dataframe; in this problem there is always only one graph
            # TODO: condsider avoiding the specification of the problem
            #       likely can be achiebed by handling nodeIDs data smartly
            if task_type == "communityDetection":
                learning_data['components'] = components

        outputs = container.List([
            learning_data, graphs_largest_all, nodeIDs_largest_all])

        debugging = True
        if debugging:
            # CSV STUFF
            print("label counts:", file=sys.stderr)
            for i in range(10):
                print("label: {}, count: {}".format(
                    i, np.sum(learning_data['label'] == str(i))), file=sys.stderr)
            # GRAPH STUFF
            print("length of the first graph: {}".format(
                len(list(graphs_largest_all[0].nodes()))), file=sys.stderr)
            print("first 20 nodes of the first graph", file=sys.stderr)
            print(list(graphs_largest_all[0].nodes())[:20], file=sys.stderr)
            # NODE IDS STUFF
            print("type of a nodeID: {}".format(
                type(nodeIDs_largest_all[0][0])), file=sys.stderr)
            print("length of the nodeIds: {}".format(
                len(nodeIDs_largest_all)), file=sys.stderr)
            print("first 20 nodesIDs", file=sys.stderr)
            print(nodeIDs_largest_all[0][:20], file=sys.stderr)
            # TASK STUFF
            print("task: {}". format(task_type), file=sys.stderr)
            print("graph reader produce ended", file=sys.stderr)

        print('lcc produce ended', file=sys.stderr)

        assert 1 == 0
        return base.CallResult(outputs)
