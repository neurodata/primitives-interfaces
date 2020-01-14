from networkx import Graph
import networkx as nx
import numpy as np
from typing import Sequence, TypeVar, Union, Dict
import os
import sys

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
    """
    A primitive for reading in a multi-graph, typically used in a JHU link prediction pipeline.
    """
    
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '09f2eea8-667c-44b8-a955-6a153ba9ccc3',
        'version': "0.1.0",
        'name': "jhu.link_pred_graph_reader",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.link_prediction.data_conversion.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'graph reader'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/link_pred_graph_reader/link_pred_graph_reader.py',
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
            "DATA_CONVERSION"
        ],
        'primitive_family': "LINK_PREDICTION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # read in graph and training csv
        np.random.seed(self.random_seed)
        graph_dataframe = inputs['0']
        csv = inputs['learningData']
        
        # antons debugging feel free to delete
        # print("start of anton debugging", file=sys.stderr)

        # print(dir(inputs), file=sys.stderr)
        # for i in inputs:
        #     print(i, file=sys.stderr)
        #     print(type(i), file=sys.stderr)
        # print(type(inputs['0']), file=sys.stderr)
#        print(inputs['0'].edges.data(), file=sys.stderr)
        # print(type(graph_dataframe.at[0, 'filename']), file=sys.stderr)
        # print(graph_dataframe.at[0, 'filename'], file=sys.stderr)

        # print("end of anton debugging", file=sys.stderr)


        temp_json = inputs.to_json_structure()
        location_uri = temp_json['location_uris'][0]
        path_to_graph = location_uri[:-15] + "graphs/" + graph_dataframe.at[0,'filename'] 
        graph = nx.read_gml(path=path_to_graph[7:]) 
        n = len(graph)

        # grab link types (values) and edge list (keys)
        values = np.array(list(nx.get_edge_attributes(graph, 'linkType').values()), dtype=int)
        keys = np.array(list(nx.get_edge_attributes(graph, 'linkType').keys()), dtype=int)

        # grab the unique link types
        uniq_linktypes = np.unique(values)
        M = len(uniq_linktypes)

        if M == 0:
            M=1
            n_edges = np.array([len(list(graph.edges))])
            values = np.zeros(n_edges[0])
            keys = np.array(list(graph.edges), dtype=int)    
        else:
            n_edges = np.zeros(M) # imputation

            for i in range(len(values)):
                temp_linktype = int(values[i])
                n_edges[temp_linktype] += 1 # imputation

        n_choose_2 = (n**2 - n) / 2
        A_imps = [0.5*(0.5 + n_edges[i]/n_choose_2)*np.ones((n, n)) for i in range(M)]

        for i in range(len(values)):
            temp_linktype = int(values[i])
            A_imps[temp_linktype][keys[i][0]-1, keys[i][1]-1] = 1
            A_imps[temp_linktype][keys[i][1]-1, keys[i][0]-1] = 1

        
        for i in range(M):
            imputations = 0
            while imputations < n_edges[i]:
                v1 = np.random.randint(n)
                v2 = np.random.randint(n)
                if v1 == v2 or A_imps[i][v1, v2] == 1:
                    pass
                else:
                    A_imps[i][v1, v2] = 0
                    A_imps[i][v2, v1] = 0
                    imputations += 1

        A = -1*np.zeros(shape = (M*n, M*n))

        for i in range(M):
            for j in range(i, M):
                A[i*n: (i + 1)*n, j*n: (j + 1)*n] = (A_imps[i] + A_imps[j])/2
                A[j*n: (j + 1)*n, i*n: (i + 1)*n] = (A_imps[i] + A_imps[j])/2 

        info = container.List([n, M])
        link_prediction = True

        # # initialize a list of graphs to pass around
        # list_of_graphs = [nx.Graph() for i in range(M)]

        # # each graph is on the same node set
        # for i in range(M):
        #     list_of_graphs[i].add_nodes_from(graph) 

        # # populate the graphs with edges
        # for i in range(len(values)):
        #     temp_G = list_of_graphs[values[i]]
        #     temp_G.add_edge(keys[i][0], keys[i][1])
        #     temp_G.add_edge(keys[i][1], keys[i][0])

        return base.CallResult(container.List([container.ndarray(A), csv, info, link_prediction]))
