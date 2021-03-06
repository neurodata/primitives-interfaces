#!/usr/bin/env python
# sgm.py
# Copyright (c) 2020. All rights reserved.

# Thanks to Ben Johnson for his SGM code

from typing import Sequence, TypeVar, Union, Dict
import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse

from .ben_sgm.backends.sparse import JVSparseSGM

from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

Inputs = container.Dataset
Outputs = container.DataFrame

class Params(params.Params):
    csv_train: container.DataFrame
    g1_idmap: dict
    g2_idmap: dict
    g1_adjmat: sparse.csr.csr_matrix
    g2_adjmat: sparse.csr.csr_matrix
    P: sparse.csr.csr_matrix

class Hyperparams(hyperparams.Hyperparams):
    None
    # threshold = hyperparams.Bounded[float](
    #         default = 0, #0.1
    #         semantic_types = [
    #         'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    #         ],
    #         lower = 0, #0.01
    #         upper = 1
    # )
    # reps = hyperparams.Bounded[int](
    #         default = 1,
    #         semantic_types = [
    #             'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    #         ],
    #         lower = 1,
    #         upper = None
    # )

class SeededGraphMatching( UnsupervisedLearnerPrimitiveBase[Inputs, Outputs,Params, Hyperparams]):
    """
    Finds the vertex alignment between two graphs that minimizes a relaxation of the Frobenious norm of the difference of the adjacency matrices of two graphs.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'ff22e721-e4f5-32c9-ab51-b90f32603a56',
        'version': "0.1.0",
        'name': "jhu.sgm",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.graph_matching.seeded_graph_matching.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph matching'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/sgm/sgm.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu',
        },
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
            },{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/neurodata/primitives-interfaces.git@{git_commit}#egg=jhu_primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
        }],
        'description': 'Finds the vertex alignment between two graphs that minimizes a relaxation of the Frobenious norm of the difference of the adjacency matrices of two graphs',
        'algorithm_types': [
            "FRANK_WOLFE_ALGORITHM"
            #metadata_module.PrimitiveAlgorithmType.FRANK_WOLFE_ALGORITHM
        ],
        'primitive_family':
            'GRAPH_MATCHING',
        'preconditions': [
            'NO_MISSING_VALUES'
        ]
       })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._fitted: bool = False
        self._csv_train = None
        self._g1_idmap = None
        self._g2_idmap = None
        self._g1_adjmat = None
        self._g2_adjmat = None
        self._P = None

    def _pad_graph(self, G1, G2):
        n_nodes = max(G1.order(), G2.order())
        for i in range(G1.order(), n_nodes):
            G1.add_node('__pad_g1_%d' % i, attr_dict = {'nodeID': i})

        for i in range(G2.order(), n_nodes):
            G2.add_node('__pad_g2_%d' % i, attr_dict = {'nodeID': i})
        assert G1.order() == G2.order()
        return G1, G2, n_nodes

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")

        return Params(
            csv_train = self._csv_train,
            g1_idmap = self._g1_idmap,
            g2_idmap = self._g2_idmap,
            g1_adjmat = self._g1_adjmat,
            g2_adjmat = self._g2_adjmat,
            P = self._P,
        )

    def set_params(self, *, params: Params) -> None:
        self._fitted = True
        self._csv_train = params['csv_train']
        self._g1_idmap = params['g1_idmap']
        self._g2_idmap = params['g2_idmap']
        self._g1_adjmat = params['g1_adjmat']
        self._g2_adjmat = params['g2_adjmat']
        self._P = params['P']

    def set_training_data(self, *, inputs: Inputs) -> None:
        # Grab both graphs. Cast as a Graph object in case inputs are Multigraphs.
        graph_dataframe0 = inputs['0']
        graph_dataframe1 = inputs['1']

        temp_json = inputs.to_json_structure()
        location_uri = temp_json['location_uris'][0]
        path_to_graph0 = location_uri[:-15] + "graphs/" + graph_dataframe0.at[0,'filename'] 
        path_to_graph1 = location_uri[:-15] + "graphs/" + graph_dataframe1.at[0,'filename'] 

        g1 = nx.read_gml(path=path_to_graph0[7:]) 
        g2 = nx.read_gml(path=path_to_graph1[7:]) 
        
        # Grab training data csv
        try:
            self._csv_train = inputs['learningData']
        except:
            self._csv_train = inputs['2']

        # Pad graphs if needed. As of 2/4/2019 only "naive" padding implemented.
        g1, g2, self._n_nodes = self._pad_graph(g1, g2)

        # it is possible that the following code can be majorly simplified

        # grab training nodeIDs
        # all nodeIDs are treated as strings (most general type)
        g1_nodeIDs_TRAIN = self._csv_train['G1.nodeID'].values.astype(str)
        g2_nodeIDs_TRAIN = self._csv_train['G2.nodeID'].values.astype(str)

        # extract nodeIDs of the whole graph
        # all nodeIDs are treated as strings (most general type)
        g1_nodeIDs = np.array(list(g1.nodes)).astype(str)
        g2_nodeIDs = np.array(list(g2.nodes)).astype(str)
            
        # Create mapping from nodeID to the node's index in its respective graph.
        # i.e. self._g1_idmap[nodeID_1] = node_1
        self._g1_idmap = dict(zip(g1_nodeIDs, range(self._n_nodes)))
        self._g2_idmap = dict(zip(g2_nodeIDs, range(self._n_nodes)))

        # Create new columns in the training csv for easy access of node indices.
        self._csv_train['new_g1_id'] = pd.Series(g1_nodeIDs_TRAIN).apply(lambda x: self._g1_idmap[x])
        self._csv_train['new_g2_id'] = pd.Series(g2_nodeIDs_TRAIN).apply(lambda x: self._g2_idmap[x])

        # Grab G1, G2 adjacency matrices.
        g1_adjmat = nx.adjacency_matrix(g1)
        g2_adjmat = nx.adjacency_matrix(g2)

        # Symmetrize the adjacency matrices of G1, G2.
        self._g1_adjmat = ((g1_adjmat + g1_adjmat.T) > 0).astype(np.float32)
        self._g2_adjmat = ((g2_adjmat + g2_adjmat.T) > 0).astype(np.float32)


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        # reps = self.hyperparams['reps']
        # Grab trianing csv to access seed information.
        csv_array = np.array(self._csv_train)

        # Grab row indices of seeds.
        seed_idx = self._csv_train['match'] == '1'

        # Initialize P as an array of ones.
        P = csv_array[:, -2:][seed_idx]

        # Transform P into an (n_nodes x n_nodes) sparse matrix with 1's corresponding to the seeded matches.
        P = sparse.csr_matrix((np.ones(P.shape[0]), (P[:,0].astype(int), P[:,1].astype(int))), shape=(self._n_nodes, self._n_nodes))

        # Initialize an SGM object that will perform the optimization.
        sgm = JVSparseSGM(A = self._g1_adjmat, 
                          B = self._g2_adjmat,
                          P = P)

        # Frank Wolfe / LAP solver.
        P_out = sgm.run(
            num_iters = 20,
            tolerance = 1,
            verbose = False,
            random_seed = self.random_seed)

        # Final P.
        P_out = sparse.csr_matrix((np.ones(self._n_nodes), (np.arange(self._n_nodes), P_out)))

        # Accessible later.
        self._P = P_out
        self._fitted = True
        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        csv_TEST = inputs['learningData']

        g1_dense = self._g1_adjmat.todense()
        g2_dense = self._g2_adjmat.todense()

        identity_objective = np.linalg.norm(g1_dense - g2_dense)

        found_P_objective = np.linalg.norm(g1_dense - self._P.T @ g2_dense @ self._P)

        if identity_objective < found_P_objective:
            self._P = np.eye(self._g1_adjmat.shape[0])

        permutation_matrix = self._P

        g1_nodeIDs_TEST = csv_TEST['G1.nodeID'].values
        g1_nodeIDs_TEST = g1_nodeIDs_TEST.astype(str)

        g2_nodeIDs_TEST = csv_TEST['G2.nodeID'].values
        g2_nodeIDs_TEST = g2_nodeIDs_TEST.astype(str)

        n_test = len(g1_nodeIDs_TEST)

        #threshold = self.hyperparams['threshold']
        threshold = 0

        matches = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            g1_ind = self._g1_idmap[str(csv_TEST['G1.nodeID'].iloc[i])]
            g2_ind = self._g2_idmap[str(csv_TEST['G2.nodeID'].iloc[i])]
            matches[i] = int(permutation_matrix[int(g1_ind), int(g2_ind)] > threshold)
        csv_TEST['match'] = matches

        predictions = {"d3mIndex": csv_TEST['d3mIndex'], "match": csv_TEST['match']}
        return base.CallResult(container.DataFrame(predictions), has_finished = True, iterations_done = 1)
