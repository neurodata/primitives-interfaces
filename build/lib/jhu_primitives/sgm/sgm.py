#!/usr/bin/env python
# sgm.py
# Copyright (c) 2017. All rights reserved.

#special thanks to Eriq Augustine from UCSC for helping us
#(Hayden and Joshua) understand how all this works

from typing import Sequence, TypeVar, Union, Dict
import os
import pandas as pd
import numpy as np
import networkx

from rpy2 import robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from ..utils.util import file_path_conversion

Inputs = container.Dataset
Outputs = container.DataFrame

PRIMITIVE_FAMILY = "GRAPH_MATCHING"

class Params(params.Params):
    None

class Hyperparams(hyperparams.Hyperparams):
    threshold = hyperparams.Bounded[float](
            default = .1,
            semantic_types = [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
            ],
            lower = 0.01,
            upper = 1
    )
    reps = hyperparams.Bounded[int](
            default = 1,
            semantic_types = [
                'https://metadata.datadrivendiscovery.org/types/TuningParameter'
            ],
            lower = 1,
            upper = None
    )


class SeededGraphMatching( UnsupervisedLearnerPrimitiveBase[Inputs, Outputs,Params, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'ff22e721-e4f5-32c9-ab51-b90f32603a56',
        'version': "0.1.0",
        'name': "jhu.sgm",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.SeededGraphMatching',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph matching'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/sgm/sgm.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:jagterb1@jhu.edu',
        },
        'installation': [{
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
        #metadata_module.PrimitiveFamily.GRAPH_MATCHING,
            PRIMITIVE_FAMILY,
        'preconditions': [
            'NO_MISSING_VALUES'
        ]
       })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._training_dataset = None
        self._g1 = None
        self._g2 = None
        self._g1_node_attributes = None
        self._g2_node_attributes = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        return CallResult[None]

    def set_training_data(self,*,inputs: Inputs) -> None:
        self._training_dataset = inputs
        self._g1 = self._training_dataset['0']
        self._g2 = self._training_dataset['1']
        self._g1_node_attributes = list(networkx.get_node_attributes(self._g1, 'nodeID').values())
        self._g2_node_attributes = list(networkx.get_node_attributes(self._g2, 'nodeID').values())
        #technically, this is unsupervised, as there is no fit function
        #instead, we just hang on to the training data and run produce with the two graphs and seeds
        #and use that to predict later on.

    def get_params(self) -> None:
        return Params

    def set_params(self, *, params: Params) -> None:
        pass
    #UnsupervisedLearner
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        #produce takes the training dataset and runs seeded graph matching using the seeds
        #then predicts using the resulting permutation_matrix

        permutation_matrix = np.asmatrix(self._seeded_graph_match(training_data=self._training_dataset))

        predictions = self._get_predictions(permutation_matrix=permutation_matrix, inputs = inputs)

        return base.CallResult(predictions)

    def _get_predictions(self,*, permutation_matrix: np.matrix, inputs: Inputs):
        testing = inputs['2']

        threshold = self.hyperparams['threshold']

        for i in range(testing.shape[0]):
            testing['match'][i] = 0
            v1 = testing['G1.nodeID'][i]
            v2 = testing['G2.nodeID'][i]
            found = False
            j = 0
            while not found:
                if self._g1_node_attributes[j] == int(v1):
                    found = True
                    v1 = j
                j += 1
            # print(found)
            found = False
            j = 0

            while not found:
                if self._g2_node_attributes[j] == int(v2):
                    found = True
                    v2 = j
                j += 1

            if permutation_matrix[v1, v2] > threshold:
                testing['match'][i] = 1
            else:
                testing['match'][i] = 0

        df = container.DataFrame({"d3mIndex": testing['d3mIndex'], "match": testing['match']})
        return df

    def _seeded_graph_match(self,*, training_data = None):
        if training_data is None:
            training_data = self._training_dataset
        seeds = training_data['2']

        new_seeds = pd.DataFrame(
            {'G1.nodeID': seeds['G1.nodeID'], 'G2.nodeID': seeds['G2.nodeID'], 'match': seeds['match']})
        new_seeds = new_seeds[new_seeds['match'] == '1']
        # we now have a seeds correspondence of nodeIDs,
        #  but we need a seed correspondence of actual vertex numbers

        # initialize the integer values to nothing:
        new_seeds['g1_vertex'] = ""
        new_seeds['g2_vertex'] = ""

        # for every seed, locate the corresponding vertex integer
        for j in range(new_seeds.shape[0]):
            found = False
            i = 0
            while not found:
                if (int(new_seeds['G1.nodeID'][j]) == self._g1_node_attributes[i]):
                    new_seeds['g1_vertex'][j] = i
                    found = True
                i += 1

        for j in range(new_seeds.shape[0]):
            found = False
            i = 0
            while not found:
                if (int(new_seeds['G2.nodeID'][j]) == self._g2_node_attributes[i]):
                    new_seeds['g2_vertex'][j] = i
                    found = True
                i += 1

        # store the vertex pairs as an m x 2 array and convert to a matrix
        seeds_array = np.array(new_seeds[['g1_vertex', 'g2_vertex']])
        seeds_array = seeds_array.astype(int)

        seeds = seeds_array
        nr, nc = seeds.shape
        seeds = ro.r.matrix(seeds, nrow=nr, ncol=nc)
        ro.r.assign("seeds", seeds)

        g1_matrix = networkx.to_numpy_array(self._g1)
        nr, nc = g1_matrix.shape
        g1_matrix = ro.r.matrix(g1_matrix, nrow=nr, ncol=nc)
        ro.r.assign("g1_matrix", g1_matrix)

        g2_matrix = networkx.to_numpy_array(self._g2)
        nr, nc = g2_matrix.shape
        g2_matrix = ro.r.matrix(g2_matrix, nrow=nr, ncol=nc)
        ro.r.assign("g2_matrix", g2_matrix)

        reps = self.hyperparams['reps']
        ro.r.assign("reps",reps)

        # run the R code:
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            "sgm.interface.R")
        path = file_path_conversion(path, uri="")

        cmd = """
                source("%s")
                fn <- function(g1_matrix, g2_matrix, seeds,reps) {
                    sgm.interface(g1_matrix, g2_matrix, seeds,reps)
                }
                """ % path
        
        result = np.array(ro.r(cmd)(g1_matrix, g2_matrix, seeds,reps))

        return container.ndarray(result)