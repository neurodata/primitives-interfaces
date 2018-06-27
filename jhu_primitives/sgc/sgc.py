#!/usr/bin/env python

# sgc.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os
import networkx
import numpy as np

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from jhu_primitives import LargestConnectedComponent
from jhu_primitives import AdjacencySpectralEmbedding
from jhu_primitives import GaussianClustering
from jhu_primitives import GaussianClassification

import jhu_primitives as jhu

Inputs = container.Dataset
Outputs = container.DataFrame

class Params(params.Params):
    supervised: bool
    pis: container.ndarray
    means: container.ndarray
    covariances: container.ndarray

class Hyperparams(hyperparams.Hyperparams):
    dim = None

class SpectralGraphClustering(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params,Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'fde90b15-155a-3c2a-866c-4a19354cf0c7',
        'version': "0.1.0",
        'name': "jhu.sgc",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.SpectralGraphClustering',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'spectral clustering', 'clustering', 'classification'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/sgc/sgc.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
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
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris': [
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "SPECTRAL_CLUSTERING"
        ],
        'primitive_family': "GRAPH_CLUSTERING"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None

        self._supervised: bool = None
        self._fitted: bool = False 

        self._CLASSIFICATION: GaussianClassification = None
        self._CLUSTERING: GaussianClustering = None

        self._embedding: container.ndarray = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Performs spectral graph clustering: {GCLUST, GCLASS} o DIMSELECT o ASE o PTR o LCC

        Inputs
            d3m Dataset

        Outputs
            class predictions

        """

        if self._supervised:
            predictions = self._CLASSIFICATION.produce(inputs = container.List([self._embedding])).value # dummy input
        else:
            predictions = self._CLUSTERING.produce(inputs = container.List([self._embedding])).value

        outputs = container.ndarray(predictions)

        return base.CallResult(outputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        hp_lcc = jhu.ase.ase.Hyperparams.defaults()
        G_lcc = LargestConnectedComponent(hyperparams = hp_lcc).produce(inputs = self._training_inputs).value

        hp_ase = jhu.ase.ase.Hyperparams({'max_dimension': min(len(G_lcc[0]) - 1, 100), 'which_elbow': 2})
        G_ase = AdjacencySpectralEmbedding(hyperparams = hp_ase).produce(inputs = G_lcc).value

        self._embedding = G_ase

        csv = self._training_inputs['1']

        if len(csv) == 0: # if passed an empty training set, we will use EM (gclust)
            self._supervised = False
            self._CLUSTERING = GaussianClustering(hyperparams = jhu.gclust.gclust.Hyperparams({'max_clusters': int(np.floor(np.log(len(G_lcc[0])))),
                                                                                                'seeds': np.array([]), 
                                                                                                'labels': np.array([])}
                                                                                                ))
            self._fitted = True
            return base.CallResult(None)

        self._supervised = True

        seeds = container.ndarray(csv['G1.nodeID'])
        labels = container.ndarray(csv['classLabel'])
        
        self._CLASSIFICATION = GaussianClassification(hyperparams = jhu.gclass.gclass.Hyperparams.defaults())

        self._CLASSIFICATION.set_training_data(inputs = container.List([self._embedding, seeds, labels]))
        self._CLASSIFICATION.fit()

        self._fitted = True

        return base.CallResult(None)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs

    def get_params(self) -> None:
        return Params

    def set_params(self, *, params: Params) -> None:
        pass