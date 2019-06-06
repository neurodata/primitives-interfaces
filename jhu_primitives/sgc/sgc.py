#!/usr/bin/env python

# sgc.py
# Copyright (c) 2017. All rights reserved.

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
    CLASSIFICATION: GaussianClassification
    CLUSTERING: GaussianClustering
    nodeIDs: np.ndarray
    embedding: container.ndarray

class Hyperparams(hyperparams.Hyperparams):
    dim = None

class SpectralGraphClustering(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params,Hyperparams]):
    """
    Classification (QDA) and clustering (EM) super primitive.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'fde90b15-155a-3c2a-866c-4a19354cf0c7',
        'version': "0.1.0",
        'name': "jhu.sgc",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.vertex_nomination.spectral_graph_clustering.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'spectral clustering', 'clustering', 'classification'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/sgc/sgc.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu',

        },
        'description': 'Classification (QDA) and clustering (EM) super primitive',
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
            "EXPECTATION_MAXIMIZATION_ALGORITHM",
            "QUADRATIC_DISCRIMINANT_ANALYSIS"
        ],
        'primitive_family':
            "VERTEX_NOMINATION",
        'preconditions': [
            'NO_MISSING_VALUES'
        ]
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None

        self._supervised: bool = None
        self._fitted: bool = False

        self._CLASSIFICATION: GaussianClassification = None
        self._CLUSTERING: GaussianClustering = None

        self._nodeIDs: np.ndarray = None
        self._embedding: container.ndarray = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Performs spectral graph clustering: {GCLUST, GCLASS} o DIMSELECT o ASE o PTR o LCC

        Inputs
            d3m Dataset

        Outputs
            class predictions

        """
        hp_lcc = jhu.lcc.lcc.Hyperparams.defaults()
        new_lcc = LargestConnectedComponent(hyperparams=hp_lcc).produce(inputs=inputs).value

        if self._supervised:
            predictions = self._CLASSIFICATION.produce(inputs = new_lcc).value
        else:
            predictions = self._CLUSTERING.produce(inputs = new_lcc).value

        return base.CallResult(predictions)

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        hp_lcc = jhu.lcc.lcc.Hyperparams.defaults()
        G_lcc = LargestConnectedComponent(hyperparams = hp_lcc).produce(inputs = self._training_inputs).value
        self._nodeIDs = G_lcc[1]
        hp_ase = jhu.ase.ase.Hyperparams({'max_dimension': min(len(G_lcc[0]) - 1, 100), 'use_attributes': True, 'which_elbow': 2})
        G_ase = AdjacencySpectralEmbedding(hyperparams = hp_ase).produce(inputs = G_lcc).value

        self._embedding = G_ase[0]

        csv = G_lcc[2]

        if len(csv) == 0: # if passed an empty training set, we will use EM (gclust)
            self._supervised = False
            self._CLUSTERING = GaussianClustering(hyperparams = jhu.gclust.gclust.Hyperparams({'max_clusters': int( np.floor (np.log( len( G_lcc[0] ))))
                                                                                                }))
            self._CLUSTERING._embedding = self._embedding
            self._fitted = True
            return base.CallResult(None)

        self._supervised = True

        self._CLASSIFICATION = GaussianClassification(hyperparams = jhu.gclass.gclass.Hyperparams.defaults())

        self._CLASSIFICATION.set_training_data(inputs=G_ase)

        self._CLASSIFICATION.fit()

        self._fitted = True

        return base.CallResult(None)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        return Params(
            supervised=self._supervised,
            CLASSIFICATION = self._CLASSIFICATION,
            CLUSTERING = self._CLUSTERING,
            nodeIDs = self._nodeIDs,
            embedding= self._embedding
        )

    def set_params(self, *, params: Params) -> None:
        self._fitted = True
        self._supervised = params['supervised']
        self._CLASSIFICATION = params['CLASSIFICATION']
        self._CLUSTERING = params['CLUSTERING']
        self._nodeIDs = params['nodeIDs']
        self._embedding = params['embedding']