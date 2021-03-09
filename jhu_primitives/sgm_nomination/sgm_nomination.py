#!/usr/bin/env python

# ase.py
# Copyright (c) 2020. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
import sys
import numpy as np

from scipy.spatial.distance import cdist
from graspologic.match import GraphMatch

from d3m import utils
from d3m import container
from d3m import exceptions
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    match: container.ndarray

class Hyperparams(hyperparams.Hyperparams):
    pass

class SgmNomination(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Creates a similarity matrix from pairwise distances and nominates one-to-one
    smallest distance vertex match.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '8a913585-c269-4b25-ad32-40feda3c540e',
        'version': "0.1.0",
        'name': "jhu.sgm_nomination",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['nomination', 'matching', 'linear sum assignment'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/sgm_nomination/sgm_nomination.py',
                'https://github.com/neurodata/primitives-interfaces',
            ],
            'contact': 'mailto:asaadel1@jhu.edu'
        },
        'description': 'Creates two similarity matrices from pairwise distances and nominates one-to-one using seeded graph matching.',
        'hyperparams_configuration': {
            # 'max_dimension': 'The maximum dimension that can be used for eigendecomposition',
            # 'which_elbow': 'The scree plot "elbow" to use for dimensionality reduction. High values leads to more dimensions selected.',
            # 'use_attributes': 'Boolean which indicates whether to use the attributes of the nodes.'
        },
        # A list of dependencies in order. These can be Python packages, system
        # packages, or Docker images. Of course Python packages can also have
        # their own dependencies, but sometimes it is necessary to install a
        # Python package first to be even able to run setup.py of another
        # package. Or you have a dependency which is not on PyPi.
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
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
        }],
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris': [
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.graph_matching.sgm_nomination.JHU',
        # Choose these from a controlled vocabulary in the schema. If anything
        # is missing which would best describe the primitive, make a merge
        # request.
        'algorithm_types': [
            "RANDOM_GRAPH"
        ],
        'primitive_family':
            'GRAPH_MATCHING',
        'preconditions': [
            'NO_MISSING_VALUES'
        ]
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0,
                 docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams,
                         random_seed=random_seed,
                         docker_containers=docker_containers)
        self._fitted: bool = False
        self._training_inputs: Inputs = None
        self._match: container.ndarray = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        xhat = self._inputs_1
        yhat = self._inputs_2

        seeds = self._reference['match'].astype(bool)

        xhat_seed_names = self._reference[self._reference.columns[1]][seeds].values
        yhat_seed_names = self._reference[self._reference.columns[2]][seeds].values
        n_seeds = len(xhat_seed_names)

        x_seeds = np.zeros(n_seeds)
        y_seeds = np.zeros(n_seeds)
        for i in range(n_seeds):
            x_seeds[i] = (xhat[xhat.columns[0]] == xhat_seed_names[i]).nonzero()[0][0]

            y_seeds[i] = (yhat[yhat.columns[0]] == yhat_seed_names[i]).nonzero()[0][0]


        # do this more carefully TODO
        xhat_embedding = xhat.values[:,1:].astype(np.float32)
        yhat_embedding = yhat.values[:,1:].astype(np.float32)

        S_xx = np.exp(-cdist(xhat_embedding, xhat_embedding, ))
        S_yy = np.exp(-cdist(yhat_embedding, yhat_embedding, ))

        gmp = GraphMatch(shuffle_input=False)
        self._match = gmp.fit_predict(S_xx, S_yy, x_seeds, y_seeds)
        self._fitted = True

        return CallResult(None)

    def produce(self, *,
                inputs_1: Inputs,
                inputs_2: Inputs,
                reference: Inputs,
                timeout: float = None,
                iterations: int = None) -> CallResult[Outputs]:
        xhat = inputs_1
        yhat = inputs_2


        matches = np.zeros(len(reference), dtype=int)
        for i in range(len(reference)):
            e_id = xhat.index[xhat['e_nodeID'] == reference['e_nodeID'].iloc[i]]
            g_id = yhat.index[yhat['g_nodeID'] == reference['g_nodeID'].iloc[i]]
            matches[i] = 1 if g_id == self._match[e_id] else 0

        reference['match'] = matches

        results = reference[['d3mIndex', 'match']]

        predictions = {"d3mIndex": reference['d3mIndex'], "match": reference['match']}
        return base.CallResult(container.DataFrame(predictions),
                               has_finished = True, iterations_done = 1)

        # return base.CallResult(reference, #results,
        #                        has_finished=True,
        #                        iterations_done=1)

    def multi_produce(self, *,
                      produce_methods: Sequence[str],
                      inputs_1: Inputs,
                      inputs_2: Inputs,
                      reference: Inputs,
                      timeout: float = None,
                      iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._multi_produce(produce_methods=produce_methods,
                                   timeout=timeout,
                                   iterations=iterations,
                                   inputs_1=inputs_1,
                                   inputs_2=inputs_2,
                                   reference=reference)

    def fit_multi_produce(self, *,
                          produce_methods: Sequence[str],
                          inputs_1: Inputs,
                          inputs_2: Inputs,
                          reference: Inputs,
                          timeout: float = None,
                          iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._fit_multi_produce(produce_methods=produce_methods,
                                       timeout=timeout,
                                       iterations=iterations,
                                       inputs_1=inputs_1,
                                       inputs_2=inputs_2,
                                       reference=reference)

    def set_training_data(self, *, 
                    inputs_1: Inputs,
                    inputs_2: Inputs,
                    reference: Inputs) -> None:
        self._inputs_1 = inputs_1
        self._inputs_2 = inputs_2
        self._reference = reference

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")

        return Params(
            match = self._match)

    def set_params(self, *, params: Params) -> None:
        self._fitted = True
        self._match = params['match']



