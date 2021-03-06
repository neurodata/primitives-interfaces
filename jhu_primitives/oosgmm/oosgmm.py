#!/usr/bin/env python

# oosase.py
# Copyright (c) 2017. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict

import os

import numpy as np
from sklearn.decomposition import TruncatedSVD
import networkx

from scipy.stats import norm
from scipy.stats import rankdata

from d3m.primitive_interfaces.transformer import UnsupervisedLearnerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult


Inputs = container.List
Outputs = container.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    max_clusters = hyperparams.Bounded[int](
        default = 2,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
        ],
        lower = 2,
        upper = None
    )
    n_in_sample = hyperparams.Bounded[int](
        default = 1000,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ],
        lower = 1,
        upper = 10000
    )

class OutOfSampleGaussianClustering(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params,Hyperparams]):
    """
    Out of sample gaussian clustering that first runs Expectation-Maximization to 'learn' a clustering and susbequently classifies the remaining objects.""
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '5a9bc315-342d-474d-a1dd-ac018d61ae54',
        'version': "0.1.0",
        'name': "jhu.oosgmm",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.graph_clustering.out_of_sample_gaussian_clustering.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'gaussian clustering'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/oosase/oosase.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu'
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/neurodata/primitives-interfaces.git@{git_commit}#egg=jhu_primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
        }],
        'description': 'Expecation-Maxmization algorithm for clustering n_in_sample objects and classifying the remaining using the learned clusters.',
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris': [
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "EXPECTATION_MAXIMIZATION_ALGORITHM"
        ],
        'primitive_family': "GRAPH_CLUSTERING",
        'preconditions': ['NO_MISSING_VALUES']
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._embedding: container.ndarray = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        HH: # TODO
        """

        in_sample_n = self.hyperparams['n_in_sample']

        nodeIDs = inputs[1]
        nodeIDS = np.array([int(i) for i in nodeIDs])

        max_clusters = self.hyperparams['max_clusters']

        if max_clusters < self._embedding.shape[1]:
            self._embedding = self._embedding[:, :max_clusters].copy()

        if in_sample_n > n:
            in_sample_n = n

        in_sample_idx = np.random.choice(n, in_sample_n)
        out_sample_idx = np.setdiff1d(list(range(n)), in_sample_idx)

        in_sample_embedding = g[np.ix_(in_sample_idx, in_sample_idx)]
        out_sample_embedding = g[np.ix_(out_sample_idx, in_sample_idx)]

        cov_types = ['full', 'tied', 'diag', 'spherical']

        clf = GaussianMixture(n_components = 1, covariance_type = 'spherical')
        clf.fit(self._embedding)
        BIC_max = -clf.bic(self._embedding)
        cluster_likelihood_max = 1
        cov_type_likelihood_max = "spherical"

        for i in range(1, max_clusters):
            for k in cov_types:
                clf = GaussianMixture(n_components=i,
                                    covariance_type=k)

                clf.fit(self._embedding)

                current_bic = -clf.bic(self._embedding)

                if current_bic > BIC_max:
                    BIC_max = current_bic
                    cluster_likelihood_max = i
                    cov_type_likelihood_max = k

        clf = GaussianMixture(n_components = cluster_likelihood_max,
                        covariance_type = cov_type_likelihood_max)
        clf.fit(self._embedding)

        predictions = clf.predict(self._embedding)

        testing = inputs[2]

        testing_nodeIDs = np.asarray(testing['G1.nodeID'])
        testing_nodeIDs = np.array([int(i) for i in testing_nodeIDs])
        final_labels = np.zeros(len(testing))

        for i in range(len(testing_nodeIDs)):
            label = predictions[i]
            final_labels[i] = int(label) + 1

        testing['classLabel'] = final_labels
        outputs = container.DataFrame(testing[['d3mIndex', 'classLabel']])
        outputs[['d3mIndex', 'classLabel']] = outputs[['d3mIndex', 'classLabel']].astype(int)
        
        return base.CallResult(outputs)


    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._embedding = self._training_inputs[0].copy()

    def get_params(self) -> Params:
        return Params(embedding = self._embedding)

    def set_params(self, *, params: Params) -> None:
        self._embedding = params['embedding']

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        return base.CallResult(None)