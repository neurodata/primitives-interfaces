#!/usr/bin/env python

# gclust.py
# Copyright (c) 2017. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
import numpy as np
import sys as sys

from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from graspy.cluster.gclust import GaussianCluster as graspyGCLUST

Inputs = container.List
Outputs = container.DataFrame

class Params(params.Params):
    embedding : container.ndarray

class Hyperparams(hyperparams.Hyperparams):
    max_clusters = hyperparams.Bounded[int](
        default = 2,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
        ],
        lower = 2,
        upper = None
    )

class GaussianClustering(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params,Hyperparams]):
    """
    Expecation-Maxmization algorithm for clustering
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '5194ef94-3683-319a-9d8d-5c3fdd09de24',
        'version': "0.1.0",
        'name': "jhu.gclust",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.graph_clustering.gaussian_clustering.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'gaussian clustering'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/gclust/gclust.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu',
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
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/neurodata/primitives-interfaces.git@{git_commit}#egg=jhu_primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
        }],
        'description': 'Expecation-Maxmization algorithm for clustering',
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris':
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

        self._embedding: container.ndarray = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        TODO: YP description

        **Positional Arguments:**

        inputs:
            - A matrix

        **Optional Arguments:**

        dim:
            - The number of clusters in which to assign the data
        """

        #print('gclust, baby!!', file=sys.stderr)
        if self._embedding is None:
            self._embedding = inputs[1][0]

        nodeIDs = inputs[2]
        nodeIDS = np.array([int(i) for i in nodeIDs])

        max_clusters = self.hyperparams['max_clusters']

        if max_clusters < self._embedding.shape[1]:
            self._embedding = self._embedding[:, :max_clusters].copy()

        gclust_object = graspyGCLUST(min_components=max_clusters, covariance_type="all")
        gclust_object.fit(self._embedding)
        model = gclust_object.model_
        

        testing = inputs[0]

        # am sure whats going on here..
        try:
            testing_nodeIDs = np.asarray(testing['nodeID']).astype(int)
        except:
            return base.CallResult(testing)

        final_labels = np.zeros(len(testing_nodeIDs))
        
        predictions = np.zeros(len(testing))
        g_indices = np.where(testing['components'] == 1)[0].astype(int)

        predictions[g_indices] = model.predict(self._embedding)
        for i in range(len(testing)):
            if i in g_indices:
                label = predictions[i]
                final_labels[i] = int(label) + 1
            else:
                final_labels[i] = int(max(predictions)) + int(testing['components'][i])
    
        testing['community'] = final_labels
        outputs = container.DataFrame(testing[['d3mIndex', 'community']])
        outputs[['d3mIndex', 'community']] = outputs[['d3mIndex', 'community']].astype(int)
        return base.CallResult(outputs)


    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs

    def get_params(self) -> Params:
        return Params(embedding = self._embedding)

    def set_params(self, *, params: Params) -> None:
        self._embedding = params['embedding']

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        return base.CallResult(None)


        # clf.fit(self._embedding)
        # BIC_max = -clf.bic(self._embedding)
        # cluster_likelihood_max = 1
        # cov_type_likelihood_max = "spherical"

        # for i in range(1, max_clusters):
        #     for k in cov_types:
        #         clf = GaussianMixture(n_components=i,
        #                             covariance_type=k)

        #         clf.fit(self._embedding)

        #         current_bic = -clf.bic(self._embedding)

        #         if current_bic > BIC_max:
        #             BIC_max = current_bic
        #             cluster_likelihood_max = i
        #             cov_type_likelihood_max = k

        # clf = GaussianMixture(n_components = cluster_likelihood_max,
        #                 covariance_type = cov_type_likelihood_max)
        # clf.fit(self._embedding)
