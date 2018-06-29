#!/usr/bin/env python

# gclust.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
import numpy as np

from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from sklearn.mixture import GaussianMixture

Inputs = container.List
Outputs = container.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    max_clusters = hyperparams.Hyperparameter[int](default = 2,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    #seeds = hyperparams.Hyperparameter[np.ndarray](default = np.array([]), semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    #labels = hyperparams.Hyperparameter[np.ndarray](default = np.array([]), semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

class GaussianClustering(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params,Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '5194ef94-3683-319a-9d8d-5c3fdd09de24',
        'version': "0.1.0",
        'name': "jhu.gclust",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.GaussianClustering',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['gaussian clustering'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/gclust/gclust.py',
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
            },
            {
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
            "EXPECTATION_MAXIMIZATION_ALGORITHM"
        ],
        'primitive_family': "CLUSTERING"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

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

        inputs = inputs[0]
        max_clusters = self.hyperparams['max_clusters']

        if max_clusters < inputs.shape[1]:
            inputs = inputs[:, :max_clusters].copy()

        cov_types = ['full', 'tied', 'diag', 'spherical']

        clf = GaussianMixture(n_components = 1, covariance_type = 'spherical')
        clf.fit(inputs)
        BIC_max = -clf.bic(inputs)
        cluster_likelihood_max = 1
        cov_type_likelihood_max = "spherical"

        for i in range(1, max_clusters + 5):
            for k in cov_types:
                clf = GaussianMixture(n_components=i, 
                                    covariance_type=k)

                clf.fit(inputs)

                current_bic = -clf.bic(inputs)

                if current_bic > BIC_max:
                    BIC_max = current_bic
                    cluster_likelihood_max = i
                    cov_type_likelihood_max = k

        clf = GaussianMixture(n_components = cluster_likelihood_max,
                        covariance_type = cov_type_likelihood_max)
        clf.fit(inputs)

        predictions = clf.predict(inputs)

        testing = inputs[2]
        testing_nodeIDs = np.asarray(testing['G1.nodeID'])
        final_labels = np.zeros(len(testing))

        for i in range(len(testing_nodeIDs)):
            temp = int(np.where(self._nodeIDs == testing_nodeIDs[i])[0][0])
            label = predictions[temp]
            final_labels[i] = label

        testing['classLabel'] = final_labels
        outputs = container.DataFrame(testing[['d3mIndex', 'classLabel']])
        return base.CallResult(outputs)


    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        
    def get_params(self) -> None:
        return Params

    def set_params(self, *, params: Params) -> None:
        pass

        """
        else:
            clf = GaussianMixture(n_components = 1, covariance_type = 'spherical')
            clf.fit(inputs[seeds, :])
            BIC_max = -clf.bic(inputs[seeds, :])
            cluster_likelihood_max = 1
            cov_type_likelihood_max = "spherical"

            for i in range(1, max_clusters + 1):
                for k in cov_types:
                    clf = GaussianMixture(n_components=i, 
                                    covariance_type=k, n_init = 50)

                    clf.fit(inputs)

                    current_bic = -clf.bic(inputs[seeds, :])

                    if current_bic > BIC_max:
                        BIC_max = current_bic
                        cluster_likelihood_max = i
                        cov_type_likelihood_max = k

            estimated_clf = GaussianMixture(n_components = cluster_likelihood_max, 
                                                covariance_type = cov_type_likelihood_max, n_init = 50)

            estimated_clf.fit(inputs[seeds, :])
            estimated_labels = estimated_clf.predict(inputs[seeds, :])

            unique_labels = np.unique(estimated_labels)
            estimated_K = len(unique_labels)
            votes = [[] for i in range(estimated_K)]

            for i in range(len(estimated_labels)):
                for k in range(estimated_K):
                    if int(estimated_labels[i]) == k:
                        votes[k].append(labels[i])

            votes = [np.array(votes[i]) for i in range(estimated_K)]

            label_mapping = -1*np.ones(estimated_K)

            for i in range(estimated_K): # majority wins
                temp_unique, temp_unique_counts = np.unique(votes[i], return_counts = True)
                temp_counts_argmax = np.argmax(temp_unique_counts)
                label_mapping[i] = temp_unique[temp_counts_argmax]

            estimated_K_labels = estimated_clf.predict(inputs)
            applied_label_mapping = -1*np.ones(inputs.shape[0])

            for i in range(len(estimated_K_labels)):
                temp_label = int(estimated_K_labels[i])
                applied_label_mapping[i] = label_mapping[temp_label]

            predictions = applied_label_mapping
            """

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        pass