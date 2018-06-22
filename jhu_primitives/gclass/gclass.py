#!/usr/bin/env python

# gclass.py
# Copyright (c) 2017. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as MVN
import numpy as np
from networkx import Graph

Inputs = container.List
Outputs = container.Dataframe

class Params(params.Params):
    pis: container.ndarray
    means: container.ndarray
    covariances: container.ndarray

class Hyperparams(hyperparams.Hyperparams):
    #number_of_clusters = hyperparams.Hyperparameter[int](default = 2,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    #seeds = hyperparams.Hyperparameter[np.ndarray](default=np.array([]), semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    #labels = hyperparams.Hyperparameter[np.ndarray](default=np.array([]), semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

class GaussianClassification(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'c9d5da5d-0520-468e-92df-bd3a85bb4fac',
        'version': "0.1.0",
        'name': "jhu.gclass",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.GaussianClassification',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['gaussian classification', 'graph', 'graphs', 'classification', 'supervised', 'supervised learning'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/gclass/glass.py',
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
            "MULTICLASS_CLASSIFICATION"
        ],
        'primitive_family': "CLASSIFICATION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._fitted: bool = False 
        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None

        self._embedding

        self._seeds: container.ndarray = None
        self._labels: container.ndarray = None

        self._ENOUGH_SEEDS: bool = False 
        self._PD: bool = False 

        self._pis: container.ndarray = None
        self._means: container.ndarray = None
        self._covariances: container.ndarray = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Gaussian classification (i.e. seeded gaussian "clustering").

        Inputs
            D - An n x d feature numpy array
        Returns
            labels - Class labels for each unlabeled vertex
        """

        if not self._fitted:
            raise ValueError("Not fitted")

        n = self._embedding.shape[0]
        final_labels = np.zeros(n)

        if self._PD and self._ENOUGH_SEEDS:
            for i in range(n): 
                if i not in self._seeds:
                    weighted_pdfs = np.array([self._pis[j]*MVN.pdf(self._embedding[i,:], self._means[j], self._covariances[j, :, :]) for j in range(K)]) 
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = label
        else:
            for i in range(n):
                if i not in self._seeds:
                    weighted_pdfs = np.array([self._pis[j]*MVN.pdf(self._embedding[i,:], self._means[j], self._covariances) for j in range(K)])
                    label = np.argmax(weighted_pdfs)
                    final_labels[i] = label
        
        for i in range(len(self._seeds)):
            final_labels[self._seeds[i]] = self._labels[i]
        
        outputs = container.ndarray(final_labels)

        return base.CallResult(outputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        self._embedding = self._training_inputs[0]

        self._seeds = self._training_inputs[1]
        self._seeds = np.array([int(i) for i in self._seeds])

        self._labels = self._training_outputs
        self._labels = np.array([int(i) for i in self._labels])

        unique_labels, label_counts = np.unique(self._labels, return_counts = True)
        K = len(unique_labels)

        n, d = self._embedding.shape

        if int(K) < d:
            self._embedding = self._embedding[:, :K].copy()
            d = int(K)

        self._ENOUGH_SEEDS = True # For full estimation

        unique_labels, label_counts = np.unique(self._labels, return_counts = True)

        for i in range(K):
            if label_counts[i] < d*(d + 1)/2:
                self._ENOUGH_SEEDS = False
                break

        self._pis = label_counts/len(self._seeds)

        for i in range(len(self._labels)): # reset labels to [0,.., K-1]
            itemindex = np.where(unique_labels==self._labels[i])[0][0]
            self._labels[i] = int(itemindex)

        x_sums = np.zeros(shape = (K, d))

        for i in range(len(self._seeds)):
            temp_feature_vector = self._embedding[int(self._seeds[i]), :]
            temp_label = self._labels[i]
            x_sums[temp_label, :] += temp_feature_vector

        estimated_means = [x_sums[i,:]/label_counts[i] for i in range(K)]

        mean_centered_sums = np.zeros(shape = (K, d, d))

        for i in range(len(seeds)):
            temp_feature_vector = inputs[self._seeds[i], :]
            temp_label = self._labels[i]
            mean_centered_feature_vector = temp_feature_vector - estimated_means[self._labels[i]]
            temp_feature_vector = np.reshape(temp_feature_vector, (len(temp_feature_vector), 1))
            mcfv_squared = temp_feature_vector.dot(temp_feature_vector.T)
            mean_centered_sums[temp_label, :, :] += mcfv_squared
        
        if self._ENOUGH_SEEDS:
            estimated_cov = np.zeros(shape = (K, d, d))
            for i in range(K):
                estimated_cov[i] = mean_centered_sums[i,:]/(label_counts[i] - 1)
        else:
            estimated_cov = np.zeros(shape = (d,d))
            for i in range(K):
                estimated_cov += mean_centered_sums[i, :]*(label_counts[i] - 1)
            estimated_cov = estimated_cov / (n - d)

        self._PD = True
        eps = 0.001
        if self._ENOUGH_SEEDS:
            for i in range(K):
                try:
                    eig_values = np.linalg.svd(estimated_cov[i, :, :])[1]
                    if len(eig_values) > len(eig_values[eig_values > -eps]):
                        self._PD = False
                        break
                except:
                    self._PD = False
                    break

        self._means = container.ndarray(estimated_means)
        self._covariances = container.ndarray(estimated_cov)

        self._fitted = True

        return base.CallResult(None)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._fitted = False

        """
        if len(seeds) == 0: # run EM if no seeds are given
            cov_types = ['full', 'tied', 'diag', 'spherical']

            BIC_max = 0
            cov_type_likelihood_max = ""
            for i in cov_types:
                clf = GaussianMixture(n_components=K, 
                                    covariance_type=i, n_init = 50)
                clf.fit(inputs)

                current_bic = clf.bic(inputs)

                if current_bic > BIC_max:
                    BIC_max = current_bic
                    cov_type_likelihood_max = i

            clf = GaussianMixture(n_components = K,
                            covariance_type = cov_type_likelihood_max)

            clf.fit(inputs)

            predictions = clf.predict(inputs)

            outputs = container.ndarray(predictions)

            return base.CallResult(outputs)
        """