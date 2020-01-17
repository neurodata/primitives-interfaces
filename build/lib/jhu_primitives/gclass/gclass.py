#!/usr/bin/env python

# gclass.py
# Copyright (c) 2017. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
import sys

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
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
Outputs = container.DataFrame

class Params(params.Params):
    pis: container.ndarray
    means: container.ndarray
    covariances: container.ndarray
    problem: str
    nodeIDs: np.ndarray
    embedding: container.ndarray
    seeds: container.ndarray
    labels: container.ndarray
    ENOUGH_SEEDS: bool
    PD: bool
    pis: container.ndarray
    means: container.ndarray
    covariances: container.ndarray

class Hyperparams(hyperparams.Hyperparams):
    hp = None

class GaussianClassification(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params,Hyperparams]):
    """
    Quadratic discriminant analysis classification procedure.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'c9d5da5d-0520-468e-92df-bd3a85bb4fac',
        'version': "0.1.0",
        'name': "jhu.gclass",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.classification.gaussian_classification.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['gaussian classification', 'graph', 'graphs', 'classification', 'supervised', 'supervised learning'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/gclass/gclass.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu',
        },
        'description': 'Quadratic discriminant analysis classification procedure',
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
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris': [
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "QUADRATIC_DISCRIMINANT_ANALYSIS"
        ],
        'primitive_family': "CLASSIFICATION",
        'preconditions': ['NO_MISSING_VALUES']
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._fitted: bool = False
        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None
        self._problem: str = ""
        self._nodeIDs : np.ndarray = None
        self._embedding: container.ndarray = None
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

        unique_labels = np.unique(self._labels)
        K = len(unique_labels)

        testing = inputs[0]

        try:
            testing_nodeIDs = np.asarray(testing['G1.nodeID'])
        except:
            testing_nodeIDs = np.asarray(testing['nodeID'])
        final_labels = np.zeros(len(testing))
        string_nodeIDs = np.array([str(i) for i in self._nodeIDs])
        #print(string_nodeIDs, file=sys.stderr)
        #print(testing_nodeIDs, file=sys.stderr)
        if self._PD and self._ENOUGH_SEEDS:
            for i in range(len(testing_nodeIDs)):
                temp = np.where(string_nodeIDs == str(testing_nodeIDs[i]))[0][0]
                weighted_pdfs = np.array([self._pis[j]*MVN.pdf(self._embedding[temp,:], self._means[j], self._covariances[j, :, :]) for j in range(K)])
                label = np.argmax(weighted_pdfs)
                final_labels[i] = int(label)
        else:

            for i in range(len(testing_nodeIDs)):
                temp = np.where(string_nodeIDs == str(testing_nodeIDs[i]))[0][0]
                try:
                    weighted_pdfs = np.array([self._pis[j]*MVN.pdf(self._embedding[temp,:], self._means[j], self._covariances) for j in range(K)])
                except:
                    self._covariances += self._covariances + np.ones(self._covariances.shape)*0.00001
                    weighted_pdfs = np.array([self._pis[j]*MVN.pdf(self._embedding[temp,:], self._means[j], self._covariances) for j in range(K)])
                label = np.argmax(weighted_pdfs)
                final_labels[i] = int(label)

        if self._problem == "VN":
            testing['classLabel'] = final_labels
            outputs = container.DataFrame(testing[['d3mIndex','classLabel']])
            outputs[['d3mIndex', 'classLabel']] = outputs[['d3mIndex', 'classLabel']].astype(int)
        else:
            testing['community'] = final_labels
            outputs = container.DataFrame(testing[['d3mIndex', 'community']])
            outputs[['d3mIndex', 'community']] = outputs[['d3mIndex', 'community']].astype(int)

        return base.CallResult(outputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        self._embedding = self._training_inputs[1][0]

        self._nodeIDs = np.array(self._training_inputs[2])

        try:
            self._seeds = self._training_inputs[0]['G1.nodeID']
        except:
            self._seeds = self._training_inputs[0]['nodeID'].astype(float).astype(int)

        self._seeds = np.array([int(i) for i in self._seeds])

        try:
            self._labels = self._training_inputs[0]['classLabel']
            self._problem = 'VN'

        except:
            self._labels = self._training_inputs[0]['community']
            self._problem = 'CD'

        self._labels = np.array([int(i) for i in self._labels])

        unique_labels, label_counts = np.unique(self._labels, return_counts = True)
        K = len(unique_labels)

        n, d = self._embedding.shape

        if int(K) < d:
            self._embedding = self._embedding[:, :K].copy()
            d = int(K)

        self._ENOUGH_SEEDS = True # For full estimation

        # get unique labels
        unique_labels, label_counts = np.unique(self._labels, return_counts = True)
        for i in range(K):
            if label_counts[i] < d*(d + 1)/2:
                self._ENOUGH_SEEDS = False
                break

        self._pis = label_counts/len(self._seeds)


        # reindex labels if necessary
        for i in range(len(self._labels)): # reset labels to [0,.., K-1]
            itemindex = np.where(unique_labels==self._labels[i])[0][0]
            self._labels[i] = int(itemindex)


        # gather the means
        x_sums = np.zeros(shape = (K, d))

        estimated_means = np.zeros((K, d))
        for i in range(K):
            temp_seeds = self._seeds[np.where(self._labels == i)[0]]
            estimated_means[i] = np.mean(self._embedding[temp_seeds], axis=0)
        #for i in range(len(self._seeds)):
        #    nodeID = np.where(self._nodeIDs == self._seeds[i])[0][0]
        #    temp_feature_vector = self._embedding[nodeID, :]
        #    temp_label = self._labels[i]
        #    x_sums[temp_label, :] += temp_feature_vector

        #estimated_means = [x_sums[i,:]/label_counts[i] for i in range(K)]

        mean_centered_sums = np.zeros(shape = (K, d, d))

        covs = np.zeros(shape = (K, d, d))
        for i in range(K):
            feature_vectors = self._embedding[self._seeds[self._labels == i], :]
            covs[i] = np.cov(feature_vectors, rowvar = False)

        if self._ENOUGH_SEEDS:
            estimated_cov = covs
        else:
            estimated_cov = np.zeros(shape = (d,d))
            for i in range(K):
                estimated_cov += covs[i]*(label_counts[i] - 1)
            estimated_cov = estimated_cov / (n - K)

        self._PD = True

        self._means = container.ndarray(estimated_means)
        self._covariances = container.ndarray(estimated_cov)

        self._fitted = True

        return base.CallResult(None)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")

        return Params(
            pis = self._pis,
            means= self._means,
            covariances = self._covariances,
            problem = self._problem,
            nodeIDs = self._nodeIDs,
            embedding = self._embedding,
            seeds = self._seeds,
            labels= self._labels,
            ENOUGH_SEEDS = self._ENOUGH_SEEDS,
            PD = self._PD
        )


    def set_params(self, *, params: Params) -> None:
        self._fitted = True
        self._pis = params['pis']
        self._means= params['means']
        self._covariances = params['covariances']
        self._problem = params['problem']
        self._nodeIDs = params['nodeIDs']
        self._embedding = params['embedding']
        self._seeds = params['seeds']
        self._labels = params['labels']
        self._ENOUGH_SEEDS = params['ENOUGH_SEEDS']
        self._PD = params['PD']
