#!/usr/bin/env python

# gclass.py
# Copyright (c) 2020. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
import sys

from d3m import utils
from d3m import container
from d3m import exceptions
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
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
    lcc_labels: container.ndarray
    unique_labels: container.ndarray
    unique_lcc_labels: container.ndarray
    ENOUGH_SEEDS: bool
    PD: bool
#    pis: container.ndarray
#    means: container.ndarray
#    covariances: container.ndarray

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
        self._lcc_labels: container.ndarray = None
        self._unique_labels = None
        self._unique_lcc_labels = None
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
        # print("gclass produce started", file=sys.stderr)

        if not self._fitted:
            raise ValueError("Not fitted")

        learning_data = inputs[0]
        headers=learning_data.columns

        for col in headers:
            if "node" in col:
                testing_nodeIDs = learning_data[col]
            if "Label" in col or "label" in col or "class" in col:
                LABEL = col

        n = self._embedding.shape[0]
        K = len(self._unique_labels)

        final_labels = np.zeros(len(learning_data))
        string_nodeIDs = container.ndarray(np.array([str(i) for i in self._nodeIDs]))
        for i in range(len(testing_nodeIDs)):
            if testing_nodeIDs[i] in self._nodeIDs:
                temp = np.where(self._nodeIDs == str(testing_nodeIDs[i]))[0][0]
                likelihoods = np.array([MVN.pdf(self._embedding[temp,:],
                                                self._means[j],
                                                self._covariances[j, :, :])
                                        for j in range(K)])
                posteriors = self._pis * likelihoods
                label = np.argmax(posteriors)
                final_labels[i] = self._unique_labels[int(label)]
            else:
                final_labels[i] = self._unique_labels[np.argmax(self._pis)]

        learning_data[LABEL] = final_labels
        outputs = container.DataFrame(learning_data[['d3mIndex',LABEL]])
        outputs[['d3mIndex', LABEL]] = outputs[['d3mIndex', LABEL]].astype(int)

        debugging = False
        if debugging:
            print("proportions: {}".format(self._pis), file=sys.stderr)
            print("final labels: {}".format(final_labels), file=sys.stderr)
        # print("gclass produce ended", file=sys.stderr)

        return base.CallResult(outputs)

    def fit(self, *,
            timeout: float = None,
            iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        # print("gclass fit started", file=sys.stderr)

        # unpack training inputs
        self._embedding = self._training_inputs[1][0]
        self._nodeIDs = np.array(self._training_inputs[2][0])
        learning_data = self._training_inputs[0]
        headers = learning_data.columns

        # take seeds and their labels from the learning data
        for col in headers:
            if "node" in col:
                self._seeds = container.ndarray(np.array(list(learning_data[col])))
            if "Label" in col or "label" in col or "class" in col:
                self._labels = container.ndarray(np.array(list(learning_data[col])))

        # subselect seeds and labels that are in the lcc
        lcc_seeds = []
        lcc_labels = []
        for seed, label in zip(self._seeds, self._labels):
            if seed in self._nodeIDs:
                lcc_seeds.append(seed)
                lcc_labels.append(label)

        # cast to d3m appropriate types
        self._lcc_seeds = container.ndarray(lcc_seeds)
        self._lcc_labels = container.ndarray(lcc_labels)

        # get unique labels
        unique_labels, label_counts = np.unique(self._labels,
                                                 return_counts = True)
        unique_lcc_labels, lcc_label_counts = np.unique(self._lcc_labels,
                                                        return_counts = True)
        # cast to d3m appropriate types
        self._unique_labels = container.ndarray(unique_labels)
        self._unique_lcc_labels = container.ndarray(unique_lcc_labels)

        debugging = False
        if debugging:
            print("shape of the embedding: {}".format(self._embedding.shape),
                  file=sys.stderr)
            print("length of the seeds: {}".format(len(self._seeds)),
                  file=sys.stderr)
            print("length of the labels: {}".format(len(self._labels)),
                  file=sys.stderr)
            print("unique labels: {}".format(self._unique_labels),
                  file=sys.stderr)
            print("label counts: {}".format(label_counts),
                  file=sys.stderr)
            print("lenth of the lcc_seeds: {}".format(len(self._lcc_seeds)),
                  file=sys.stderr)
            print("lenth of the lcc_labels: {}".format(len(self._lcc_labels)),
                  file=sys.stderr)
            print("unique lcc labels: {}".format(self._unique_lcc_labels),
                  file=sys.stderr)
            print("lcc label counts: {}".format(lcc_label_counts),
                  file=sys.stderr)
            print("label types: {}".format(type(self._labels[0])),
                  file=sys.stderr)

        if np.all(self._unique_labels != self._unique_lcc_labels):
            raise exceptions.NotSupportedError(
                'nodes from some classes are not present in the lcc; ' + 
                'the problem is ill-defined')

        n, d = self._embedding.shape
        K = len(self._unique_labels)

        if int(K) < d:
            self._embedding = self._embedding[:, :K].copy()
            d = int(K)

        # heuristically check if we have enough datapoints in all classes
        # if we do - perform qda. else - lda.
        # maybe this should use label counts in the lcc only..
        self._ENOUGH_SEEDS = True
        for i in range(K):
            if label_counts[i] < d*(d + 1)/2:
                self._ENOUGH_SEEDS = False
                break
        self._ENOUGH_SEEDS = False

        # prior probabilities estimation (note that they are global, not lcc)
        self._pis = container.ndarray(np.array(label_counts/len(self._seeds)))
        
        debugging = False
        if debugging:
            print("prior probabilities: {}".format(self._pis),
                  file=sys.stderr)
            print("sum of prior probabilities: {}".format(np.sum(self._pis)),
                  file=sys.stderr)

        # MY VERSION #
        # estimate means and covariances
        estimated_means = np.zeros((K, d))
        estimated_covs = np.zeros((K, d, d))
        for i, lab in enumerate(self._unique_lcc_labels):
            temp_seeds = self._seeds[np.where(self._lcc_labels == lab)[0]]
            temp_seeds_mask = np.isin(self._nodeIDs, temp_seeds)
            feature_vectors = self._embedding[temp_seeds_mask]
            estimated_means[i] = np.mean(feature_vectors, axis=0)
            estimated_covs[i] = np.cov(feature_vectors, rowvar = False)
        self._means = container.ndarray(estimated_means)
        # use 'pooled covariance' if we are using lda
        if not self._ENOUGH_SEEDS:
            pooled_cov = np.sum(
                estimated_covs * (label_counts - 1).reshape(-1, 1, 1),
                axis=0) / (n - K)
            estimated_covs = np.repeat(pooled_cov.reshape(1, d, d), K, axis=0)
        self._covariances = container.ndarray(estimated_covs)
        self._PD = True

        self._fitted = True

        # print("gclass fit ended", file=sys.stderr)

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
            lcc_labels= self._lcc_labels,
            unique_labels= self._unique_labels,
            unique_lcc_labels= self._unique_lcc_labels,
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
        self._lcc_labels = params['lcc_labels']
        self._unique_labels = params['unique_labels']
        self._unique_lcc_labels = params['unique_lcc_labels']
        self._ENOUGH_SEEDS = params['ENOUGH_SEEDS']
        self._PD = params['PD']
