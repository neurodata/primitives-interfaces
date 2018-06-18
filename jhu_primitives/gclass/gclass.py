#!/usr/bin/env python

# gclass.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
import numpy as np
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as MVN

Inputs = container.ndarray
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    number_of_clusters = hyperparams.Hyperparameter[int](default = 2,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    seeds = hyperparams.Hyperparameter[np.ndarray](default=np.array([]), semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    labels = hyperparams.Hyperparameter[np.ndarray](default=np.array([]), semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

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
        'keywords': ['gaussian classification'],
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
            "HIGHER_ORDER_SINGULAR_VALUE_DECOMPOSITION"
        ],
        'primitive_family': "DATA_TRANSFORMATION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Inputs
            D - An n x d feature numpy array
        Returns
            labels - Class labels for each unlabeled vertex
        """
        
        K = self.hyperparams['number_of_clusters']
        seeds = self.hyperparams['seeds']
        labels = self.hyperparams['labels']

        n, d = inputs.shape

        if len(seeds) == 0: # run EM if no seeds are given
            cov_types = ['full', 'tied', 'diag', 'spherical']

            BIC_max = 0
            cov_type_likelihood_max = ""
            for i in cov_types:
                clf = GaussianMixture(n_components=K, 
                                    covariance_type=i)
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

        unique_labels, label_counts = np.unique(labels, return_counts = True)

        pi = label_counts/len(seeds)

        for i in range(len(labels)): # reset labels to [0,.., K-1]
            itemindex = np.where(unique_labels==labels[i])[0][0]
            labels[i] = int(itemindex)

        x_sums = np.zeros(shape = (K, d))

        for i in range(len(seeds)):
            temp_feature_vector = inputs[int(seeds[i]), :]
            temp_label = labels[i]
            x_sums[temp_label, :] += temp_feature_vector

        estimated_means = [x_sums[i,:]/label_counts[i] for i in range(K)]

        mean_centered_sums = np.zeros(shape = (K, d, d))

        for i in range(len(seeds)):
            temp_feature_vector = inputs[int(seeds[i]), :]
            temp_label = labels[i]
            mean_centered_feature_vector = temp_feature_vector - estimated_means[temp_label]
            mcfv_squared = temp_feature_vector.T.dot(temp_feature_vector)
            mean_centered_sums[temp_label, :, :] += mcfv_squared

        estimated_cov = np.zeros(shape = (K, d, d))
        for i in range(K):
            estimated_cov[i] = mean_centered_sums[i,:,:]/(label_counts[i] - 1)

        final_labels = np.zeros(n)

        print(estimated_cov.shape, type(estimated_cov))

        for i in range(n):
            if i not in seeds:
                weighted_pdfs = np.zeros(K)
                for j in range(K):
                    try:
                        #weighted_pdfs = [pi[j]*MVN(inputs[i,:], estimated_means[j], estimated_cov[j, :, :]) for j in range(K)]
                        weighted_pdf_temp = pi[j]*MVN(inputs[i,:], estimated_means[j], estimated_cov[j, :, :]) # avoiding non-PD covariances...
                        weighted_pdfs[j] = weighted_pdf_temp
                        label = argmax(weighted_pdfs)
                        final_labels[i] = label
                    except:
                        weighted_pdfs[j] = 0

            else:
                final_labels[i] = labels[i]

        print(final_labels)

        outputs = container.ndarray(final_labels)

        print("supervised return")

        return base.CallResult(outputs)