#!/usr/bin/env python

# numclust.py
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

Inputs = container.ndarray
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    #hp = hyperparams.Hyperparameter[None](default = None)
    max_clusters = hyperparams.Hyperparameter[int](default = 2,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

class NumberOfClusters(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'be28f722-6a53-3f70-8781-bd3666946264',
        'version': "0.1.0",
        'name': "jhu.numclust",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.NumberOfClusters',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['number clustering', 'gaussian clustering', 'model selection', 'clusters', 'community', 'clustering', 'cluster selection'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/numclust/numclust.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
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
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris': [
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "LOW_RANK_MATRIX_APPROXIMATIONS"
        ],
        'primitive_family': "GRAPH_CLUSTERING"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Inputs
            D - n x d feature matrix
        Return
            An array with the max BIC and AIC values for each number of clusters (1, .., max_clusters)
        """

        max_clusters = self.hyperparams['max_clusters']

        cov_types = ['full', 'tied', 'diag', 'spherical']

        example = ('number of clusters', 'BIC', 'AIC')

        BICs = []

        AICs = []

        results = [example]

        for i in range(1, max_clusters + 1):

            clf = GaussianMixture(n_components=i, 
                                    covariance_type='spherical')
            clf.fit(inputs)
            temp_max_BIC, temp_max_AIC = -clf.bic(inputs), -clf.aic(inputs)
            for k in cov_types:
                clf = GaussianMixture(n_components=i, 
                                    covariance_type=k)

                clf.fit(inputs)

                temp_BIC, temp_AIC = -clf.bic(inputs), -clf.aic(inputs)

                if temp_BIC > temp_max_BIC:
                    temp_max_BIC = temp_BIC

                if temp_AIC > temp_max_AIC:
                    temp_max_AIC = temp_AIC

            results.append((i, temp_max_BIC, temp_max_AIC))
            BICs.append(temp_max_BIC)
            AICs.append(temp_max_AIC)

        BICs2 = list(BICs)
        AICs2 = list(AICs)

        KBICs = []

        KAICs = []

        while len(BICs2) > 0:
            temp, temp_index = max(BICs2), np.argmax(BICs2)
            index = BICs.index(temp)
            KBICs.append(index + 1)
            BICs2.pop(temp_index)

            temp, temp_index = max(AICs2), np.argmax(AICs2)
            index = AICs.index(temp)
            KAICs.append(index + 1)
            AICs2.pop(temp_index)

        KBICs = ['Ranked number of clusters (BIC)'] + KBICs
        KAICs = ['Ranked number of clusters (AIC)'] + KAICs

        outputs = [KBICs, KAICs, results]

        return base.CallResult(outputs)