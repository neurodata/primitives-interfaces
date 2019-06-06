#!/usr/bin/env python

# lse.py
# Copyright (c) 2017. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
import numpy as np
import networkx

from scipy.stats import rankdata
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from graspy.embed import LaplacianSpectralEmbed as graspyLSE
from graspy.embed import OmnibusEmbed as graspyOMNI
from graspy.utils import pass_to_ranks as graspyPTR

Inputs = container.List
Outputs = container.List

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    max_dimension = hyperparams.Hyperparameter[int](default=2, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

    which_elbow = hyperparams.Hyperparameter[int](default=1, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

    use_attributes = hyperparams.Hyperparameter[bool](default = False, semantic_types = [
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

class LaplacianSpectralEmbedding(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Spectral-based trasformation of the Laplacian.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '8fa6178b-84f7-37d8-87e8-4d3a44c86569',
        'version': "0.1.0",
        'name': "jhu.lse",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_transformation.laplacian_spectral_embedding.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['laplacian embedding'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/lse/lse.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu'
        },
        'description': 'Spectral-based trasformation of the Laplacian',
        'hyperparams_configuration': {
            'max_dimension': 'The maximum dimension that can be used for eigendecomposition',
            'which_elbow': 'The scree plot "elbow" to use for dimensionality reduction. High values leads to more dimensions selected.',
            'use_attributes': 'Boolean which indicates whether to use the attributes of the nodes.'
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
            "HIGHER_ORDER_SINGULAR_VALUE_DECOMPOSITION"
        ],
        'primitive_family': "DATA_TRANSFORMATION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        np.random.seed(1234)

        G = inputs[0].copy()

        g = graspyPTR(G)

        n = g.shape[0]

        max_dimension = self.hyperparams['max_dimension']

        if max_dimension > n:
            max_dimension = n 

        n_elbows = self.hyperparams['which_elbow']

        """
        What does Omni(DAD) even look like? 

        if self.hyperparams['use_attributes']:
            adj = [g]
            MORE_ATTR = True
            attr_number = 1

            while MORE_ATTR:
                attr = 'attr'
                temp_attr = np.array(list(networkx.get_node_attributes(G, 'attr' + str(attr_number)).values()))
                if len(temp_attr) == 0:
                    MORE_ATTR = False
                else:
                    K = np.sum((temp_attr[:, np.newaxis][:, np.newaxis, :] - temp_attr[:, np.newaxis][np.newaxis, :, :])**2, axis = -1)
                    adj.append(graspyPTR(K))
                    attr_number += 1
            M = len(adj)
            if M > 1:
                g = self._omni(adj)
                lse_object = graspyLSE(n_components = max_dimension, n_elbows=n_elbows)
                X_hats = lse_object.fit_transform(g)

                d = X_hats.shape[1]

                X_hats_reshaped = X_hats.reshape((M, n, d))
                X_hat = np.mean(X_hats_reshaped, axis = 0)

                embedding = X_hat.copy()

                inputs[0] = container.ndarray(embedding)

                return base.CallResult(inputs)
        """

        lse_object = graspyLSE(n_components = max_dimension, n_elbows=n_elbows)
        X_hat = lse_object.fit_transform(g)

        inputs[0] = container.ndarray(X_hat)

        return base.CallResult(inputs)