#!/usr/bin/env python

# ase.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.


import numpy as np
from typing import Sequence, TypeVar, Union, Dict
import networkx
import os
import sys

from scipy.stats import norm
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import utils, container
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from graspy.embed import AdjacencySpectralEmbed as graspyASE
from graspy.embed import OmnibusEmbed as graspyOMNI
from graspy.utils import pass_to_ranks as graspyPTR

Inputs = container.List
Outputs = container.List

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    max_dimension = hyperparams.Bounded[int](
        default=2,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ],
        lower = 1,
        upper = None
    )

    which_elbow = hyperparams.Bounded[int](
        default = 1,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ],
        lower = 1,
        upper = 2
    )
    use_attributes = hyperparams.Hyperparameter[bool](
        default = False,
        semantic_types = [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

class AdjacencySpectralEmbedding(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Spectral-based trasformation of weighted or unweighted adjacency matrix.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'b940ccbd-9e9b-3166-af50-210bfd79251b',
        'version': "0.1.0",
        'name': "jhu.ase",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['ase primitive', 'graph', 'spectral', 'embedding', 'spectral method', 'adjacency', 'matrix'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces',
            ],
            'contact': 'mailto:hhelm2@jhu.edu'
        },
        'description': 'Spectral-based trasformation of weighted or unweighted adjacency matrix',
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
        'python_path': 'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "SINGULAR_VALUE_DECOMPOSITION"
        ],
        'primitive_family': "DATA_TRANSFORMATION",
        'preconditions': ['NO_MISSING_VALUES']
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        np.random.seed(1234)
        #print('ase, baby!', file=sys.stderr)
        G = inputs[1][0].copy()

        try:
            link_predicton = inputs[3]
            if type(link_predicton) is not bool:
                link_predicton = False
        except:
            link_predicton = False

        if link_predicton:
            g = np.array(G.copy())
        else:
            g = graspyPTR(G)

        n = g.shape[0]

        max_dimension = self.hyperparams['max_dimension']

        if max_dimension > n:
            max_dimension = n 

        n_elbows = self.hyperparams['which_elbow']

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
                omni_object = graspyOMNI(n_components = max_dimension, n_elbows = n_elbows)
                X_hats = omni_object.fit_transform(adj)
                X_hat = np.mean(X_hats, axis = 0)

                embedding = X_hat.copy()

                inputs[1][0] = container.ndarray(embedding)

                return base.CallResult(inputs)

        ase_object = graspyASE(n_components=max_dimension, n_elbows = n_elbows)
        X_hat = ase_object.fit_transform(g)

        inputs[1][0] = container.ndarray(X_hat)

        return base.CallResult(inputs)
