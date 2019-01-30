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

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult


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

    n_in_sample = hyperparams.Bounded[int](
        default = 1000,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ],
        lower = 1,
        upper = None
    )

class OutOfSampleAdjacencySpectralEmbedding(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Out of sample nodes embedded using out of sample adjacency spectral embedding.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '8e1f9112-c40b-4ed1-8820-7c88a3353a1d',
        'version': "0.1.0",
        'name': "jhu.oosase",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_transformation.out_of_sample_adjacency_spectral_embedding.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'embedding'],
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
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris': [
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "SINGULAR_VALUE_DECOMPOSITION"
        ],
        'description' : "Out of sample nodes embedded using out of sample adjacency spectral embedding.",
        'primitive_family': "DATA_TRANSFORMATION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)


    def _profile_likelihood_maximization(self,U, n_elbows):
        """
        Inputs
            U - An ordered or unordered list of eigenvalues
            n - The number of elbows to return

        Return
            elbows - A numpy array containing elbows
        """
        if type(U) == list:  # cast to array for functionality later
            U = np.array(U)

        if n_elbows == 0:  # nothing to do..
            return np.array([])

        if U.ndim == 2:
            U = np.std(U, axis=0)

        if len(U) == 0:
            return np.array([])

        elbows = []

        if len(U) == 1:
            return np.array(elbows.append(U[0]))

        U.sort()  # sort
        U = U[::-1]  # reverse array so that it is sorted in descending order
        n = len(U)

        while len(elbows) < n_elbows and len(U) > 1:
            d = 1
            sample_var = np.var(U, ddof=1)
            sample_scale = sample_var ** (1 / 2)
            elbow = 0
            likelihood_elbow = -100000000
            while d < len(U):
                mean_sig = np.mean(U[:d])
                mean_noise = np.mean(U[d:])
                sig_likelihood = 0
                noise_likelihood = 0
                for i in range(d):
                    sig_likelihood += np.log(norm.pdf(U[i], mean_sig, sample_scale))
                for i in range(d, len(U)):
                    noise_likelihood += np.log(norm.pdf(U[i], mean_noise, sample_scale))

                likelihood = noise_likelihood + sig_likelihood
                if likelihood > likelihood_elbow:
                    likelihood_elbow = likelihood
                    elbow = d
                d += 1
            if len(elbows) == 0:
                elbows.append(elbow)
            else:
                elbows.append(elbow + elbows[-1])
            U = U[elbow:]

        if len(elbows) == n_elbows:
            return np.array(elbows)

        if len(U) == 0:
            return np.array(elbows)
        else:
            elbows.append(n)
            return np.array(elbows)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Perform Out of Sample Adjacency Spectral Embedding on a graph.
        """
        np.random.seed(1234)

        g = inputs[0].copy()
        if type(g) == networkx.classes.graph.Graph:
            g = networkx.to_numpy_array(g)

        n = g.shape[0]

        in_sample_n = self.hyperparams['n_in_sample']

        if self.hyperparams['max_dimension'] >= in_sample_n:
            self.hyperparams['max_dimension'] = in_sample_n - 1
        
        d_max = self.hyperparams['max_dimension']

        if in_sample_n > n:
            in_sample_n = n
            # TODO ASE HERE
        in_sample_idx = np.random.choice(n, in_sample_n)
        out_sample_idx = np.setdiff1d(list(range(n)), in_sample_idx)

        in_sample_A = g[np.ix_(in_sample_idx, in_sample_idx)]
        out_sample_A = g[np.ix_(out_sample_idx, in_sample_idx)]

        # hp_ase = ase_hyperparameters({'max_dimension': dim, 'use_attributes': False, 'which_elbow': self.hyperparams['which_elbow']})
        # ASE = ase(hyperparams = hp_ase)
        # embedding = ASE.produce(inputs = [g]).value[0]

        tsvd = TruncatedSVD(n_components = d_max)
        tsvd.fit(in_sample_A)

        eig_vectors = tsvd.components_.T
        eig_values = tsvd.singular_values_

        elbow = self._profile_likelihood_maximization(eig_values, self.hyperparams['which_elbow'])[-1]

        eig_vectors = eig_vectors[:, :elbow + 1].copy()
        eig_values = eig_values[:elbow + 1].copy()
        d = len(eig_values)

        in_sample_embedding = eig_vectors.dot(np.diag(eig_values**0.5))

        out_sample_embedding = out_sample_A @ eig_vectors @ np.diag(1/np.sqrt(eig_values))
        embedding = np.zeros((n,d))
        embedding[in_sample_idx] = in_sample_embedding
        embedding[out_sample_idx] = out_sample_embedding

        inputs[0] = container.ndarray(embedding)

        return base.CallResult(inputs)
