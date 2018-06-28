#!/usr/bin/env python

# lse.py
# Copyright (c) 2017. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import numpy as np
import networkx

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from ..utils.util import file_path_conversion

Inputs = container.List
Outputs = container.List

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    d_max = hyperparams.Hyperparameter[int](default=100, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    which_elbow = hyperparams.Hyperparameter[int](default=2, semantic_types=
    ['https://metadata.datadrivendiscovery.org/types/TuningParameter'
     ])

class LaplacianSpectralEmbedding(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '8fa6178b-84f7-37d8-87e8-4d3a44c86569',
        'version': "0.3.0",
        'name': "jhu.lse",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.LaplacianSpectralEmbedding',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['laplacian embedding'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/lse/lse.py',
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

    def _profile_likelihood_maximization(self,U, n_elbows, threshold):
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

        # select values greater than the threshold
        U.sort()  # sort
        U = U[::-1]  # reverse array so that it is sorted in descending order
        n = len(U)

        while len(elbows) < n_elbows and len(U) > 1:
            d = 1
            sample_var = np.var(U, ddof=1)
            sample_scale = sample_var ** (1 / 2)
            elbow = 0
            likelihood_elbow = 0
            while d < len(U):
                mean_sig = np.mean(U[:d])
                mean_noise = np.mean(U[d:])
                sig_likelihood = 0
                noise_likelihood = 0
                for i in range(d):
                    sig_likelihood += norm.pdf(U[i], mean_sig, sample_scale)
                for i in range(d, len(U)):
                    noise_likelihood += norm.pdf(U[i], mean_noise, sample_scale)

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
#    def embed(self, *, g : JHUGraph, dim: int):

        G = inputs[0]

        if type(G) == networkx.classes.graph.Graph:
            if networkx.is_weighted(G):
                G = self._pass_to_ranks(G)
        elif type(G) is np.ndarray:
            G = networkx.to_networkx_graph(G)
            G = self._pass_to_ranks(G)
        else:
            return

        A = robjects.Matrix(G)
        robjects.r.assign("A", A)

        d_max = self.hyperparams['d_max']

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "lse.interface.R")
        path = file_path_conversion(path, uri = "")

        cmd = """
        source("%s")
        fn <- function(inputs, embedding_dimension) {
            lse.interface(inputs, embedding_dimension)
        }
        """ % path


        result = robjects.r(cmd)(A, d_max)
        eig_values = container.ndarray(result[1])

        d = self._get_elbows(eigenvalues=eig_values)
        vectors = container.ndarray(result[0])[:,0:d]

        return base.CallResult(container.List([vectors]))


    def _get_elbows(self, eigenvalues):
        elbows = self._profile_likelihood_maximization(U=eigenvalues
                                                   , n_elbows=self.hyperparams['which_elbow']
                                                   )
        return (elbows[- 1])

    def _pass_to_ranks(self,G):
        #iterates through edges twice

        #initialize edges
        edges = np.repeat(0,networkx.number_of_edges(G))

        #loop over the edges and store in an array
        j = 0
        for u, v, d in G.edges(data=True):
            edges[j] = d['weight']
            j += 1


        #grab the number of edges
        nedges = edges.shape[0]
        #ranked_values = np.argsort(edges) #get the index of the sorted elements
        #ranked_values = np.argsort(ranked_values) + 1
        ranked_values = rankdata(edges)

        #loop through the edges and assign the new weight:
        j = 0
        for u, v, d in G.edges(data=True):
            edges[j] = ranked_values[j]*2/(nedges + 1)
            d['weight'] = edges[j]
            j += 1

        return networkx.to_numpy_array(G)
