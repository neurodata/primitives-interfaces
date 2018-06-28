#!/usr/bin/env python

# ase.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.


import numpy as np
from typing import Sequence, TypeVar, Union, Dict
import networkx
import igraph
import os
import networkx
from scipy.stats import norm
from scipy.stats import rankdata
from rpy2 import robjects
import rpy2.robjects.numpy2ri
robjects.numpy2ri.activate()

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import utils, container
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from ..utils.util import file_path_conversion


Inputs = container.List
Outputs = container.List

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    max_dimension = hyperparams.Hyperparameter[int](default=100, semantic_types= [
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

    which_elbow = hyperparams.Hyperparameter[int](default = 2, semantic_types= [
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

    use_attributes = hyperparams.Hyperparameter[bool](default = False, semantic_types = [
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

class AdjacencySpectralEmbedding(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'b940ccbd-9e9b-3166-af50-210bfd79251b',
        'version': "0.3.0",
        'name': "jhu.ase",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['ase primitive', 'graph', 'spectral', 'embedding', 'spectral method', 'adjacency', 'matrix'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [
            {
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
        'python_path': 'd3m.primitives.jhu_primitives.AdjacencySpectralEmbedding',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "HIGHER_ORDER_SINGULAR_VALUE_DECOMPOSITION"
        ],
        'primitive_family': "DATA_TRANSFORMATION"
    })

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

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        G = inputs[0]
        if type(G) == networkx.classes.graph.Graph:
            if networkx.is_weighted(G):
                g = self._pass_to_ranks(G)
        elif type(G) is np.ndarray:
            G = networkx.to_networkx_graph(G)
            g = self._pass_to_ranks(G)
        else:
            raise ValueError("networkx Graph and n x d numpy arrays only")

        if use_attributes:
            adj = [g]
            MORE_ATTR = True
            attr_number = 1
            while MORE_ATTR:
                temp_attr = np.array(list(networkx.get_node_attributes(G, 'attr' + attr_number).values()))
                if len(temp_attr) == 0:
                    MORE_ATTR = False
                else:
                    adj.append(temp_attr)
                    attr_number += 1
            for i in range(1, len(adj)):
                adj[i] = self._pass_to_ranks(self, G, matrix = True)

            g = self._omni(adj)

        A = robjects.Matrix(g)
        robjects.r.assign("A", A)

        d_max = self.hyperparams['max_dimension']

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "ase.interface.R")
        path = file_path_conversion(path, uri = "")
        
        cmd = """
        source("%s")
        fn <- function(inputs, d_max) {
            ase.interface(inputs, d_max)
        }
        """ % path

        result = robjects.r(cmd)(A, d_max)
        eig_values = container.ndarray(result[1])

        d = self._get_elbows(eigenvalues=eig_values)
        vectors = container.ndarray(result[0])[:,0:d]

        return base.CallResult(container.List([vectors]))

    def _get_elbows(self,  eigenvalues):
        elbows = self._profile_likelihood_maximization(U=eigenvalues
                        , n_elbows=self.hyperparams['which_elbow']
                       )
        return(elbows[-1])
        
    def _pass_to_ranks(self, G, matrix = False):
        #iterates through edges twice

        #initialize edges
        if not matrix:
            edges = np.repeat(0,networkx.number_of_edges(G))

            #loop over the edges and store in an array
            j = 0
            for u, v, d in G.edges(data=True):
                edges[j] = d['weight']
                j += 1


            #grab the number of edges
            nedges = networkx.number_of_edges(G)
            #ranked_values = np.argsort(edges) #+ 1#get the index of the sorted elements
            #ranked_values = np.argsort(ranked_values) + 1
            ranked_values = rankdata(edges)
            #loop through the edges and assign the new weight:
            j = 0
            for u, v, d in G.edges(data=True):
                edges[j] = (ranked_values[j]*2)/(nedges + 1)
                d['weight'] = edges[j]
                j += 1

            return networkx.to_numpy_array(G)
        else:
            n = len(G)
            similarity_mat = numpy.zeros(shape = (n, n))
            for i in range(n):
                for k in range(i + 1, n):
                    temp = -np.sqrt((G[i] - G[k])**2)
                    similarity_mat[i,k] = np.exp(temp)
                    similarity_mat[k,i] = similarity_mat[i,k]
            unraveled_sim = similarity_sim.ravel()
            sorted_indices = numpy.argsort(unraveled_sim)
            E = int((n**2 - n)/2) # or E = int(len(single)/a1_sim.shape[0]) 
            for i in range(E):
                unraveled_sim[sorted_indices[(n - 2) + 2*(i + 1)]] = i/E
                unraveled_sim[sorted_indices[(n - 2) + 2*(i + 1) + 1]] = i/E
            ptr = unraveled_sim.reshape((n,n))
            return ptr

    def _ommni(self, list_of_sim_matrices):
        """
        Inputs
            list_of_sim_matrices - The adjacencies to create the omni for

        Returns
            omni - The omni of the adjacency matrix and its attributes
        """

        adj = [G]


        omni = zeros(shape = (300,300))

        for i in range(len(adj)):
            for j in range(i, len(adj)):
                for k in range(A.shape[0]):
                    for m in range(k + 1, A.shape[1]):
                        if i == j:
                            omni[i*100 + k, j*100 + m] = adj[i][k, m] 
                            omni[j*100 + m, i*100 + k] = adj[i][k, m] # symmetric
                        else:
                            omni[i*100 + k, j*100 + m] = (adj[i][k,m] + adj[j][k,m])/2
                            omni[j*100 + m, i*100 + k] = (adj[i][k,m] + adj[j][k,m])/2

