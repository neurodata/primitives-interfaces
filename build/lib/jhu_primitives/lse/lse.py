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

from ..utils.util import file_path_conversion

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
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/lse/lse.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu'
        },
        'description': 'Spectral-based trasformation of the Laplacian',
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

        U = U[:self.hyperparams['max_dimension']].copy()

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
        np.random.seed(1234)

        G = inputs[0].copy()

        if type(G) == networkx.classes.graph.Graph:
            if networkx.is_weighted(G):
                E = int(networkx.number_of_edges(G))
                g = self._pass_to_ranks(G, nedges = E)
            else:
                E = int(networkx.number_of_edges(G))
                g = networkx.to_numpy_array(G)
        elif type(G) is np.ndarray:
            G = networkx.to_networkx_graph(G)
            E = int(networkx.number_of_edges(G))
            g = self._pass_to_ranks(G, nedges = E)
        else:
            raise ValueError("networkx Graph and n x d numpy arrays only")

        n = g.shape[0]

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
                    adj.append(temp_attr)
                    attr_number += 1
            for i in range(1, len(adj)):
                adj[i] = self._pass_to_ranks(adj[i], nedges = E, matrix = True)

            if len(adj) > 1:
                g = self._omni(adj)
                D = np.linalg.pinv(np.diag(g.sum(axis=1))**(1/2))
                L = 1/np.sqrt(D) @ G @ 1/np.sqrt(D)

                M = len(adj)

                tsvd = TruncatedSVD(n_components = self.hyperparams['max_dimension'])
                tsvd.fit(L)

                eig_vectors = tsvd.components_.T
                eig_values = tsvd.singular_values_

                d = self._get_elbows(eigenvalues=eig_values)

                X_hat = eig_vectors_copy.dot(np.diag(eig_values**0.5))
                avg = np.zeros(shape = (n, d))

                for i in range(M):
                    for j in range(n):
                        avg[j] += X_hat[i*n + j]
                for j in range(n):
                    avg[j, :] = avg[j,:]/M

                embedding = avg.copy()

                inputs[0] = container.ndarray(embedding)

                return base.CallResult(inputs)

        D = np.linalg.pinv(np.diag(g.sum(axis=1))**(1/2))

        L = D @ g @ D

        d_max = self.hyperparams['max_dimension']

        tsvd = TruncatedSVD(n_components = d_max)
        tsvd.fit(L)

        eig_vectors = tsvd.components_.T
        eig_values = tsvd.singular_values_

        eig_vectors_copy = eig_vectors[:,:].copy()

        X_hat = eig_vectors_copy.dot(np.diag(eig_values**0.5))

        inputs[0] = container.ndarray(X_hat)

        return base.CallResult(inputs)

    def _get_elbows(self, eigenvalues):
        elbows = self._profile_likelihood_maximization(U=eigenvalues
                                                   , n_elbows=self.hyperparams['which_elbow']
                                                   )
        return (elbows[- 1])

    def _pass_to_ranks(self, G, nedges = networkx.number_of_edges, matrix = False):
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
            #nedges = networkx.number_of_edges(G)

            #ranked_values = np.argsort(edges) #+ 1#get the index of the sorted elements
            #ranked_values = np.argsort(ranked_values) + 1

            ranked_values = rankdata(edges)
            #loop through the edges and assign the new weight:
            j = 0
            for u, v, d in G.edges(data=True):
                #edges[j] = ranked_values[j]*2/(nedges + 1)
                #d['weight'] = edges[j]
                d['weight'] = ranked_values[j]*2/(nedges + 1)
                j += 1

            return networkx.to_numpy_array(G)

        else:
            n = len(G)
            similarity_mat = np.zeros(shape = (n, n))
            for i in range(n):
                for k in range(i + 1, n):
                    temp = -np.sqrt((G[i] - G[k])**2)
                    similarity_mat[i,k] = np.exp(temp)
                    similarity_mat[k,i] = similarity_mat[i,k]
            unraveled_sim = similarity_mat.ravel()
            sorted_indices = np.argsort(unraveled_sim)

            if nedges == 0:
                E = int((n**2 - n)/2) # or E = int(len(single)/a1_sim.shape[0])
                for i in range(E):
                    unraveled_sim[sorted_indices[(n - 2) + 2*(i + 1)]] = i/E
                    unraveled_sim[sorted_indices[(n - 2) + 2*(i + 1) + 1]] = i/E

            else:
                for i in range(nedges):
                    unraveled_sim[sorted_indices[-2*i - 1]] = (nedges - i)/nedges
                    unraveled_sim[sorted_indices[-2*i - 2]] = (nedges - i)/nedges

                for i in range(n**2 - int(2*nedges)):
                    unraveled_sim[sorted_indices[i]] = 0

            ptred = unraveled_sim.reshape((n,n))
            return ptred

    def _omni(self, list_of_sim_matrices):
        """
        Inputs
            list_of_sim_matrices - The adjacencies to create the omni for

        Returns
            omni - The omni of the adjacency matrix and its attributes
        """

        M = len(list_of_sim_matrices)
        n = len(list_of_sim_matrices[0])
        omni = np.zeros(shape = (M*n, M*n))

        for i in range(M):
            for j in range(i, M):
                for k in range(n):
                    for m in range(k + 1, n):
                        if i == j:
                            omni[i*n + k, j*n + m] = list_of_sim_matrices[i][k, m]
                            omni[j*n + m, i*n + k] = list_of_sim_matrices[i][k, m] # symmetric
                        else:
                            omni[i*n + k, j*n + m] = (list_of_sim_matrices[i][k,m] + list_of_sim_matrices[j][k,m])/2
                            omni[j*n + m, i*n + k] = (list_of_sim_matrices[i][k,m] + list_of_sim_matrices[j][k,m])/2
        return omni
