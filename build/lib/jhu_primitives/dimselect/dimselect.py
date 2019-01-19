#!/usr/bin/env python

# dimselect.py
# Copyright (c) 2017. All rights reserved.


from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
import numpy as np
from d3m import utils, container
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from jhu_primitives.utils.util import file_path_conversion
from .. import AdjacencySpectralEmbedding
import networkx
from scipy.stats import norm

Inputs = container.ndarray
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    n_elbows = hyperparams.Hyperparameter[int](default=3, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    error_threshold = hyperparams.Hyperparameter[float](default = 0.001, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

def profile_likelihood_maximization(U, n_elbows, threshold):
    """
    Inputs
        U - An ordered or unordered list of eigenvalues
        n - The number of elbows to return

    Return
        elbows - A numpy array containing elbows
    """
    if type(U) == list: # cast to array for functionality later
        U = np.array(U)
    
    if n_elbows == 0: # nothing to do..
        return np.array([])
    
    if U.ndim == 2:
        U = np.std(U, axis = 0)
    
    U = U[U > threshold]
    
    if len(U) == 0:
        return np.array([])
    
    elbows = []
    
    if len(U) == 1:
        return np.array(elbows.append(U[0]))
    
    # select values greater than the threshold
    U.sort() # sort
    U = U[::-1] # reverse array so that it is sorted in descending order
    n = len(U)

    while len(elbows) < n_elbows and len(U) > 1:
        d = 1
        sample_var = np.var(U, ddof = 1)
        sample_scale = sample_var**(1/2)
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

class DimensionSelection(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '7b8ff08a-f887-3be5-86c8-9f0123bd4936',
        'version': "0.3.0",
        'name': "jhu.dimselect",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.DimensionSelection',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['dimselect primitive', 'dimension selection', 'dimension reduction', 'subspace', 'elbow', 'scree plot'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/dimselect/dimselect.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com//neurodata/primitives-interfaces.git',
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
        'primitive_family': "FEATURE_SELECTION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Select the right number of dimensions within which to embed given
        an adjacency matrix

        Inputs
            X - An n x n matrix or an ordered/unordered list of eigenvalues
            n - The number of elbows to return

        Return
            elbows - A numpy array containing elbows
        """

        #convert U to a matrix:
        U = self._convert_inputs(inputs=inputs)



        elbows = profile_likelihood_maximization(U, self.hyperparams['n_elbows'], self.hyperparams['error_threshold'])

        return base.CallResult(container.ndarray(elbows))

