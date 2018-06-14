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
from scipy.stats import norm
from d3m import utils, container
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from jhu_primitives.utils.util import file_path_conversion

Inputs = container.ndarray
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    n_elbows = hyperparams.Hyperparameter[int](default=3, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

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
        'keywords': ['dimselect primitive'],
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
            "HIGHER_ORDER_SINGULAR_VALUE_DECOMPOSITION"
        ],
        'primitive_family': "DATA_TRANSFORMATION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Select the right number of dimensions within which to embed given
        an adjacency matrix

        **Positional Arguments:**

        X:
            - Adjacency matrix
        """
        
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "dimselect.interface.R")

        path = file_path_conversion(path, uri="")
        n_elbows = self.hyperparams['n_elbows']

        cmd = """
        source("%s")
        fn <- function(X, n_elbows) {
            dimselect.interface(X, n_elbows)
        }
        """ % path

        result = np.array(robjects.r(cmd)(inputs, n_elbows))

        outputs = container.ndarray(result)

        return base.CallResult(outputs)

        ### BEGIN - PYTHON IMPLEMENTATION

        """
        TODO: If a graph, use ASE...

        if type(g) == numpy.ndarray:
            if g.shape[0] == g.shape[1]: # n x n matrix
                g = networkx.Graph(g) # convert to networkx graph to be able to extract edge list 
            elif g.shape[1] == 2: # n x 2 matrix
                g = igraph.Graph(list(g))
            else:
                print("Neither n x n nor n x 2. Please submit a square matrix or edge list.")
                return
                
        if type(g) == networkx.classes.graph.Graph: # networkx graph
            g = igraph.Graph(list(g.edges)) # convert to igraph graph, find the clusters

		if inputs is <networkx.graph.Graph>

		if inputs is <type(ase.produce.value[1]) or flat np.array>:
			elbows = profile_likelihood_maximization(U, self.hyperparams['n_elbows'])
		

	def profile_likelihood_maximization(U, n_elbows, elbows_found = []):
		"""
		"""
		Inputs
			U - An ordered list of eigenvalues
			n - The number of elbows to return

		Return
			elbows - An numpy array containing elbows

		if len(elbows_found) == n_elbows:
			return elbows_found

		n = len(X)

		if n == 0:
			return container.ndarray(elbows_found)

		if n == 1:
			if len(elbows_found) == 0:
				return U
			else:
				return container.ndarray(elbows_found.append(U[0] + elbows_found[-1]))

    	d = 1
    	sample_var = var(U, ddof = 1)
    	sample_scale = sample_var**(1/2)
    	elbow = 1 # Initialize elbow
    	likelihood_elbow = 0
    	likelihoods = np.array([])
    	while d < n:
        	mean_sig = mean(X[:d])
        	mean_noise = mean(X[d:])
        	sig_likelihood = 0
        	noise_likelihood = 0
        	for i in range(d):
            	sig_likelihood += norm.pdf(X[i], mean_sig, sample_scale)
        	for i in range(d, n):
            	noise_likelihood += norm.pdf(X[i], mean_noise, sample_scale)
            
        	likelihood = noise_likelihood + sig_likelihood
        	likelihoods.append(likelihood)
        
        	if likelihood > likelihood_elbow:
            	likelihood_elbow = likelihood 
            	elbow = d
        	d += 1
    	return profile_likelihood_maximization(U[:d], n_elbows, elbows_found.append(elbow + elbows_found[-1]))

        """

