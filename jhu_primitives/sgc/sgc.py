#!/usr/bin/env python

# sgc.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os


from primitive_interfaces.transformer import TransformerPrimitiveBase
#from jhu_primitives.core.JHUGraph import JHUGraph
import numpy as np
from d3m_metadata import container, hyperparams, metadata as metadata_module, params, utils
from primitive_interfaces import base
from primitive_interfaces.base import CallResult


Inputs = container.ndarray
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    dim = hyperparams.Hyperparameter[None](default=None)

class SpectralGraphClustering(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'fde90b15-155a-3c2a-866c-4a19354cf0c7',
        'version': "0.1.0",
        'name': "jhu.sgc",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.SpectralGraphClustering',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['spectral clustering'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/sgc/sgc.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
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
#            {
#            'type': 'UBUNTU',
#            'package': 'r-base-dev',
#            'version': '3.4.2'
#            },
#            {
#            'type': 'UBUNTU',
#            'package': 'r-recommended',
#            'version': '3.4.2'
#            },
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
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "GAUSSIAN_PROCESS"
        ],
        'primitive_family': "GRAPH_CLUSTERING"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, str] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Perform spectral graph clustering

        **Positional Arguments:**

        inputs:
            - JHUGraph adjacency matrix
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "sgc.interface.R")
        cmd = """
        source("%s")
        fn <- function(inputs) {
            sgc.interface(inputs)
        }
        """ % path
        print(cmd)

        result = np.array(robjects.r(cmd)(inputs))

        outputs = container.ndarray(result)

        return base.CallResult(outputs)

'''
import os
from rpy2 import robjects
from typing import Sequence, TypeVar
import numpy as np

from primitive_interfaces.transformer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph

Input = JHUGraph
Output = np.ndarray
Params = TypeVar('Params')

class SpectralGraphClustering(TransformerPrimitiveBase[Input, Output, Params]):
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        """
        TODO: YP description

        **Positional Arguments:**

        g:
            - A graph in R 'igraph' format
        """
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "sgc.interface.R")
        cmd = """
        source("%s")
        fn <- function(g) {
            sgc.interface(g)
        }
        """ % path

        return np.array(robjects.r(cmd)(inputs._object))
'''
