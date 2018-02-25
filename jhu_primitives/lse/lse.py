#!/usr/bin/env python

# lse.py
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


Inputs = container.matrix
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    dim = hyperparams.Hyperparameter[int](default=2, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

class LaplacianSpectralEmbedding(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'b940ccbd-9e9b-3166-af50-210bfd79251b',
        'version': "crap",
        'name': "Laplacian Spectral Embedding",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['test primitive'],
        'source': {
            'name': "boss",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirecto\
ry=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        # URIs at which one can obtain code for the primitive, if available.
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.test.MonomialPrimitive',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.LINEAR_REGRESSION,
        ],
        'primitive_family': metadata_module.PrimitiveFamily.REGRESSION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, str] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
#    def embed(self, *, g : JHUGraph, dim: int):
        """
        Perform Laplacian Spectral Embedding on a graph
        TODO: YP description

        **Positional Arguments:**

        g:
            - Graph in JHUGraph format

        **Optional Arguments:**

        dim:
            - The number of dimensions in which to embed the data
        """

        dim = self.hyperparams['dim']

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "lse.interface.R")
        cmd = """
        source("%s")
        fn <- function(inputs, dim) {
            lse.interface(inputs, dim)
        }
        """ % path
        print(cmd)

        result = np.array(robjects.r(cmd)(inputs, dim))

        outputs = container.ndarray(result)

        return base.CallResult(outputs)
