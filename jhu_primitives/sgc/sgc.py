#!/usr/bin/env python

# sgc.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os


from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
import numpy as np
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from .. import LargestConnectedComponent
from .. import PassToRanks
from .. import AdjacencySpectralEmbedding
from .. import DimensionSelection
from .. import GaussianClustering

Inputs = container.ndarray
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    dim = None

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
            "SPECTRAL_CLUSTERING"
        ],
        'primitive_family': "GRAPH_CLUSTERING"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Perform spectral graph clustering

        Inputs
            n x 2, n x n

        inputs:
            - JHUGraph adjacency matrix
        """

        G = inputs

        hp_lcc = LargestConnectedComponent.Hyperparams.defaults()
        lcc = LargestConnectedComponent(hyerparams = hp_lcc).produce(inputs = G).value

        hp_ptr = PassToRanks.Hyperparams.defaults()
        ptr = PassToRanks(hyperparams = hp_ptr).produce(inputs = lcc).value

        max_dim = min(100, len(ptr)/2) 
        hp_ase = AdjacencySpectralEmbedding.Hyperparams({'embedding_dimension': max_dim})
        ase = AdjacencySpectralEmbedding(hyperparams = hp_ase).produce(inputs = ptr).value

        vectors = ase[0]
        values = ase[1]

        hp_dimselect = DimensionSelection.Hyperparams({'n_elbows': 2})
        ds = DimensionSelection(hyperparams = hp_dimselect).produce(inputs = values).value

        elbow_2 = ds[1]

        hp_gclust = GaussianClustering.Hyperparams({'max_clusters': 30})
        labels = GaussianClustering(hyperparams = hp_gclust).produce(inputs = vectors[:, elbow_2]).value

        outputs = container.ndarray(labels)

        return base.CallResult(outputs)