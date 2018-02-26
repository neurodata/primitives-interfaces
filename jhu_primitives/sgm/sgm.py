#!/usr/bin/env python
# sgm.py
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
    seed = hyperparams.Hyperparameter[np.ndarray](default=np.array([0]), semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

class SeededGraphMatching(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'ff22e721-e4f5-32c9-ab51-b90f32603a56',
        'version': "0.1.0",
        'name': "jhu.sgm",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.SeededGraphMatching',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph matching'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/sgm/sgm.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/neurodata/primitives-interfaces.git@{git_commit}#egg=primitives-interfaces'.format(
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

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, str] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
#    def embed(self, *, g : JHUGraph, dim: int):
        """
        Perform seeded graph matching

        **Positional Arguments:**

        inputs:
            - ndarray of two graph adjacency matrices

        **Optional Arguments:**

        seed:
            - The matrix of seed indices. The first column corresponds to seed index
              for graph 1 and second column corresponds to seed index for
              graph 2, where each row corresponds to a seed pair.
              If empty, assumes no seeds are used.
        """

        seed = self.hyperparams['seed']

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "sgm.interface.R")
        cmd = """
        source("%s")
        fn <- function(g1, g2, seed) {
            sgm.interface(g1, g2, seed)
        }
        """ % path
        print(cmd)

        result = np.array(robjects.r(cmd)(inputs[0], inputs[1], seed))

        outputs = container.ndarray(result)

        return base.CallResult(outputs)

'''
from rpy2 import robjects
from typing import Sequence, TypeVar, Any
import os
import numpy as np

from primitive_interfaces.transformer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph

Input = TypeVar('Input')
Output = TypeVar('Output')
Params = TypeVar('Params')

class SeededGraphMatching(TransformerPrimitiveBase[Input, Output, Params]):
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    def match(self, *, g1: JHUGraph, g2: JHUGraph, seeds: Any = 0):
        """
        TODO: YP description

        **Positional Arguments:**

        g1:
            - The first input graph object - in JHUGraph format
        g2:
            - The second input graph object - in JHUGraph format

        seeds:
            - The matrix of seed indices. The first column corresponds to seed index
              for graph 1 and second column corresponds to seed index for
              graph 2, where each row corresponds to a seed pair.
              If empty, assumes no seeds are used.
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "sgm.interface.R")

        cmd = """
        source("%s")
        fn <- function(g1, g2, s) {
            sgm.interface(g1, g2, s)
        }
        """ % path

        return np.array(robjects.r(cmd)(g1._object, g2._object, seeds))
'''
