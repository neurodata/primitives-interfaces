#!/usr/bin/env python

# vnsgm.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os


from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
#from jhu_primitives.core.JHUGraph import JHUGraph
import numpy as np
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult


Inputs = container.ndarray
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    hp = None

class VertexNominationSeededGraphMatching(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'eeb707f5-bd15-35c6-90e4-32c56203ee01',
        'version': "0.1.0",
        'name': "jhu.vnsgm",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.VertexNominationSeededGraphMatching',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph matching'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/vnsgm/vnsgm.py',
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

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
#    def embed(self, *, g : JHUGraph, dim: int):
        """
        Perform seeded graph matching

        **Positional Arguments:**

        inputs[0]:
            - first graph adjacency matrix
        inputs[1]:
            - second graph adjacency matrix
        inputs[2]:
            - vector of indices for vertices of interest
        inputs[3]:
            - the matrix of seeds, s x 2 where s is number of seeds and
              column i seeds are for graph i            
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "vnsgm.interface.R")
        cmd = """
        source("%s")
        fn <- function(g1, g2, voi, seed = matrix(nrow = 0,ncol = 2)) {
            vnsgm.interface(g1, g2, voi, seed)
        }
        """ % path
        print(cmd)

        if len(inputs) == 3:
            result = np.array(robjects.r(cmd)(inputs[0], inputs[1], inputs[2] ))
        else:
            result = np.array(robjects.r(cmd)(inputs[0], inputs[1], inputs[2], inputs[3]))


        outputs = container.ndarray(result)

        return base.CallResult(outputs)

'''
from rpy2 import robjects
from typing import Sequence, TypeVar, Any
import os
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from primitive_interfaces.transformer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph
import numpy as np

Input = TypeVar('Input')
Output = TypeVar('Output')
Params = TypeVar('Params')

class
VertexNominationSeededGraphMatching(TransformerPrimitiveBase[Input, Output, Params]):
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    def match(self, *, g1 : JHUGraph, g2 : JHUGraph, voi : np.array, seeds: Input):
        """
        TODO: YP description

        **Positional Arguments:**

        g1:
            - A graph in R igraph format
        g2:
            - A graph in R igraph format
        voi:
            - vector of indices for vertices of interest
        seeds:
            - the matrix of seeds, s x 2 where s is number of seeds and
              column i seeds are for graph i
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "vnsgm.interface.R")
        cmd = """
        source("%s")
        fn <- function(g1, g2, voi, seeds) {
            vnsgm.interface(g1, g2, voi, seeds)
        }
        """ % path

        return np.array(robjects.r(cmd)(g1._object, g2._object, voi, seeds))
'''
