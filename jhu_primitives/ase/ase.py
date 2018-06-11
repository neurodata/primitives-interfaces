#!/usr/bin/env python

# ase.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
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


Inputs = container.matrix
Outputs = container.ndarray

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    max_dimension = hyperparams.Hyperparameter[int](default=2, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

def file_path_conversion(abs_file_path, uri="file"):
    local_drive, file_path = abs_file_path.split(':')[0], abs_file_path.split(':')[1]
    path_sep = file_path[0]
    file_path = file_path[1:]  # Remove initial separator
    if len(file_path) == 0:
        print("Invalid file path: len(file_path) == 0")
        return

    s = ""
    if path_sep == "/":
        s = file_path
    elif path_sep == "\\":
        splits = file_path.split("\\")
        data_folder = splits[-1]
        for i in splits:
            if i != "":
                s += "/" + i
    else:
        print("Unsupported path separator!")
        return

    if uri == "file":
        return "file://localhost" + s
    else:
        return local_drive + ":" + s

class AdjacencySpectralEmbedding(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'b940ccbd-9e9b-3166-af50-210bfd79251b',
        'version': "0.3.0",
        'name': "jhu.ase",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['ase primitive'],
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
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.AdjacencySpectralEmbedding',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "SINGULAR_VALUE_DECOMPOSITION"
        ],
        'primitive_family': "DATA_TRANSFORMATION"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str,base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
#    def embed(self, *, g : JHUGraph, dim: int):
        """
        Perform Adjacency Spectral Embedding on a graph
        TODO: YP description

        **Positional Arguments:**

        g:
            - Graph in JHUGraph format

        **Optional Arguments:**

        dim:
            - The number of dimensions in which to embed the data
        """

        max_dimension = self.hyperparams['max_dimension']

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "ase.interface.R")
        path = file_path_conversion(path, uri = "")
        cmd = """
        source("%s")
        fn <- function(inputs, max_dimension) {
            ase.interface(inputs, max_dimension)
        }
        """ % path

        result = robjects.r(cmd)(inputs, max_dimension)
        vectors = container.ndarray(result[0])
        eig_values = container.ndarray(result[1])

        return base.CallResult([vectors, eig_values])
        

    def set_training_data(self) -> None:  # type: ignore
        """
        A noop.
        """

        return

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        A noop.
        """

        return

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return


"""
metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'b940ccbd-9e9b-3166-af50-210bfd79251b',
        'version': '0.3.0',
        'name': "Monomial Regressor",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['test primitive'],
        'source': {
            'name': 'JHU Team',
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
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
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
    })"""
