#!/usr/bin/env python

# gclust.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Union, Dict
import os
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
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

    max_clusters = hyperparams.Hyperparameter[int](default = 2,semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'])

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


class GaussianClustering(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '5194ef94-3683-319a-9d8d-5c3fdd09de24',
        'version': "0.1.0",
        'name': "jhu.gclust",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.GaussianClustering',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['gaussian clustering'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/gclust/gclust.py',
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

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, str] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        TODO: YP description

        **Positional Arguments:**

        inputs:
            - A matrix

        **Optional Arguments:**

        dim:
            - The number of clusters in which to assign the data
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "gclust.interface.R")
        path = file_path_conversion(path, uri = "")

        max_clusters = self.hyperparams['max_clusters'] #change this to differentiate

        cmd = """
        source("%s")
        fn <- function(X, max_clusters) {
            gclust.interface(X, max_clusters)
        }
        """ % path

        #print(cmd)

        #result = int(robjects.r(cmd)(inputs, dim)[0])
        result = robjects.r(cmd)(inputs, max_clusters)

        outputs = container.ndarray(result)

        return base.CallResult(outputs)




'''
import os
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from typing import Sequence, TypeVar
from primitive_interfaces.transformer import TransformerPrimitiveBase

Input = np.ndarray
Output = np.ndarray
Params = TypeVar('Params')

class GaussianClustering(TransformerPrimitiveBase[Input, Output, Params]):
    def cluster(self, *, inputs: Input) -> int:
        """
        TODO: YP description

        **Positional Arguments:**

        inputs:
            - A matrix

        **Optional Arguments:**

        dim:
            - The number of clusters in which to assign the data
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "gclust.interface.R")
        cmd = """
        source("%s")
        fn <- function(X, dim) {
            gclust.interface(X, dim)
        }
        """ % path
        return int(robjects.r(cmd)(inputs, dim)[0])

    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        self.cluster(inputs=inputs)
'''
