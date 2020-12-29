#!/usr/bin/env python

# ase.py
# Copyright (c) 2020. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
import sys
import numpy as np

from scipy.linalg import orthogonal_procrustes

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import utils, container
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    pass
    # max_dimension = hyperparams.Bounded[int](
    #     default=2,
    #     semantic_types= [
    #         'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    # ],
    #     lower = 1,
    #     upper = None
    # )

    # which_elbow = hyperparams.Bounded[int](
    #     default = 1,
    #     semantic_types= [
    #         'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    # ],
    #     lower = 1,
    #     upper = 2
    # )

    # use_attributes = hyperparams.Hyperparameter[bool](
    #     default = False,
    #     semantic_types = [
    #         'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    # ],
# )

class PartialProcrustes(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Creates a similarity matrix from pairwise distances and nominates one-to-one
    smallest distance vertex match.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'd830b3c3-e8f1-480d-9970-c404e5064edd',
        'version': "0.1.0",
        'name': "jhu.partial_procrustes",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['procrustes', 'matching', 'alignment'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/partial_procrustes/partial_procrustes.py',
                'https://github.com/neurodata/primitives-interfaces',
            ],
            'contact': 'mailto:asaadel1@jhu.edu'
        },
        'description': 'Aligns two datasets based on the rotation for a seeded group of entries',
        'hyperparams_configuration': {
            # 'max_dimension': 'The maximum dimension that can be used for eigendecomposition',
            # 'which_elbow': 'The scree plot "elbow" to use for dimensionality reduction. High values leads to more dimensions selected.',
            # 'use_attributes': 'Boolean which indicates whether to use the attributes of the nodes.'
        },
        # A list of dependencies in order. These can be Python packages, system
        # packages, or Docker images. Of course Python packages can also have
        # their own dependencies, but sometimes it is necessary to install a
        # Python package first to be even able to run setup.py of another
        # package. Or you have a dependency which is not on PyPi.
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
            },
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
        'python_path': 'd3m.primitives.graph_matching.partial_procrustes.JHU',
        # Choose these from a controlled vocabulary in the schema. If anything
        # is missing which would best describe the primitive, make a merge
        # request.
        'algorithm_types': [
            "RANDOM_GRAPH"
        ],
        'primitive_family':
            'GRAPH_MATCHING',
        'preconditions': [
            'NO_MISSING_VALUES'
        ]
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0,
                 docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams,
                         random_seed=random_seed,
                         docker_containers=docker_containers)

    def produce(self, *,
                inputs_1: Inputs,
                inputs_2: Inputs,
                reference: Inputs,
                timeout: float = None,
                iterations: int = None) -> CallResult[Outputs]:
        xhat = inputs_1
        yhat = inputs_2

        seeds = reference['match'].astype(bool)

        xhat_seed_names = reference[reference.columns[1]][seeds]
        yhat_seed_names = reference[reference.columns[2]][seeds]




        xhat_embedding_s = xhat.loc[xhat[xhat.columns[0]].isin(xhat_seed_names)].values[:,1:].astype(float32)
        yhat_embedding_s = yhat.loc[yhat[yhat.columns[0]].isin(yhat_seed_names)].values[:,1:].astype(float32)
        print(xhat_embedding_s.dtype, file=sys.stderr)
        print(yhat_embedding_s.dtype, file=sys.stderr)

        print(xhat_embedding_s, file=sys.stderr)
        print(yhat_embedding_s, file=sys.stderr)


        w, _ = orthogonal_procrustes(yhat_embedding_s, xhat_embedding_s)
        yhat_embedding_align = yhat.values[:,1:] @ w 
        yhat_align = yhat.copy()
        yhat_align[yhat.columns[1:]] = yhat_embedding_align


    
        return base.CallResult(yhat_align,
                               has_finished = True, iterations_done = 1)

        # return base.CallResult(reference, #results,
        #                        has_finished=True,
        #                        iterations_done=1)

    def multi_produce(self, *,
                      produce_methods: Sequence[str],
                      inputs_1: Inputs,
                      inputs_2: Inputs,
                      reference: Inputs,
                      timeout: float = None,
                      iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._multi_produce(produce_methods=produce_methods,
                                   timeout=timeout,
                                   iterations=iterations,
                                   inputs_1=inputs_1,
                                   inputs_2=inputs_2,
                                   reference=reference)

    def fit_multi_produce(self, *,
                          produce_methods: Sequence[str],
                          inputs_1: Inputs,
                          inputs_2: Inputs,
                          reference: Inputs,
                          timeout: float = None,
                          iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._fit_multi_produce(produce_methods=produce_methods,
                                       timeout=timeout,
                                       iterations=iterations,
                                       inputs_1=inputs_1,
                                       inputs_2=inputs_2,
                                       reference=reference)




