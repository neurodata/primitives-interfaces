#!/usr/bin/env python

# ase.py
# Copyright (c) 2021. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
import numpy as np

from scipy.spatial.distance import cdist

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

class NearestNeighborNomination(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Creates a similarity matrix from pairwise distances, and subsequently
    nominates the closest neighbor in the second graph to each vertex in the
    first graph.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever.
        # Generated using "uuid.uuid4()".
        'id': '66e09f5b-3538-4d9a-9397-e32230608a35',
        'version': "0.1.0",
        'name': "jhu.nearest_neighbor_nomination",
        # Keywords do not have a controlled vocabulary. Authors can put here
        # whatever they find suitable.
        'keywords': ['nearest', 'neighbor', 'nomination', 'matching'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/nearest_neighbor_nomination/nearest_neighbor_nomination.py',
                'https://github.com/neurodata/primitives-interfaces',
            ],
            'contact': 'mailto:asaadel1@jhu.edu'
        },
        'description': 'Creates a similarity matrix from pairwise distances, and subsequently nominates the closest neighbor in the second graph to each vertex in the first graph.',
        'hyperparams_configuration': { },
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
        'python_path': 'd3m.primitives.graph_matching.nearest_neighbor_nomination.JHU',
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

        # do this more carefully TODO
        xhat_embedding = xhat.values[:,1:].astype(np.float32)
        yhat_embedding = yhat.values[:,1:].astype(np.float32)

        S = cdist(xhat_embedding, yhat_embedding, )
        match = np.argmin(S, axis=1)

        matches = np.zeros(len(reference), dtype=int)
        for i in range(len(reference)):
            e_id = xhat.index[xhat[xhat.columns[0]] == reference[reference.columns[1]].iloc[i]]
            g_id = yhat.index[yhat[yhat.columns[0]] == reference[reference.columns[2]].iloc[i]]
            matches[i] = 1 if g_id == match[e_id] else 0

        reference['match'] = matches

        results = reference[['d3mIndex', 'match']]

        predictions = {"d3mIndex": reference['d3mIndex'], "match": reference['match']}
        return base.CallResult(container.DataFrame(predictions),
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



