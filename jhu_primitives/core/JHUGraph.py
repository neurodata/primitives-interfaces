#!/usr/bin/env python

# JHUGraph.py
# Created on 2017-09-13.


from typing import NamedTuple, Sequence, Optional
from primitive_interfaces.clustering import ClusteringPrimitiveBase
from jhu_primitives.wrapper.read_graph_r import read_graph
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_adjacency_matrix
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_num_vertices
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_num_edges
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_dangling_nodes
from jhu_primitives.wrapper.ig_wrapper_r import ig_is_directed
from jhu_primitives.wrapper.ig_wrapper_r import ig_is_weighted
from jhu_primitives.wrapper.ig_wrapper_r import ig_summary
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_dense_matrix
from primitive_interfaces.base import Hyperparams
from d3m_metadata import container, hyperparams, metadata as metadata_module, params, utils
from d3m_metadata.params import Params
import os

from primitive_interfaces.base import CallResult

import numpy as np

Inputs = container.matrix
Outputs = container.ndarray

class Hyperparams(hyperparams.Hyperparams):
    # TODO: Fix medatadata parameter
    dtype = hyperparams.Hyperparameter[str](default="gml", semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])

class JHUGraph(ClusteringPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    # TODO: Create metadata for this
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'b940ccbd-9e9b-3166-af50-210bfd79251b',
        'version': "crap",
        'name': "Monomial Regressor",
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

    _adjacency_matrix = None
    _num_vertices = None
    _num_edges = None
    _directed = None
    _weighted = None
    _dangling_nodes = None

    def read_graph(self, *, fname: str) -> None:

        dtype = self.hyperparams['dtype']

        if dtype == "gml":
            self._object = read_graph(fname, "gml")
        elif dtype.startswith("edge"):
            self._object = read_graph(fname, "edge")
        else:
            raise NotImplementedError("Reading graphs of type '{}'".\
                    format(dtype))

        self._num_vertices = ig_get_num_vertices(self._object)
        self._num_edges = ig_get_num_edges(self._object)
        self._directed = ig_is_directed(self._object)
        self._weighted = ig_is_weighted(self._object)

    def compute_statistics(self) -> Outputs:
        self._dangling_nodes = ig_get_dangling_nodes(self._object)

    def get_adjacency_matrix(self) -> Outputs:
        return ig_get_adjacency_matrix(self._object)

    def get_dense_matrix(self) -> Outputs:
        return ig_get_dense_matrix(self._object)

    def get_num_vertices(self) -> int:
        return self._num_vertices

    def get_num_edges(self) -> int:
        return self._num_edges

    def is_directed(self) -> bool:
        return self._directed

    def is_weighted(self) -> bool:
        return self._weighted

    def get_dangling_nodes(self) -> Outputs:
        if (self._dangling_nodes is None):
            self.compute_statistics()
        return self._dangling_nodes

    def summary(self) -> None:
        ig_summary(self._object)

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        pass

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        return base.CallResult(self.get_adjacency_matrix())

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        return base.CallResult(None)

    def get_params(self) -> Params:
        return Params(other={})

    def set_params(self, *, params: Params) -> None:
        return None
