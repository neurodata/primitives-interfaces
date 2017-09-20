#!/usr/bin/env python

# JHUGraph.py
# Created on 2017-09-13.


from typing import NamedTuple, Sequence, Optional
from primitive_interfaces.graph import GraphPrimitiveBase
from jhu_primitives.wrapper.read_graph_r import read_graph
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_adjacency_matrix
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_num_vertices
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_num_edges
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_dangling_nodes
from jhu_primitives.wrapper.ig_wrapper_r import ig_is_directed
from jhu_primitives.wrapper.ig_wrapper_r import ig_is_weighted
from jhu_primitives.wrapper.ig_wrapper_r import ig_summary

import numpy as np

Input = str
Output = None

class Params(NamedTuple):
    adjacency_matrix: str # TODO: scipy.sparse.csc_matrix

class JHUGraph(GraphPrimitiveBase[Input, Output, Params]):

    adjacency_matrix = None
    _num_vertices = None
    _num_edges = None
    _directed = None
    _weighted = None
    _dangling_nodes = None

    def read_graph(self, *, fname: str, dtype: str = "gml", separator: str ="\t"):
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

    def compute_statistics(self):
        self._dangling_nodes = ig_get_dangling_nodes(self._object)

    def get_adjacency_matrix(self):
        return ig_get_adjacency_matrix(self._object)

    def get_num_vertices(self):
        return self._num_vertices

    def get_num_edges(self):
        return self._num_edges

    def is_directed(self):
        return self._directed

    def is_weighted(self):
        return self._weighted

    def get_dangling_nodes(self):
        if (self._dangling_nodes is None):
            self.compute_statistics()
        return self._dangling_nodes

    def summary(self):
        ig_summary(self._object)

    # TODO: Below
    def set_training_data(self, *, inputs: Sequence[Input] = None,
            outputs: Sequence[Output] = None) -> None:
        pass

    def fit(self, *, timeout: float = None, iterations: Optional[int] = 1) -> bool:
        pass

    def get_params(self) -> Params:
        pass

    def set_params(self, *, params: Params) -> None:
        pass

    def set_random_seed(self, *, seed: int) -> None:
        pass

    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass # TODO: figure out what this should do ...
