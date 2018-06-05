#!/usr/bin/env python

# Graph.py
# Created on 2017-09-13.

import abc

class Graph(object):
    adjacency_matrix = None
    _num_vertices = None
    _num_edges = None
    _directed = None
    _weighted = None
    _dangling_nodes = None

    def __init__(self):
        pass

    @abc.abstractmethod
    def read_graph(self, fname, dtype="gml", separator="\t"):
        pass

    @abc.abstractmethod
    def compute_statistics(self):
        pass

    @abc.abstractmethod
    def get_adjacency_matrix(self):
        return self.adjacency_matrix

    @abc.abstractmethod
    def get_num_vertices(self):
        return self._num_vertices

    @abc.abstractmethod
    def get_num_edges(self):
        return self._num_edges

    @abc.abstractmethod
    def is_directed(self):
        return self._directed

    def is_weighted(self):
        return self._weighted

    @abc.abstractmethod
    def get_dangling_nodes(self):
        if (self._dangling_nodes is None):
            self.compute_statistics()
        return self._dangling_nodes
