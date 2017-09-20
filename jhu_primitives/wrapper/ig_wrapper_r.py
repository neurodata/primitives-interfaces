#! /usr/bin/env python

# ig_wrapper_r.py
# Created on 2017-09-14.

from rpy2 import robjects
import numpy as np
from jhu_primitives.utils.util import gen_graph_r

def ig_get_adjacency_matrix(ig):
    fn = robjects.r("""
    fn <- function(ig) {
        suppressMessages(require(igraph))
        get.adjacency(ig)
    }
    """)
    return fn(ig)

def ig_get_num_vertices(ig):
    fn = robjects.r("""
    fn <- function(ig) {
        suppressMessages(require(igraph))
        vcount(ig)
    }
    """)
    return fn(ig)[0]

def ig_get_num_edges(ig):
    fn = robjects.r("""
    fn <- function(ig) {
        suppressMessages(require(igraph))
        ecount(ig)
    }
    """)
    return fn(ig)[0]

def ig_get_dangling_nodes(ig):
    fn = robjects.r("""
    fn <- function(ig) {
        suppressMessages(require(igraph))
        which(degree(ig) == 0)
    }
    """)
    return np.array(fn(ig))

def ig_is_weighted(ig):
    fn = robjects.r("""
    fn <- function(ig) {
        suppressMessages(require(igraph))
       is.weighted(ig)
    }
    """)
    return bool(fn(ig)[0])

def ig_is_directed(ig):
    fn = robjects.r("""
    fn <- function(ig) {
        suppressMessages(require(igraph))
        is.directed(ig)
    }
    """)
    return bool(fn(ig)[0])

def ig_summary(ig):
    fn = robjects.r("""
    fn <- function(ig) {
        suppressMessages(require(igraph))
        summary(ig)
    }
    """)
    fn(ig)

def test_ig_get_adjacency_matrix():
    rg = gen_graph_r()
    r_adj_mat = ig_get_adjacency_matrix(rg)
    print(r_adj_mat)

def test():
    test_ig_get_adjacency_matrix()
