#! /usr/bin/env python

# util.py
# Created on 2017-09-14.

import igraph
from jhu_primitives.wrapper.read_graph_r import read_graph

def gen_graph_r(n=10, p=.2):
    g = igraph.Graph.Erdos_Renyi(n, p)
    gpath = "/tmp/graph.gml"
    g.write_gml(gpath)

    ig =  read_graph(gpath, "gml")
    return (gpath, ig)
