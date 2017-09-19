#!/usr/bin/env python

# read_graph_r.py
# Created on 2017-09-14.

from rpy2 import robjects

def read_graph(path, fmt):
    """
    Read a graph and secretly return an R igraph ... sshh

    **Positional Arguments:**

    path:
        - The path to a gml data file

    **Optional Arguments:**

    """

    reader = robjects.r("""
    fn <- function(path, fmt) {
        if(!suppressMessages(require(igraph))) {
            install.packages("igraph")
        }
        suppressMessages(require(igraph))
        read_graph(path, fmt)
    }
    """)

    return reader(path, fmt)

def print_summary(g):
    """
    Print the summary of an R igraph
    """
    ps = robjects.r("""
    fn <- function(g) {
        suppressMessages(require(igraph))
        summary(g)
    }
    """)

    ps(g)

def test():
    import igraph
    g = igraph.Graph.Erdos_Renyi(50, .2)
    gpath = "/tmp/graph.gml"
    g.write_gml(gpath)
    print(("python graph says:\n {}\n".format(g.summary())))

    rg = read_graph(gpath, "gml")
    print ("R graph says:\n"); print_summary(rg)
