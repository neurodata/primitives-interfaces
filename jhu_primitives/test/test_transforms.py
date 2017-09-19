#!/usr/bin/env python

# test_transforms.py

from jhu_primitives.utils.util import gen_graph_r
from jhu_primitives.core.JHUGraph import JHUGraph
from jhu_primitives.ase.SRC.ase import AdjacencySpectralEmbedding

def test():
    gpath, rig = gen_graph_r(n=50, p=.1)

    g = JHUGraph()
    g.read_graph(fname=gpath)

    print("Summary: ")
    g.summary()

    ASE = AdjacencySpectralEmbedding()
    print("ASE: ", ASE.embed(g=g, dim=4), "\n\n")

test()
