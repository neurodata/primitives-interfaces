#!/usr/bin/env python

# test_transforms.py

from util import gen_graph_r
from JHUGraph import JHUGraph
import ase

def test():
    gpath, rig = gen_graph_r(n=50, p=.1)

    g = JHUGraph()
    g.read_graph(fname=gpath)

    print("Summary: ")
    g.summary()

    import pdb; pdb.set_trace()
    ASE = ase.AdjacencySpectralEmbedding()
    print("ASE: ", ASE.embed(g), "\n\n")

test()
