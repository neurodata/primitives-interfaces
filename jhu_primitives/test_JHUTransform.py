#!/usr/bin/env python

# test_JHUTransform.py

from util import gen_graph_r
from JHUGraph import JHUGraph
from JHUTransform import JHUTransform

def test():
    gpath, rig = gen_graph_r(n=50, p=.1)

    g = JHUGraph()
    g.read_graph(gpath)

    print("Summary: ")
    g.summary()
    t = JHUTransform()

    # ASE
    ASE = t.ase_transform(g, 4)
    print("ASE: ", ASE, "\n\n")

    # LSE
    LSE = t.lse_transform(g, 4)
    print("LSE: ", ASE, "\n\n")

    # PTR
    PTR = t.ptr_transform(g)
    print("PTR: ", PTR, "\n\n")

    # DIM
    DIM = t.dimselect_transform(g.get_adjacency_matrix())
    print("DIM: ", DIM, "\n\n")

    # SGM
    gpath, rig2 = gen_graph_r(n=50, p=.1)
    g2 = JHUGraph()
    g2.read_graph(gpath)

    SGM = t.sgm_transform(g, g2, 3)
    print("SGM: ", SGM, "\n\n")

    # GCLUST
    GCLUST = t.gclust_transform(ASE, 2)
    print("GCLUST: ", GCLUST, "\n\n")

    # NONPAR
    # NONPAR = t.nonpar_transform(ASE[:3,:3], LSE[:3, :3])
    # print "NONPAR: ", NONPAR, "\n\n"

    # SGC
    SGC = t.sgc_transform(g)
    print("SGC: ", SGC, "\n\n")

    # NUMCLUST
    NUMCLUST = t.numclust_transform(SGC)
    print("NUMCLUST: ", NUMCLUST, "\n\n")

    # VNSGM
    VNSGM = t.vnsgm_transform(g, g2, 2, 10)
    print("VNSGM: ", VNSGM, "\n\n")

test()
