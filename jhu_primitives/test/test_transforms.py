#!/usr/bin/env python

# test_transforms.py

import numpy as np

from jhu_primitives.utils.util import gen_graph_r
from jhu_primitives.core.JHUGraph import JHUGraph
from jhu_primitives import *


def test():
    gpath, rig = gen_graph_r(n=50, p=.1)

    g = JHUGraph()
    g.read_graph(fname=gpath)

    
    print("Summary: ")
    g.summary()

    ASE = AdjacencySpectralEmbedding()
    print("ASE: ", ASE.embed(g=g, dim=4), "\n\n")

    LSE = LaplacianSpectralEmbedding()
    print("LSE: ", LSE.embed(g=g, dim=4), "\n\n")

    DIMSELECT = DimensionSelection()
    print("DIMSELECT: ",
            DIMSELECT.produce(inputs=np.random.random((128, 16))), "\n\n")

    GCLUST = GaussianClustering()
    print("GCLUST: ",
            GCLUST.cluster(inputs=np.random.random((64, 8)), dim=4), "\n\n")
    print("GCLUST: ",
            GCLUST.produce(inputs=np.random.random((64, 8))), "\n\n")

    NONPAR = NonParametricClustering()
    xhat1 = np.random.random((16, 2))
    xhat2 = np.random.random((16, 2))
    print("NONPAR: ", NONPAR.cluster(xhat1=xhat1, xhat2=xhat2), "\n\n")

    # OOCASE = OutOfCoreAdjacencySpectralEmbedding()
    # print("OOCASE: ", OOCASE.embed(g=g), "\n\n")

    NUMCLUST = NumberOfClusters()

    print("NUMCLUST: ",
            NUMCLUST.produce(inputs=np.random.random((128, 16))), "\n\n")

    PTR = PassToRanks()
    print("PTR: ", PTR.produce(inputs=g), "\n\n")

    SGC = SpectralGraphClustering()
    print("SGC: ", SGC.produce(inputs=g), "\n\n")
    
    gpath, rig = gen_graph_r(n=50, p=.1)
    g2 = JHUGraph()
    g2.read_graph(fname=gpath)

    SGM = SeededGraphMatching()
    print("SGM: ", SGM.match(g1=g, g2=g2), "\n\n")

    VNSGM = VertexNominationSeededGraphMatching()
    print("VNSGM: ", VNSGM.match(g1=g, g2=g2,
        voi=np.array([1,2,3]), seeds=np.array([[8,4,3],[1,2,3]])), "\n\n")

test()
