#!/usr/bin/env python

# test_transforms.py

import numpy as np

from jhu_primitives.utils.util import gen_graph_r
from jhu_primitives.core.JHUGraph import JHUGraph
from jhu_primitives import *
from jhu_primitives.wrapper.read_graph_r import read_graph
from jhu_primitives.wrapper.ig_wrapper_r import ig_get_adjacency_matrix


def test():
    gpath, rig = gen_graph_r(n=50, p=.1)
    hyperparams_class = JHUGraph.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    g = JHUGraph(hyperparams=hyperparams_class.defaults())
    g.produce(inputs=gpath)

    print("Summary: ")
    g.summary()

    hyperparams_class = AdjacencySpectralEmbedding.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    ASE = AdjacencySpectralEmbedding(hyperparams=hyperparams_class.defaults())
    print("ASE: ", ASE.produce(inputs=g.get_adjacency_matrix()).value, "\n\n")

    hyperparams_class = LaplacianSpectralEmbedding.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    LSE = LaplacianSpectralEmbedding(hyperparams=hyperparams_class.defaults())
    print("LSE: ", LSE.produce(inputs=g.get_adjacency_matrix()).value, "\n\n")

    hyperparams_class = DimensionSelection.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    DIMSELECT = DimensionSelection(hyperparams=hyperparams_class.defaults())
    print("DIMSELECT: ",
            DIMSELECT.produce(inputs=np.random.random((128,6))).value, "\n\n")

    hyperparams_class = GaussianClustering.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    GCLUST = GaussianClustering(hyperparams=hyperparams_class.defaults())
    print("GCLUST: ",
            GCLUST.produce(inputs=np.random.random((64,8))).value, "\n\n")

    hyperparams_class = NonParametricClustering.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    NONPAR = NonParametricClustering(hyperparams=hyperparams_class.defaults())
    xhat1 = np.random.random((16, 2))
    xhat2 = np.random.random((16, 2))
    print("NONPAR: ",
            NONPAR.produce(inputs=np.stack((xhat1,xhat2))).value, "\n\n")

    '''
    hyperparams_class = OutOfCoreAdjacencySpectralEmbedding.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    OOCASE = OutOfCoreAdjacencySpectralEmbedding(hyperparams=hyperparams_class.defaults())
    print("OOCASE: ", OOCASE.produce(inputs=g.get_adjacency_matrix()).value, "\n\n")
    '''

    hyperparams_class = NumberOfClusters.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    NUMCLUST = NumberOfClusters(hyperparams=hyperparams_class.defaults())
    print("NUMCLUST: ",
            NUMCLUST.produce(inputs=np.random.random((128, 16))).value, "\n\n")

    hyperparams_class = PassToRanks.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    PTR = PassToRanks(hyperparams=hyperparams_class.defaults())
    print("PTR: ", PTR.produce(inputs=g.get_adjacency_matrix()).value, "\n\n")

    hyperparams_class = SpectralGraphClustering.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    SGC = SpectralGraphClustering(hyperparams=hyperparams_class.defaults())
    print("SGC: ", SGC.produce(inputs=g.get_adjacency_matrix()).value, "\n\n")

    gpath, rig = gen_graph_r(n=50, p=.1)
    hyperparams_class = JHUGraph.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    g2 = JHUGraph(hyperparams=hyperparams_class.defaults())
    g2.produce(inputs=gpath)

    hyperparams_class = SeededGraphMatching.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    SGM = SeededGraphMatching(hyperparams=hyperparams_class.defaults())
    print("SGM: ", SGM.produce(inputs=np.array([g.get_adjacency_matrix(),g2.get_adjacency_matrix()])).value, "\n\n")

    hyperparams_class = VertexNominationSeededGraphMatching.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    VNSGM = VertexNominationSeededGraphMatching(hyperparams=hyperparams_class.defaults())
    print("VNSGM: ", VNSGM.produce(inputs=np.array([g.get_adjacency_matrix(),g2.get_adjacency_matrix(),np.array([1,2,3]),np.transpose(np.array([[8,4,3],[1,2,3]]))])).value, "\n\n")

test()
