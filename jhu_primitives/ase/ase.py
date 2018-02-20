#!/usr/bin/env python

# ase.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar
import os

from primitive_interfaces.transformer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph
import numpy as np
from d3m_metadata import container, hyperparams, metadata as metadata_module, params, utils
from primitive_interfaces import base


Input = TypeVar('Inputs')
Output = TypeVar('Outputs')
#Params = TypeVar('Params')


class AdjacencySpectralEmbedding(TransformerPrimitiveBase[Input, Output, None]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
    "id": "b940ccbd-9e9b-3166-af50-210bfd79251b",
    "name": "jhu.ase",
    "common_name": "JHU Graph Embedding",
    "description": "This is the R implementation of selecting the number of significant singular values (or column variances), by finding the 'elbow' of the scree plot, in a principled way. The dimensionality d is chosen to maximize the likelihood when the d largest singular values (or column variances) are assigned to one component of the mixture and the rest of the singular values (column variances) assigned to the other component.",
    "languages": ["R", "python3.6"],
    "library": "gmmase",
    "version": "0.3.0",
    "is_class": True,
    "team": "JHU",
    "schema_version": 1.0,
    "tags": [
        "embed",
        "graph",
        "feature",
        "dimension"
    ],
    "algorithm_type": ["spectral embedding"],
    "learning_type":  ["unsupervised"],
    "interface_type": "unsupervised_learning_clustering",
    "source_code": "https://github.com/neurodata/primitives-interfaces/tree/master/jhu_primitives",
    "methods_available": [
        {
            "name": "embed_adjacency_matrix",
            "id": "jhu.ase",
            "description": "reference: Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E. A Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs, Journal of the American Statistical Association, Vol. 107(499), 2012",
            "parameters": [
                {
                    "shape": "n_samples, n_samples",
                    "type": "array-like",
                    "is_hyperparameter": False,
                    "name": "graph",
                    "description": "an input graph, it must be JHUGraph format"
                },
                {
                    "type": "float",
                    "name": "dim",
                    "is_hyperparameter": False,
                    "default": "2",
                    "optional": True,
                    "description": "embedding dimension of the spectral embedding"
                }
            ],
            "returns": {
                    "shape": "n_samples, m_features",
                    "type": "array-like",
                    "name": "X",
                    "description": "Estimated latent positions"
            }
        }
    ],
    "task_type": ["modeling"],
    "build": [
        {
            "type": "pip",
            "package": "git+https://github.com/neurodata/primitives-interfaces.git"
        }
    ],
    "compute_resources": {
        "sample_size": [1, 10, 100, 1000],
        "sample_unit": ["MB", "MB", "MB", "MB"],
        "num_nodes": [1, 1, 1, 1],
        "cores_per_node": [1, 1, 1, 1],
        "gpus_per_node": [0, 0, 0, 0],
        "mem_per_node": [2.001, 2.01, 2.1, 3],
        "disk_per_node": [2.001, 2.01, 2.1, 3],
        "mem_per_gpu": [0, 0, 0, 0],
        "expected_running_time": [0.1, 0.2, 1, 10]
    }
})

    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    def embed(self, *, g : JHUGraph, dim: int = 2):
        """
        Perform Adjacency Spectral Embedding on a graph
        TODO: YP description

        **Positional Arguments:**

        g:
            - Graph in JHUGraph format

        **Optional Arguments:**

        dim:
            - The number of dimensions in which to embed the data
        """
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "ase.interface.R")
        cmd = """
        source("%s")
        fn <- function(g, dim) {
            ase.interface(g, dim)
        }
        """ % path
        print(cmd)

        return np.array(robjects.r(cmd)(g._object, dim))
