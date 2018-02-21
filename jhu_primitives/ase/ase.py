#!/usr/bin/env python

# ase.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar
import os

from primitive_interfaces.transformer import TransformerPrimitiveBase
#from primitive_interfaces.base import Hyperparams
from jhu_primitives.core.JHUGraph import JHUGraph
import numpy as np
from d3m_metadata import container, hyperparams, metadata as metadata_module, params, utils, types
from primitive_interfaces import base
from primitive_interfaces.base import CallResult


Input = container.matrix
Output = container.matrix

Inputs = TypeVar('Inputs', bound=types.Container)
Outputs = TypeVar('Outputs', bound=types.Container)

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    pass

class AdjacencySpectralEmbedding(TransformerPrimitiveBase[Input, Output, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            'id': 'b940ccbd-9e9b-3166-af50-210bfd79251b',
            'version': '0.3.0',
            'name': "Adjacency Spectral Embedding",
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        pass

    def embed(self, *, g : JHUGraph, dim: int):
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
