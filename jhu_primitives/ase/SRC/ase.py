#!/usr/bin/env python

# ase.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar
import os

from primitive_interfaces.transfomer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph

Input = TypeVar('Input')
Output = TypeVar('Output')

class AdjacencySpectralEmbedding(TransformerPrimitiveBase[Input, Output]):
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

        return robjects.r(cmd)(g._object, dim)
