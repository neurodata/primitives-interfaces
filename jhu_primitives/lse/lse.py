#!/usr/bin/env python

# lse.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Dict, Sequence, TypeVar
import os

from primitive_interfaces.transfomer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph
import numpy as np

Input = TypeVar('Input')
Output = TypeVar('Output')

class LaplacianSpectralEmbedding(TransformerPrimitiveBase[Input, Output]):
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    def embed(self, *, g : JHUGraph, dim: int = 2):
        """
        Perform Laplacian Spectral Embedding on a graph
        TODO: YP description

        **Positional Arguments:**

        g:
            - Graph in JHUGraph format

        **Optional Arguments:**

        dim:
            - The number of dimensions in which to embed the data
        """
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "lse.interface.R")
        cmd = """
        source("%s")
        fn <- function(g, dim) {
            lse.interface(g, dim)
        }
        """ % path

        return np.array(robjects.r(cmd)(g._object, dim))
