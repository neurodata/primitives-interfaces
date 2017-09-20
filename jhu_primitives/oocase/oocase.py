#!/usr/bin/env python

# oocase.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, Any, TypeVar
import os

from primitive_interfaces.transfomer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph

Input = TypeVar('Input')
Output = TypeVar('Output')

class OutOfCoreAdjacencySpectralEmbedding(TransformerPrimitiveBase[Input, Output]):

    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    def embed(self, *, g : JHUGraph, dim: int = 2):
        """
        TODO: YP description

        **Positional Arguments:**

        g:
            - A graph

        **Optional Arguments:**

        dim:
            - The number of dimensions in which to embed the data
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "oocase.interface.R")

        cmd = """
        source("%s")
        fn <- function(g, dim) {
            oocase.interface(g, dim)
        }
        """ % path

        return robjects.r(cmd)(g._object, dim)
