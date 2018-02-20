#!/usr/bin/env python

# vnsgm.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Any
import os
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from primitive_interfaces.transformer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph
import numpy as np

Input = TypeVar('Input')
Output = TypeVar('Output')
Params = TypeVar('Params')

class
VertexNominationSeededGraphMatching(TransformerPrimitiveBase[Input, Output, Params]):
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    def match(self, *, g1 : JHUGraph, g2 : JHUGraph, voi : np.array, seeds: Input):
        """
        TODO: YP description

        **Positional Arguments:**

        g1:
            - A graph in R igraph format
        g2:
            - A graph in R igraph format
        voi:
            - vector of indices for vertices of interest
        seeds:
            - the matrix of seeds, s x 2 where s is number of seeds and
              column i seeds are for graph i
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "vnsgm.interface.R")
        cmd = """
        source("%s")
        fn <- function(g1, g2, voi, seeds) {
            vnsgm.interface(g1, g2, voi, seeds)
        }
        """ % path

        return np.array(robjects.r(cmd)(g1._object, g2._object, voi, seeds))
