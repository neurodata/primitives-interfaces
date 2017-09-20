#!/usr/bin/env python
# sgm.py
# Copyright (c) 2017. All rights reserved.

from rpy2 import robjects
from typing import Sequence, TypeVar, Any
import os

from primitive_interfaces.transfomer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph

Input = TypeVar('Input')
Output = TypeVar('Output')

class SeededGraphMatching(TransformerPrimitiveBase[Input, Output]):
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    def match(self, *, g1: JHUGraph, g2: JHUGraph, seeds: Any = 0):
        """
        TODO: YP description

        **Positional Arguments:**

        g1:
            - The first input graph object - in JHUGraph format
        g2:
            - The second input graph object - in JHUGraph format

        seeds:
            - The matrix of seed indices. The first column corresponds to seed index
              for graph 1 and second column corresponds to seed index for
              graph 2, where each row corresponds to a seed pair.
              If empty, assumes no seeds are used.
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "sgm.interface.R")

        cmd = """
        source("%s")
        fn <- function(g1, g2, s) {
            sgm.interface(g1, g2, s)
        }
        """ % path

        return robjects.r(cmd)(g1._object, g2._object, seeds)
