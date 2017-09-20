#!/usr/bin/env python

# sgc.py
# Copyright (c) 2017. All rights reserved.

import os
from rpy2 import robjects
from typing import Sequence
import numpy as np

from primitive_interfaces.transfomer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph

Input = JHUGraph
Output = np.ndarray


class SpectralGraphClustering(TransformerPrimitiveBase[Input, Output]):
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        """
        TODO: YP description

        **Positional Arguments:**

        g:
            - A graph in R 'igraph' format
        """
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "sgc.interface.R")
        cmd = """
        source("%s")
        fn <- function(g) {
            sgc.interface(g)
        }
        """ % path

        return np.array(robjects.r(cmd)(inputs._object))
