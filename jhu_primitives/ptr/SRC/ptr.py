#!/usr/bin/env python

# ptr.py
# Created by Disa Mhembere, Heather Patsolic on 2017-09-11.
# Copyright (c) 2017. All rights reserved.

import os
from rpy2 import robjects
from typing import Sequence
import numpy as np

from primitive_interfaces.transfomer import TransformerPrimitiveBase
from jhu_primitives.core.JHUGraph import JHUGraph

Input = JHUGraph
Output = np.ndarray

class PassToRanks(TransformerPrimitiveBase[Input, Output]):
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        """
        TODO: YP description

        **Positional Arguments:**

        g:
            - An r igraph object repr in python
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "ptr.interface.R")
        cmd = """
        source("%s")
        fn <- function(inputs) {
            ptr.interface(inputs)
        }
        """ % path

        return np.array(robjects.r(cmd)(inputs._object))
