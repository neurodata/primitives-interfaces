#!/usr/bin/env python

# dimselect.py
# Copyright (c) 2017. All rights reserved.

import os
from rpy2 import robjects
import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from typing import Sequence
from primitive_interfaces.transfomer import TransformerPrimitiveBase

Input = np.ndarray
Output = np.ndarray

class DimensionSelection(TransformerPrimitiveBase[Input, Output]):
    """
    Select the right number of dimensions within which to embed given
    an adjacency matrix
    """

    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        """
        Run the dimension selection algorithm

        **Positional Arguments:**

        inputs:
            - A data matrix

        ** Returns: ***
        The right number of dimensions within which to embed

        """
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "dimselect.interface.R")
        cmd = """
        source("%s")
        fn <- function(X) {
            dimselect.interface(X)
        }
        """ % path

        return np.array(robjects.r(cmd)(inputs))
