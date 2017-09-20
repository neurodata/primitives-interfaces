#!/usr/bin/env python

# gclust.py
# Copyright (c) 2017. All rights reserved.

import os
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from typing import Sequence
from primitive_interfaces.transfomer import TransformerPrimitiveBase

Input = np.ndarray
Output = np.ndarray

class GaussianClustering(TransformerPrimitiveBase[Input, Output]):
    def cluster(self, *, inputs: Input, dim : int =2) -> int:
        """
        TODO: YP description

        **Positional Arguments:**

        inputs:
            - A matrix

        **Optional Arguments:**

        dim:
            - The number of clusters in which to assign the data
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "gclust.interface.R")
        cmd = """
        source("%s")
        fn <- function(X, dim) {
            gclust.interface(X, dim)
        }
        """ % path
        return int(robjects.r(cmd)(inputs, dim)[0])

    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        self.cluster(inputs=inputs)
