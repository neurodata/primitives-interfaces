#!/usr/bin/env python

# nonpar.py
# Copyright (c) 2017. All rights reserved.

import os
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import numpy as np

from typing import Sequence
from primitive_interfaces.transfomer import TransformerPrimitiveBase

Input = np.ndarray
Output = np.ndarray

class NonParametricClustering(TransformerPrimitiveBase[Input, Output]):

    def cluster(self, *, xhat1 : Input, xhat2 : Input, sigma : float = 0.5):
        """
        Non-parametric clustering

        **Positional Arguments:**

        xhat1:
            - A numpy.ndarray type "matrix"
        xhat2:
            - A numpy.ndarray type "matrix"

        **Optional Arguments:**

        sigma:
            - a sigma for the Gaussian kernel
        """

        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "nonpar.interface.R")

        cmd = """
        source("%s")
        fn <- function(xhat1, xhat2, sigma) {
            nonpar.interface(xhat1, xhat2, sigma)
        }
        """ % path

        return robjects.r(cmd)(xhat1, xhat2, sigma)

    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        #self.cluster(inputs[0], inputs[1])
        pass # TODO
