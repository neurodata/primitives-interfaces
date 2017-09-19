#!/usr/bin/env python

# nonpar.py
# Copyright (c) 2017. All rights reserved.

import os
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def nonpar(xhat1, xhat2, sigma=0.5):
    """
    TODO: YP description

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
