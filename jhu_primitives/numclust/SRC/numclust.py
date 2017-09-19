#!/usr/bin/env python

# numclust.py
# Copyright (c) 2017. All rights reserved.

import os
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def numclust(X):
    """
    TODO: YP description

    **Positional Arguments:**

    X:
        - TODO: YP description
    """

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "numclust.interface.R")
    cmd = """
    source("%s")
    fn <- function(X) {
        numclust.interface(X)
    }
    """ % path

    return robjects.r(cmd)(X)
