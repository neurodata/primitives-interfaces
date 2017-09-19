#!/usr/bin/env python

# dimselect.py
# Created by Disa Mhembere on 2017-09-11.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.

import os
import rpy2.robjects as robjects

def dimselect(X):
    """
    TODO: YP description

    **Positional Arguments:**

    X:
        - Input data matrix TODO: YP format
    """

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "dimselect.interface.R")
    cmd = """
    source("%s")
    fn <- function(X) {
        dimselect.interface(X)
    }
    """ % path

    return robjects.r(cmd)(X)
