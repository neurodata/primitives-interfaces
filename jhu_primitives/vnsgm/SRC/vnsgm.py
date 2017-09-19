#!/usr/bin/env python

# vnsgm.py
# Copyright (c) 2017. All rights reserved.

import os
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def vnsgm(g1, g2, voi, s):
    """
    TODO: YP description

    **Positional Arguments:**

    g1:
        - A graph in R igraph format
    g2:
        - A graph in R igraph format
    voi:
        - vector of indices for vertices of interest
    s:
        - the number of seeds, assumed to be the first "seeds" vertices
                  in both graphs with identity correspondence
    """

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "vnsgm.interface.R")
    cmd = """
    source("%s")
    fn <- function(g1, g2, voi, s) {
        vnsgm.interface(g1, g2, voi, s)
    }
    """ % path

    return robjects.r(cmd)(g1._object, g2._object, voi, s)
