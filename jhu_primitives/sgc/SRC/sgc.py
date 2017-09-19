#!/usr/bin/env python

# sgc.py
# Copyright (c) 2017. All rights reserved.

import os
import rpy2.robjects as robjects

def sgc(g):
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

    return robjects.r(cmd)(g._object)
