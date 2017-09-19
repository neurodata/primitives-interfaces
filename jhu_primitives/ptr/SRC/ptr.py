#!/usr/bin/env python

# ptr.py
# Created by Disa Mhembere, Heather Patsolic on 2017-09-11.
# Copyright (c) 2017. All rights reserved.

import os
import rpy2.robjects as robjects

def ptr(g):
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
    fn <- function(g) {
        ptr.interface(g)
    }
    """ % path

    return robjects.r(cmd)(g._object)
