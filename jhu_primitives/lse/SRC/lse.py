#!/usr/bin/env python

# lse.py
# Created by Disa Mhembere on 2017-09-12.
# Email: disa@jhu.edu
# Copyright (c) 2017. All rights reserved.

import argparse
import rpy2.robjects as robjects

def lse(datafn, dim=2):
    """
    Perform Laplacian Spectral Embedding on a graph
    TODO: YP description

    **Positional Arguments:**

    g:
        - Graph in JHUGraph format

    **Optional Arguments:**

    dim:
        - The number of dimensions in which to embed the data
    """

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "lse.interface.R")
    cmd = """
    source("%s")
    fn <- function(g, dim) {
        lse.interface(g, dim)
    }
    """ % path

    _lse = robjects.r(cmd)
    return _lse(g._object, dim)
