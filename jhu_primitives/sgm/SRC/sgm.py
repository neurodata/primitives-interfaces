#!/usr/bin/env python

# sgm.py
# Copyright (c) 2017. All rights reserved.

import os
import rpy2.robjects as robjects

def sgm(g1, g2, numseeds):
    """
    TODO: YP description

    **Positional Arguments:**

    g1:
        - The first input graph object - in JHUGraph format
    g2:
        - The second input graph object - in JHUGraph format

    numseeds:
        - the number of seeds, assumed to be the first "seeds" vertices
          in both graphs with identity correspondence
    """

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "sgm.interface.R")

    cmd = """
    source("%s")
    fn <- function(g1, g2, s) {
        sgm.interface(g1, g2, s)
    }
    """ % path

    return robjects.r(cmd)(g1._object, g2._object, numseeds)
