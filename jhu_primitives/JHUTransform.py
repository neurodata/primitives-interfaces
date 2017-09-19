#!/usr/bin/env python

# JHUTransform.py

import sys, os
from Transform import Transform
import numpy as np

sys.path.append(os.path.abspath("../"))

from ase.SRC.ase import ase
from lse.SRC.lse import lse
from ptr.SRC.ptr import ptr
from dimselect.SRC.dimselect import dimselect
from sgm.SRC.sgm import sgm
from gclust.SRC.gclust import gclust
from oocase.SRC.oocase import oocase # FIXME: Expects a FlashGraph object
from nonpar.SRC.nonpar import nonpar
from sgc.SRC.sgc import sgc
from numclust.SRC.numclust import numclust
from vnsgm.SRC.vnsgm import vnsgm

class JHUTransform(Transform):

    def ase_transform(self, g, dim=2):
        """
        TODO: YP document
        """
        return np.array(ase(g, dim))

    def lse_transform(self, g, dim=2):
        """
        TODO: YP document
        """
        return np.array(ase(g, dim))

    def ptr_transform(self, g):
        """
        TODO: YP document
        """
        return np.array(ptr(g))

    def dimselect_transform(self, X):
        """
        TODO: YP document
        """
        return np.array(dimselect(X))

    def sgm_transform(self, g1, g2, numseeds):
        """
        TODO: YP document
        """
        return np.array(sgm(g1, g2, numseeds))

    def gclust_transform(self, X, dim=2):
        """
        TODO: YP document
        """
        return gclust(X, dim)[0]

    def oocase_transform(self, g, dim=2):
        """
        TODO: YP document
        """
        return oocase(g, dim)

    def nonpar_transform(self, mat1, mat2, sigma=.5):
        """
        TODO: YP document
        """
        return nonpar(mat1, mat2, sigma)

    def sgc_transform(self, g):
        """
        TODO: YP document
        """
        return np.array(sgc(g))

    def numclust_transform(self, X):
        """
        TODO: YP document
        """
        return numclust(X)[0]

    def vnsgm_transform(self, g, g2, voi, numseeds):
        """
        TODO: YP document
        """
        return np.array(vnsgm(g, g2, voi, numseeds))
