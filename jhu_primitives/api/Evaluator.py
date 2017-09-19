#!/usr/bin/env python

# Evaluator.py

import abc

class Evaluator(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_accuracy(self, **kwargs):
        return 0
