#!/usr/bin/env python

# JHUEvaluator.py

from sklearn.metrics import adjusted_rand_score
from Evaluator import Evaluator

class JHUEvaluator(Evaluator):
    def __init__(self):
        pass

    def get_accuracy(self, **kwargs):
        """

        Use ARI to evaluate our procedure

        ** Keyword Arguments **:

        predicted_labels:
            - The predicted labels from your model

        true_labels:
            - The true known labels from your model
        """
        if "predicted_labels" in kwargs and "true_labels" in kwargs:
            return 100*(adjusted_rand_score(kwargs["predicted_labels"],
                kwargs["true_labels"]))
        else:
            return 0

def test():
    ev = JHUEvaluator()
    print("Ev: ", ev.get_accuracy(predicted_labels=[1,2,3,4,5],
            true_labels=[5,4,3,2,1]))

# test() # Not run
