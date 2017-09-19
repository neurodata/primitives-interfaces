import abc
from typing import *

from .base import *

__all__ = ('SupervisedLearnerPrimitiveBase',)


class SupervisedLearnerPrimitiveBase(PrimitiveBase[Input, Output, Params]):
    """
    A base class for primitives which have to be fitted on both input and output data
    before they can start producing (useful) outputs from inputs.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Sequence[Input], outputs: Sequence[Output]) -> None:
        pass
