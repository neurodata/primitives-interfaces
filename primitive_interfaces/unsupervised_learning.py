import abc
from typing import *

from .base import *

__all__ = ('UnsupervisedLearnerPrimitiveBase',)


class UnsupervisedLearnerPrimitiveBase(PrimitiveBase[Input, Output, Params]):
    """
    A base class for primitives which have to be fitted before they can start
    producing (useful) outputs from inputs, but they are fitted only on input data.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Sequence[Input], outputs: None = None) -> None:
        pass
