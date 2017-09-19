import abc
from typing import *

from .base import *

__all__ = ('GraphPrimitiveBase',)


class GraphPrimitiveBase(PrimitiveBase[Input, Output, Params]):
    """
    A base class for primitives which take Graph objects as input.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

