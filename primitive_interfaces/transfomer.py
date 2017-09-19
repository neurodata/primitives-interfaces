import abc
from typing import *

from .base import *

__all__ = ('TransformerPrimitiveBase',)


class TransformerPrimitiveBase(PrimitiveBase[Input, Output, None]):
    """
    A base class for primitives which are not fitted at all and can
    simply produce (useful) outputs from inputs directly. As such they
    also do not have any state (params).

    This class is parametrized using only two type variables, ``Input`` and ``Output``.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Sequence[Input]) -> Sequence[Output]:
        pass

    def set_training_data(self, *, inputs: None = None, outputs: None = None) -> None:
        """
        A noop.
        """

        return

    def fit(self, *, timeout: float = None, iterations: Optional[int] = 1) -> bool:
        """
        A noop.
        """

        return True

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, params: None) -> None:
        """
        A noop.
        """

        return
