from numpy import ndarray
import numpy as np
from base.operation import Operation


class Tanh(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad * (1 - self.output * self.output)