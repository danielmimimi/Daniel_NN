from numpy import ndarray
import numpy as np
from base.operation import Operation


class ReLU(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        mask = self.output >= 0
        return output_grad * mask