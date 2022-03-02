import numpy as np
from numpy import ndarray
from base.operation import Operation
from layer.dropout import Dropout
from layer.layer import Layer
from operations.biasAdd import BiasAdd
from operations.sigmoid import Sigmoid
from operations.weightedMulitply import WeightMultiply


class Dense(Layer):
    '''
    A fully connected layer which inherits from "Layer"
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid(),
                 dropout: float = 1.0,
                 weight_init: str = "standard"):
        '''
        Requires an activation function upon initialization
        '''
        super().__init__(neurons)
        self.activation = activation
        self.weight_init = weight_init
        self.dropout =  dropout

    def _setup_layer(self, input_: ndarray) -> None:
        '''
        Defines the operations of a fully connected layer.
        '''
        if self.seed:
            np.random.seed(self.seed)

        num_in = input_.shape[1]
        if self.weight_init == "glorot":
            scale = 2/(num_in + self.neurons)
        else:
            scale = 1.0

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None