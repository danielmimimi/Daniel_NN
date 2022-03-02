from numpy import ndarray
import numpy as np

from loss.loss import Loss

class MeanSquaredError(Loss):

    def __init__(self, normalize:bool = False) -> None:
        '''Pass'''
        super().__init__()
        self.normalize = normalize

    def _output(self) -> float:
        '''
        Computes the per-observation squared error loss
        '''
        if self.normalize:
            self.prediction = self.prediction / self.prediction.sum(axis=1, keepdims=True)
        
        loss = (
            np.sum(np.power(self.prediction - self.target, 2)) / 
            self.prediction.shape[0]
        )

        return loss

    def _input_grad(self) -> ndarray:
        '''
        Computes the loss gradient with respect to the input for MSE loss
        '''        

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]