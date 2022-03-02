class Optimizer(object):
    '''
    Base class for a neural network optimizer.
    '''
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None) -> None:
        '''
        Every optimizer must have an initial learning rate.
        '''
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True


    def _decay_lr(self) -> None:

        if not self.decay_type:
            return

        if self.decay_type == 'exponential':
            self.lr *= self.decay_per_epoch

        elif self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch

    def step(self,
             epoch: int = 0) -> None:
        '''
        Every optimizer must implement the "step" function.
        '''

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()