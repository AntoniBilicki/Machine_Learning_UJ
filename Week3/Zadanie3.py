# Zadanie3: Uzupełnij definicję poniżej klasy.
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer

from typing import List


class Optimizer:
    """Base class for each optimizer"""

    def __init__(self, initial_params):
        # store model weights
        self.params = initial_params

    def step(self):
        """Updates the weights stored in self.params"""
        raise NotImplementedError()

    def zero_grad(self):
        """Torch accumulates gradients, so we need to clear them after every update"""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


class GradientDescent(Optimizer):

    def __init__(self, initial_params: List[torch.tensor], learning_rate):
        super().__init__(initial_params)
        self.learning_rate = learning_rate

    @torch.no_grad()
    def step(self):
        for param in self.params:
            param -= self.learning_rate * param.grad


class Momentum(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma):
        super().__init__(initial_params)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.oldUpdtVec = 0
        self.deltas = []

        for param in self.params:
            self.deltas.append(torch.zeros_like(param))

    @torch.no_grad()
    def step(self):
        for idx, param in enumerate(self.params):
            self.oldUpdtVec = self.gamma*self.oldUpdtVec + self.learning_rate * param.grad
            param -= self.gamma*self.oldUpdtVec + self.learning_rate * param.grad



#visualize_optimizer(GradientDescent, n_steps=20, learning_rate=0.1, title='Za mały LR')

visualize_optimizer(Momentum, n_steps=20, learning_rate=0.05, gamma=0.8)
plt.show()


