# Zadanie6: Uzupełnij definicję poniżej klasy (Adadelta).
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
import numpy as np
from typing import List


class Adadelta(Optimizer):

    def __init__(self, initial_params, gamma, epsilon):
        super().__init__(initial_params)
        self.gamma = gamma
        self.epsilon = epsilon
        self.ht = torch.zeros_like(self.params[0])
        self.dt = torch.zeros_like(self.params[0])

    @torch.no_grad()
    def step(self):
        for param in self.params:
            self.ht = self.gamma*self.ht + (1-self.gamma)*torch.pow(param.grad, 2)
            term = ((self.dt + self.epsilon)**0.5 / (self.ht + self.epsilon) ** 0.5) * param.grad
            param -= term
            self.dt = self.gamma*self.dt + (1-self.gamma)*term**2



visualize_optimizer(Adadelta, n_steps=20, epsilon=5e-2, gamma=0.9)
plt.show()

