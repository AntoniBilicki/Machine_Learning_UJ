# Zadanie6: Uzupełnij definicję poniżej klasy (Adadelta).
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
import numpy as np
from typing import List


class RMSProp(Optimizer):

    def __init__(self, initial_params, gamma, epsilon):
        super().__init__(initial_params)
        self.gamma = gamma
        self.epsilon = epsilon
        self.ht = torch.zeros_like(self.params[0])
        self.dt = torch.zeros_like(self.params[0])

        self.gradMemory = []

        for param in self.params:
            self.gradMemory.append(torch.zeros_like(param))

    @torch.no_grad()
    def step(self):
        for param in self.params:
            self.gradMemory.append(torch.pow(param.grad, 2))
            self.ht = self.gamma*self.ht + (1-self.gamma)*torch.pow(param.grad, 2)
            param -= (self.learning_rate / (self.ht + self.epsilon) ** 0.5) * param.grad



visualize_optimizer(Adagrad, n_steps=20, learning_rate=1.0, epsilon=1e-8)
plt.show()

