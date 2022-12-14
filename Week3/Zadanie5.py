# Zadanie5: Uzupełnij definicję poniżej klasy (RMSPROP).
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
import numpy as np
from typing import List


class RMSProp(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.ht = torch.zeros_like(self.params[0])

    @torch.no_grad()
    def step(self):
        for param in self.params:
            self.ht = self.gamma*self.ht + (1-self.gamma)*torch.pow(param.grad, 2)
            param -= (self.learning_rate / (self.ht + self.epsilon) ** 0.5) * param.grad



visualize_optimizer(RMSProp, n_steps=10, learning_rate=0.5, gamma=0.9, epsilon=1e-8)
plt.show()

