# Zadanie3: Uzupełnij definicję poniżej klasy.
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
import numpy as np
from typing import List

class Adagrad(Optimizer):

    def __init__(self, initial_params, learning_rate, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gt = torch.zeros_like(self.params[0])
        self.gradMemory = []

        for param in self.params:
            self.gradMemory.append(torch.zeros_like(param))

    @torch.no_grad()
    def step(self):
        for param in self.params:
            self.gradMemory.append(torch.pow(param.grad, 2))
            self.gt = torch.sum(torch.stack(self.gradMemory), dim=0)
            param -= (self.learning_rate/(self.gt + self.epsilon)**0.5) * param.grad




#test_optimizer(Adagrad)
visualize_optimizer(Adagrad, n_steps=20, learning_rate=1.0, epsilon=1e-8)
plt.show()

