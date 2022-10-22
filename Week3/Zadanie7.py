# Zadanie7: Uzupełnij definicję poniżej klasy (Adadelta).
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
import numpy as np
from typing import List


class Adam(Optimizer):

    def __init__(self, initial_params, learning_rate, beta1, beta2, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mi = torch.zeros_like(self.params[0])
        self.vi = torch.zeros_like(self.params[0])
        self.count = 0


    @torch.no_grad()
    def step(self):
        for idx, param in enumerate(self.params):
            self.mi = self.beta1*self.mi + (1-self.beta1)*param.grad
            self.vi = self.beta2*self.vi + (1-self.beta2)*torch.pow(param.grad, 2)
            licznik = self.mi / (1-self.beta1**(self.count+1))
            mianownik_part = self.vi / (1-self.beta2**(self.count+1))
            param -= self.learning_rate * licznik / (mianownik_part**0.5 + self.epsilon)
            self.count += 1




visualize_optimizer(Adam, n_steps=20, learning_rate=0.35, beta1=0.9, beta2=0.999, epsilon=1e-8)
plt.show()