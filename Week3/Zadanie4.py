# Zadanie3: Uzupełnij definicję poniżej klasy.
import matplotlib.pyplot as plt
import torch
from utils import visualize_optimizer
from checker import test_optimizer
from Zadanie3 import Optimizer
from typing import List

class Adagrad(Optimizer):

    def __init__(self, initial_params, learning_rate, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gt = 0

        @torch.no_grad()
        def step(self):
            for idx, param in enumerate(self.params):
            pass

