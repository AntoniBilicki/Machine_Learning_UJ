# Zadanie1: Zdefiniuj klasę implementującą regresję liniową z regularyzacją L1/L2 dla dowolnej liczby zmiennych.
import numpy as np

class LinearModel2v:
    def __init__(self, eta=0.001, diff=0.001, w1=1, w2=1, w0=1, maxiter=1000, alfa=0, regression='L1'):
        self.eta = eta
        self.diff = diff
        self.w1 = w1
        self.w2 = w2
        self.w0 = w0
        self.maxiter = maxiter
        self.alfa = alfa
        self.regression = regression  # argument responsible for regularization choice. Possible options: 'L0' 'L1' 'L2'

    def loss_function(self, X, t):  # loss_function responsible for determining regression score
        N = len(X)
        C = 0
        for i in range(N):
            C += (X[i][0] * self.w1 + X[i][1] * self.w2 + self.w0 - t[i]) ** 2

        return C / (2 * N) + self._calc_regularization()

    def _calc_regularization(self):

        match self.regression:  # now adding LF+ parameter responsible for regularization
            case 'L1':
                lfp = self.alfa * (abs(self.w1) + abs(self.w2) + abs(self.w0))
            case 'L2':
                lfp = self.alfa * (self.w1**2 + self.w2**2 + self.w0**2)
            case _:
                lfp = 0
        return lfp

    def _predict_y_hat(self, X):

        #doesnt implement more than two weights but will work for now
        w = np.array([self.w1, self.w2])
        X = np.array(X)
        return list((X*w).sum(axis=1) + self.w0)


    def update_weights(self, X, t):  # function responsible for updating model weights, will break for large 'eta' values
        N = len(X)
        dC1 = 0
        dC2 = 0
        dC0 = 0
        y_hat = self._predict_y_hat(X)
        for i in range(N):
            dC1 += 2 * X[i][0] * (y_hat[i] - t[i])
            dC2 += 2 * X[i][1] * (y_hat[i] - t[i])
            dC0 += 2 * (y_hat[i] - t[i])

        #wasnt sure if I should recalculate lf as I go or use a static value. Decided to go with 2nd approach
        lf = self._calc_regularization()
        self.w1 = self.w1 - self.eta * (dC1 / (2 * N) + lf)
        self.w2 = self.w2 - self.eta * (dC2 / (2 * N) + lf)
        self.w0 = self.w0 - self.eta * (dC0 / (2 * N) + lf)`

        # if self.w1 > 1e100:
        #     breakpoint()

    def train(self, X, t):  # calls update_weights, until difference in loss_function is < diff or limit is reached
        l = []
        ile = 0
        while True:
            l.append(self.loss_function(X, t))
            self.update_weights(X, t)
            ile += 1
            if len(l) > 2:
                if abs(l[-1] - l[-2]) / l[-1] < self.diff or ile > self.maxiter:
                    break
