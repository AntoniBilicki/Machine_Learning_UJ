import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
boston_data = load_boston()



d = boston_data['data']
d2 = d[:, [2,5]] #wyciągamy tylko 2 cechy: INDUS, RM
target = boston_data['target']

# Klasa dedykowana regresji liniowej 2 zmiennych

class LinearModel2v:
    def __init__(self, eta=0.001, diff=0.001, w1=1, w2=1, w0=1, maxiter=1000, alfa=0, regression='L1'):
        self.eta = eta
        self.diff = diff
        self.w1 = w1
        self.w2 = w2
        self.w0 = w0
        self.maxiter = maxiter
        self.alfa = alfa
        self.regression = regression #parametr odpowiedzalny za wybor regularyzacji. Mozliwe opcje: 'L0' 'L1' 'L2'

    def loss_function(self, X, t):
        N = len(X)
        C = 0
        for i in range(N):
            C += (X[i][0] * self.w1 + X[i][1] * self.w2 + self.w0 - t[i]) ** 2
            # if math.isinf(C) or math.isnan(C):
            #     breakpoint()


        #Dodaje parametr LF+ odpowiedzialny za korekte.
        match self.regression:
            case 'L0':
                lfp = 0
            case 'L1':
                lfp = self.alfa * (abs(self.w1) + abs(self.w2) + abs(self.w0))
            case 'L2':
                lfp = self.alfa * (self.w1**2 + self.w2**2 + self.w0**2)

        return C / (2 * N) + lfp

    def update_weights(self, X, t):
        N = len(X)
        dC1 = 0
        dC2 = 0
        dC0 = 0
        for i in range(N):
            y_pred = X[i][0] * self.w1 + X[i][1] * self.w2 + self.w0
            dC1 += 2 * X[i][0] * (y_pred - t[i])
            dC2 += 2 * X[i][1] * (y_pred - t[i])
            dC0 += 2 * (y_pred - t[i])

        self.w1 = self.w1 - self.eta * dC1 / (2 * N)
        self.w2 = self.w2 - self.eta * dC2 / (2 * N)
        self.w0 = self.w0 - self.eta * dC0 / (2 * N)

        # if self.w1 > 1e100:
        #     breakpoint()

    def train(self, X, t):
        l = []
        ile = 0
        while True:
            l.append(self.loss_function(X, t))
            self.update_weights(X, t)
            ile += 1
            if len(l) > 2:
                if abs(l[-1] - l[-2]) / l[-1] < self.diff or ile > self.maxiter:
                    break


X_train, X_test, y_train, y_test = train_test_split(d2, target, test_size=0.4, random_state=42)
X_walid, X_test, y_walid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


tested_ni = list(np.logspace(-5, 0, num=30))
tested_alfa = list(np.logspace(-10, 1, num=10))
combo = list(itertools.product(tested_ni, tested_alfa))

result = []

for i in combo:
    model = LinearModel2v(eta=i[0], alfa=i[1])
    model.train(X_walid, y_walid)
    result.append(model.loss_function(X_walid, y_walid))

print(f'Najmniejsza wartosc funkcji kosztu otrzymano dla eta = {combo[result.index(min(result))][0]}, alfa = {combo[result.index(min(result))][1]} ')


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

for idx, x in enumerate(result):
    ax.scatter(np.log10(combo[idx][0]), np.log10(combo[idx][1]), result[idx])

plt.xlabel('log10 eta')
plt.ylabel('log10 alfa')
ax.set_zlabel('Wartosc funkcji kosztu')
plt.show()
