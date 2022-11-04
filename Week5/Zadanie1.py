# Zadanie1: Zaimplementuj algorytm grupowania górskiego Zastosuj do poniższych danych.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import math

X, Y = make_blobs(n_samples=200, random_state=100, n_features=2, centers = 3, cluster_std = 1.2)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

################

# assign initial parameters, circle radius size and inputs
class mountain_clustering:

    def __init__(self, x, radius):
        self.data = x
        self.radius = radius
        self.y = [0]*len(x[:, 0])
        self.density = self.calculate_density(self.data, self.y, self.radius)
        self.currentCluster = 0


    def calculate_density(self, x, y, radius):
        """determine density for each point by counting distance to other points
        (yes, this is slow and inefficient for large datasets)"""

        density = []
        for main_idx, pair in enumerate(x[:,0]):
            density_score = 0
            for sub_idx, ele in enumerate(x[:,0]):
                distance = (math.dist((x[main_idx, 0], x[main_idx, 1]), (x[sub_idx, 0], x[sub_idx, 1])))
                if distance < radius and y[sub_idx] == 0:
                    density_score += 1
            density.append(density_score)
        return density

    def classify_radius(self, x, density):
        """Classifies samples around the most dense point based on radius"""

        self.currentCluster += 1
        dense_idx = density.index(max(density))
        # if distance to max_dense is lower than radius, and point isn't classified yet, classify it
        for idx, pair in enumerate(x[:, 0]):
            if self.y[idx] == 0:
                distance = math.dist((x[dense_idx, 0], x[dense_idx, 1]), (x[idx, 0], x[idx, 1]))
                if distance < self.radius:
                    self.y[idx] = self.currentCluster
        self.density = self.calculate_density(self.data, self.y, self.radius)


A = mountain_clustering(X, 4)

LABEL_COLOR_MAP = {0: 'r',
                   1: 'k',
                   2: 'b',
                   3: 'g'}

plt.scatter(X[:, 0], X[:, 1], c=[LABEL_COLOR_MAP[l] for l in A.y])
plt.show()

A.classify_radius(A.data, A.density)

plt.scatter(X[:, 0], X[:, 1], c=[LABEL_COLOR_MAP[l] for l in A.y])
plt.show()

A.classify_radius(A.data, A.density)

plt.scatter(X[:, 0], X[:, 1], c=[LABEL_COLOR_MAP[l] for l in A.y])
plt.show()

A.classify_radius(A.data, A.density)

plt.scatter(X[:, 0], X[:, 1], c=[LABEL_COLOR_MAP[l] for l in A.y])
plt.show()