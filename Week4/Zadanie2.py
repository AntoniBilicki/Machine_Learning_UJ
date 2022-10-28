# Zadanie2: Uzasadnij wartość entropii = 1.58, dla podanego datasetu poprzez bezpośrednie obliczenia.

from sklearn import datasets
import numpy as np
import math
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)


def calc_entropy(y):
    uniq_elements, uniq_counts = np.unique(y, return_counts=True)
    uniq_weights = uniq_counts/len(y)
    entropy = 0
    for weight in uniq_weights:
        entropy -= weight*math.log2(weight)
    return entropy

print(calc_entropy(y_train))