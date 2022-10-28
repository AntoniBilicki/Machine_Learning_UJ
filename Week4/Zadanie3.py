# Zadanie3: Uzasadnij wartość indeksu Giniego w datasecie poprzez bezpośrednie obliczenia.

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

def calc_gini(y):
    uniq_elements, uniq_counts = np.unique(y, return_counts=True)
    uniq_weights = uniq_counts/len(y)
    gini = 1
    for weight in uniq_weights:
        gini -= weight**2
    return gini

print(calc_gini(y_train))