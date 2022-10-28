# Zadanie5: Modele lasu drzew decyzyjnych często wykorzystują metodę modyfikowania danych treningowych - agregację bootstrapową (bootstraping).
# Polega ona na wielokrotnym losowaniu ze zwracaniem. Jeżeli takie losowanie powtózymy n
# razy, otrzymamy elementowy zbiór danych treningowych, w którym część przypadków będzie się powtarzać.
# Pokaż, że dla dużych n próba będzie zawierała średnio 63% przypadków z orginalnego zbioru.

from sklearn import datasets
import random
import numpy as np
import difflib
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# creating a list containing 100 different elements
y = range(1, 100)

def element_lotto(set,n):
    # create a list saving the amount of repeating elements from each bagging
    repeating_elements = []
    for i in range(n):
        new_set = []
        for j in range(len(set)):
            # bag a new set of items
            new_set.append(set[random.randint(0, len(set)-1)])
        # add count of repeating elements
        repeating_elements.append(len(np.unique(new_set)))
    return repeating_elements

print(np.mean(element_lotto(y,1000)))