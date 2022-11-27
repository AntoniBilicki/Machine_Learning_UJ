# Zadanie3: Rozważ jeszcze raz dane dotyczące twarzy różnych osób (fetch_lw_people) rozważane podczas ćwiczenia z SVM.
# Zastosuj podobnie jak wtedy PCA ze 150 komponentnami. Jaki procent wariancji opisują te komponenty?
# Narsuj kilka początkowych komponentów (wektory własne); oczywiście podobnie jak obrazy są to wektory 64 * 47
# wymiarowe - przed narysowaniem należy jeszcze zastosować metodę reshape. Zastanów się nad ich interpretacją.

from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people

data = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
