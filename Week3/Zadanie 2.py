# Zadanie2: Przetestować inne wartości (np. 10, 50) dla danych ponizej.
# Skomentować wyniki w konkteście definicji parametru gamma.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets

# generate data
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

############################################

# normalization of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# stitching data together
X_all = np.vstack((X_train_std, X_test_std))
y_all = np.hstack((y_train, y_test))


def plot_SVC(gamm, nplot):

    plt.subplot(3, 3, nplot)
    svm = SVC(kernel='rbf', C=1., gamma=gamm, random_state=0)
    svm.fit(X_train_std, y_train)

    plot_decision_regions(X=X_all, y=y_all, clf=svm)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title(f'SVC plot for gamma = {gamm}')


tested_gamma = [0.1, 0.5, 1, 2, 4, 8, 20, 40, 80]

for idx, gamm in enumerate(tested_gamma):
    plot_SVC(gamm, idx+1)

plt.tight_layout(pad=1.0)
plt.show()

# Comment: gamma parameter seems to influence how tightly the classification area is wrapped around its objects.