# Zadanie1: Rozważ poniższe dane. Zastosuj algorytm SVM dla różnych parametrów C: 0.01, 10.
# Zwizualizuj i skomentuj w kilku zdaniach otzymane wyniki.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from mlxtend.plotting import plot_decision_regions



X, y = make_blobs(n_samples=200, random_state=1, n_features=2, centers = 2, cluster_std = 2.4)

#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

####################

# normalization of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# creating first model
svm = SVC(kernel='linear', C=0.01, random_state=0)
svm.fit(X_train_std, y_train)

# creating second model
svm2 = SVC(kernel='linear', C=10, random_state=0)
svm2.fit(X_train_std, y_train)

plt.figure(figsize=(8, 6))

X_all = np.vstack((X_train_std, X_test_std))
y_all = np.hstack((y_train, y_test))

plt.subplot(2,1,1)
plot_decision_regions(X=X_all, y=y_all, clf=svm)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Wykres SVM dla hiperparametru C = 0.01')
L = plt.legend(loc='upper left')
L.get_texts()[0].set_text('Blob A')
L.get_texts()[1].set_text('Blob B')

plt.subplot(2,1,2)
plot_decision_regions(X=X_all, y=y_all, clf=svm2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Wykres SVM dla hiperparametru C = 10')
L = plt.legend(loc='upper left')
L.get_texts()[0].set_text('Blob A')
L.get_texts()[1].set_text('Blob B')
plt.show()

# Answer: High values of C result in a tighter fit to the data in order to avoid missclassification,
# but also tend to overfit the resulting model.