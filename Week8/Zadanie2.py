# Zadanie2: Rozważ dane iris. Dokonaj standaryzacji a następnie przeprowadź algorytm PCA.
# Która z cech wydaje się najistoniejsza? Ile komponentów wystarczająco dobrze opisuje te dane?
# Skomentuj otrzymane wyniki.


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
df_iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

X = df_iris.iloc[:,:-1]
Y = df_iris.iloc[:,-1]

stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)

cov_mat = np.cov(X_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('Eigenvalues \n%s'% eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]

plt.figure(figsize=(8,6))
plt.bar( range(1, 5), var_exp, alpha=0.5,   align='center', label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

# Dane sa dobrze opisane przez dwa eigenvectory, ktore razem objasniaja 90% zmiennosci zawartej w datasecie.

