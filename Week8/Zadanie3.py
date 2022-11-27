# Zadanie3: Rozważ jeszcze raz dane dotyczące twarzy różnych osób (fetch_lw_people) rozważane podczas ćwiczenia z SVM.
# Zastosuj podobnie jak wtedy PCA ze 150 komponentnami. Jaki procent wariancji opisują te komponenty?
# Narsuj kilka początkowych komponentów (wektory własne); oczywiście podobnie jak obrazy są to wektory 64 * 47
# wymiarowe - przed narysowaniem należy jeszcze zastosować metodę reshape. Zastanów się nad ich interpretacją.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

data = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = data.images.shape

X = data.data
n_features = X.shape[1]

y = data.target
target_names = data.target_names
n_classes = target_names.shape[0]

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=150, svd_solver="randomized", whiten=True).fit(X_std)
eigenfaces = pca.components_.reshape((150, h, w))


var_exp_cum = np.cumsum(pca.explained_variance_ratio_)
plt.bar(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_, align='center', label='individual explained variance')
plt.step(range(len(pca.explained_variance_ratio_)), var_exp_cum, where='mid', label='cumulative explained variance',color = "r")
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('% of variation described by components')
plt.legend(loc='best')
plt.show()


fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5

for i in range(1, columns*rows +1):
    img = eigenfaces[i-1].reshape((h, w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap=plt.cm.gray)
plt.show()

