import numpy as np
import matplotlib.pyplot as plt

N = 100
X = np.random.multivariate_normal([5, 5], [[3, 20], [20, 50]], N)
X -= X.mean(axis=0)

cov_mat = np.cov(X, rowvar=False)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

indice = np.argsort(eigen_vals)[::-1]
eigen_vals = eigen_vals[indice]
eigen_vecs = eigen_vecs[:, indice]

X_ = X.dot(eigen_vecs)

plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.scatter(X_[:, 0], X_[:, 1], marker='x')
plt.grid()
plt.show()
