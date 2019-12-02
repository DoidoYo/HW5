import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.spatial import distance
from scipy import stats
import statistics
from collections import defaultdict


# TODO: You can import anything from numpy or scipy here!

def normalize_data(X):
    var_x = np.std(X, axis=0)
    mean_x = np.mean(X, axis=0)

    std_X = np.zeros(X.shape)
    for idx, val in enumerate(var_x):
        if val == 0:  # if variance is zero, only subtract mean
            val = 1
        std_X[:, idx] = (X[:, idx] - mean_x[idx]) / val

    return std_X

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class PCA(Model):

    def __init__(self, X, target_dim):
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]
        self.target_dim = target_dim
        self.W = None

    def fit(self, X):
        std_X = normalize_data(X)
        cov = np.cov(std_X.T)
        A, Q = la.eig(cov)
        sorted_idx = np.flip(np.argsort(A))[:self.target_dim]
        self.W = Q[:, sorted_idx]

        return std_X @ self.W


class LLE(Model):

    def __init__(self, X, target_dim, lle_k):
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]

        self.target_dim = target_dim
        self.k = lle_k

        self.W = np.zeros((self.num_x, self.num_x))

    def fit(self, X):
        std_X = normalize_data(X)

        # get neighbors 1
        dist = distance.cdist(std_X, std_X, 'euclidean')
        neighs = np.argsort(dist, axis=0)[1:self.k+1, :].T

        # 2. Solve for reconstruction weights
        for idx, i in enumerate(std_X):
            Z = np.zeros((self.x_dim, self.k))
            Z = np.subtract(std_X[neighs[idx, :]], i.T).T

            C = Z.T @ Z
            e = 1e-3 * np.trace(C)
            C = C + e * np.eye(C.shape[0])

            w = la.solve(C, np.ones(self.k).T, sym_pos=True)
            w = w/np.sum(w)

            self.W[idx, neighs[idx, :]] = w

        # 3.
        IW = np.eye(self.num_x) - self.W
        M = IW.T @ IW
        A, Q = sla.eigsh(M, k=self.target_dim+1, sigma=0.0)

        return Q[:, 1:]


class KNN(Model):

    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, X, y):
        self.data = X
        self.labels = y

    def predict(self, X):
        dist = distance.cdist(self.data, X, 'euclidean')
        neighs = np.argsort(dist, axis=0)[0:self.k, :].T
        neigh_labels = self.labels[neighs]
        mode, _ = stats.mode(neigh_labels, axis=1)

        return mode
