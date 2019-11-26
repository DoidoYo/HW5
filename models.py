import numpy as np
from collections import defaultdict

# TODO: You can import anything from numpy or scipy here!

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
        # TODO: Implement!
        raise NotImplementedError()

class LLE(Model):

    def __init__(self, X, target_dim, lle_k):
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]

        self.target_dim = target_dim
        self.k = lle_k

    def fit(self, X):
        # TODO: Implement!
        raise NotImplementedError()

class KNN(Model):

    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, X, y):
        self.data = X
        self.labels = y

    def predict(self, X):
        # TODO: Implement!
