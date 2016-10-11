"""
gFM with PCA. Also support sample_weight interface in order to do Adaboost
@author: Ming Lin
@contact: linmin@umich.edu
"""
import sklearn.decomposition
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy


class PCAgFM(BaseEstimator, ClassifierMixin):
    def __init__(self, gFM_estimator, n_components):
        self.gFM_estimator = gFM_estimator
        self.n_components = n_components
        return

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is None:
            sample_weight = numpy.ones((X.shape[0],))
            sample_weight = sample_weight/numpy.sum(sample_weight)
        sample_weight = sample_weight[:,numpy.newaxis]

        X = X.T
        y = y[:, numpy.newaxis]
        y = numpy.asarray(y, dtype=numpy.float)

        self.d = X.shape[0]
        n = X.shape[1]

        # weighted z-score normalization
        X_times_sample_weight = X * sample_weight
        self.data_mean_1 = X_times_sample_weight.mean(axis=1, keepdims=True)
        X = X - self.data_mean_1
        X_weighted_std = numpy.mean((X ** 2) * sample_weight, axis=1, keepdims=True)
        self.data_std_1 = numpy.maximum(X_weighted_std, 1e-12)
        X = X / self.data_std

        # weighted PCA
        X_times_sample_weight = X * sample_weight
        U, s, V = numpy.linalg.svd(X_times_sample_weight)
        X = U[:, 0:self.n_components].T.dot(X_times_sample_weight)
        self.U = U[:, 0:self.n_components]

        # weighted z-score normalization of PCA features
        X_times_sample_weight = X * sample_weight
        self.data_mean_2 = X_times_sample_weight.mean(axis=1, keepdims=True)
        X = X - self.data_mean_2
        X_weighted_std = numpy.mean((X ** 2) * sample_weight, axis=1, keepdims=True)
        self.data_std_2 = numpy.maximum(X_weighted_std, 1e-12)
        X = X / self.data_std_2

        self.gFM_estimator.fit(X.T, y.flatten(), sample_weight=sample_weight)

    pass

    def predict(self, X):
        X = X.T
        X = X - self.data_mean_1
        X = X/self.data_std_1
        X = self.U.T.dot(X)
        X = X - self.data_mean_2
        X = X/self.data_std_2
        return self.gFM_estimator.predict(X)
# end class