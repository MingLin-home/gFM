"""
gFM with PCA. Also support sample_weight interface in order to do Adaboost
@author: Ming Lin
@contact: linmin@umich.edu
"""
import sklearn.decomposition
import sklearn.preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
import gFM
import numpy

class AdaptivePCAgFM(BaseEstimator, ClassifierMixin):
    def __init__(self, pca_n_components_list, gFM_rank_k_list, lambda_w_list, lambda_M_list, learning_rate=1):
        self.pca_n_components_list = pca_n_components_list
        self.gFM_rank_k_list = gFM_rank_k_list
        self.lambda_w_list = lambda_w_list
        self.lambda_M_list = lambda_M_list
        self.learning_rate = learning_rate
        return

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is None:
            sample_weight = numpy.ones((X.shape[0],))
            sample_weight = sample_weight/numpy.sum(sample_weight)
        sample_weight = sample_weight[:,numpy.newaxis]

        self.classes_ = numpy.unique(y)
        self.n_classes_ = len(self.classes_)

        X_orig = X.T
        X = X_orig
        y = numpy.asarray(y, dtype=numpy.float)

        self.d = X.shape[0]
        n = X.shape[1]

        for pca_n_component, gFM_rank_k, lambda_w,lambda_M in zip(self.pca_n_components_list,
                                                                  self.gFM_rank_k_list,
                                                                  self.lambda_w_list,self.lambda_M_list):
            X = X_orig
            # weighted z-score normalization
            X_times_sample_weight = n * X * sample_weight.T
            self.data_mean_1 = X_times_sample_weight.mean(axis=1, keepdims=True)
            X = X - self.data_mean_1
            X_weighted_std = numpy.sqrt(n * numpy.mean((X ** 2) * sample_weight.T, axis=1, keepdims=True))
            self.data_std_1 = numpy.maximum(X_weighted_std, 1e-12)
            X = X / self.data_std_1

            # weighted PCA
            X_times_sample_weight = X * numpy.sqrt(n*sample_weight.T)
            U, s, V = numpy.linalg.svd(X_times_sample_weight)
            X = U[:, 0:pca_n_component].T.dot(X)
            self.U = U[:, 0:pca_n_component]

            # weighted z-score normalization of PCA features
            X_times_sample_weight = n * X * sample_weight.T
            self.data_mean_2 = X_times_sample_weight.mean(axis=1, keepdims=True)
            X = X - self.data_mean_2
            X_weighted_std = numpy.sqrt(n * numpy.mean((X ** 2) * sample_weight.T, axis=1, keepdims=True))
            self.data_std_2 = numpy.maximum(X_weighted_std, 1e-12)
            X = X / self.data_std_2

            self.gFM_estimator = gFM.BatchSolver(rank_k=gFM_rank_k, lambda_M=lambda_M,lambda_w=lambda_w,learning_rate=self.learning_rate)

            self.gFM_estimator.fit(X.T, y.flatten(), sample_weight=sample_weight.flatten())

            y_pred = self.gFM_estimator.predict(X.T)

            the_error = numpy.sum(numpy.abs(y_pred - y)/2*sample_weight.flatten())
            if the_error<0.5: break
        # end for
        if the_error>=0.5: print 'the_error=%f, end boosting!' %(the_error)
        return self
    # end def

    def decision_function(self, X):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Compute the decision values $s$ of X such that $\sign{s}$ is the predicted labels of X
        :param X: $n \times d$.
        :return: The decision values of X, $n \times 1$ vector
        """

        X = X.T
        X = X - self.data_mean_1
        X = X/self.data_std_1
        X = self.U.T.dot(X)
        X = X - self.data_mean_2
        X = X/self.data_std_2
        y_pred = self.gFM_estimator.decision_function(X.T).flatten()
        # y_pred = ((y_pred - numpy.min(y_pred))/(numpy.max(y_pred)-numpy.min(y_pred))-0.5)*2
        return y_pred

    def predict(self, X):
        # y_pred = numpy.sign(self.decision_function(X))
        y_pred = self.decision_function(X)
        return y_pred
# end class



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

        self.classes_ = numpy.unique(y)
        self.n_classes_ = len(self.classes_)

        X = X.T
        y = y[:, numpy.newaxis]
        y = numpy.asarray(y, dtype=numpy.float)

        self.d = X.shape[0]
        n = X.shape[1]

        # weighted z-score normalization
        X_times_sample_weight = n * X * sample_weight.T
        self.data_mean_1 = X_times_sample_weight.mean(axis=1, keepdims=True)
        X = X - self.data_mean_1
        X_weighted_std = numpy.sqrt(n * numpy.mean((X ** 2) * sample_weight.T, axis=1, keepdims=True))
        self.data_std_1 = numpy.maximum(X_weighted_std, 1e-12)
        X = X / self.data_std_1

        # weighted PCA
        X_times_sample_weight = X * numpy.sqrt(n*sample_weight.T)
        U, s, V = numpy.linalg.svd(X_times_sample_weight)
        X = U[:, 0:self.n_components].T.dot(X)
        self.U = U[:, 0:self.n_components]

        # weighted z-score normalization of PCA features
        X_times_sample_weight = n * X * sample_weight.T
        self.data_mean_2 = X_times_sample_weight.mean(axis=1, keepdims=True)
        X = X - self.data_mean_2
        X_weighted_std = numpy.sqrt(n * numpy.mean((X ** 2) * sample_weight.T, axis=1, keepdims=True))
        self.data_std_2 = numpy.maximum(X_weighted_std, 1e-12)
        X = X / self.data_std_2

        self.gFM_estimator.fit(X.T, y.flatten(), sample_weight=sample_weight.flatten())

    pass

    def decision_function(self, X):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Compute the decision values $s$ of X such that $\sign{s}$ is the predicted labels of X
        :param X: $n \times d$.
        :return: The decision values of X, $n \times 1$ vector
        """

        X = X.T
        X = X - self.data_mean_1
        X = X/self.data_std_1
        X = self.U.T.dot(X)
        X = X - self.data_mean_2
        X = X/self.data_std_2
        y_pred = self.gFM_estimator.decision_function(X.T).flatten()
        # y_pred = ((y_pred - numpy.min(y_pred))/(numpy.max(y_pred)-numpy.min(y_pred))-0.5)*2
        return y_pred

    def predict(self, X):
        y_pred = numpy.sign(self.decision_function(X))
        # y_pred = self.decision_function(X)
        return y_pred
# end class