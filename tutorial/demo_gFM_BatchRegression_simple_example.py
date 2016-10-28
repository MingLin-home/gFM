'''
Generate synthetic dataset and use batch regression to train gFM. Do training in one fit() call.

@author: Ming Lin
@contact: linmin@umich.edu
'''

import numpy
import os
import sys
import matplotlib.pyplot as plt
import scipy.sparse
import sklearn.metrics
import distutils.dir_util

sys.path.append('../')
import gFM


dim = 100 # the dimension of features
rank_k = 3 # the rank of gFM
total_iteration = 20 # number of iterations to train gFM
max_init_iter = 10 # number of iterations to initialize gFM

print 'generating training toy set'
n_trainset = 30 * rank_k * dim # size of training set
n_testset = 10000 # size of testing set

X = numpy.random.randn(n_trainset, dim) # training set instances
X_test = numpy.random.randn(n_testset, dim) # testing set instances

# ---------- Synthetic $M^*$ and $w^*$ as our ground-truth gFM model
U_true = numpy.random.randn(dim, rank_k) / numpy.sqrt(dim)
w_true = numpy.random.randn(dim, 1) / numpy.sqrt(dim)
M_true = U_true.dot(U_true.T)

# generate true labels for training
y = X.dot(w_true) + gFM.A_(X.T, U_true, U_true)
y = y.flatten()

# generate true labels for testing
y_test = X_test.dot(w_true) + gFM.A_(X_test.T, U_true, U_true)
y_test = y_test.flatten()

# Initialize gFM BatchRegression. We set max_init_iter=max_init_iter steps in the initialization with accuracy init_tol=1e-3.
# Similary the number of training iteration is set by max_iter=total_iteration with accuracy tol=1e-6.
# The learning rate is set to 1.
# We can set the learning_rate to a small number e.g. learning_rate=0.1 when the data is not well conditioned.
# lambda_M=1000,lambda_w=1000 constrain the norm of M and w during the regression. Since we have sufficient training data, we actually don't need them here.
the_estimator = gFM.BatchRegression(rank_k=rank_k, max_init_iter=max_init_iter, max_iter=total_iteration,
                                    learning_rate=1, init_tol=1e-3, tol=1e-6,
                                    lambda_M=1000,lambda_w=1000)

# Initialize gFM with no iteration. This will assign memory space for U,V,w without iteration.
the_estimator.fit(X, y)
y_trainset_pred = the_estimator.decision_function(X)
y_testset_pred = the_estimator.decision_function(X_test)
trainset_error = sklearn.metrics.mean_absolute_error(y, y_trainset_pred) / sklearn.metrics.mean_absolute_error(y, numpy.zeros((len(y),)))
testset_error = sklearn.metrics.mean_absolute_error(y_test, y_testset_pred) / sklearn.metrics.mean_absolute_error(y_test, numpy.zeros((len(y_test),)))
the_recovery_error = (numpy.linalg.norm(the_estimator.w.flatten() - w_true.flatten()) + \
                      numpy.linalg.norm(the_estimator.U.dot(the_estimator.V.T) - M_true, 2)) / (numpy.linalg.norm(w_true.flatten()) + numpy.linalg.norm(M_true, 2))
print '[trainset predict error= %g, testset predict error=%g, the recovery error=%g] ' % \
      (trainset_error, testset_error, the_recovery_error)