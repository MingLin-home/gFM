'''
Generate synthetic dataset and use batch regression to train gFM. Show the learning curves along iteration.

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


repeat_times = 1 # repeat experiments
dim = 100 # the dimension of features
rank_k = 3 # the rank of gFM
total_iteration = 60 # number of iterations to train gFM
total_record = 20 # number of check points
max_init_iter = 20 # number of iterations to initialize gFM

# We save the recovery error, training set prection error and testing set prediction error along iteration
recovery_error_record = numpy.zeros((repeat_times, total_record))
trainset_error_record = numpy.zeros((repeat_times, total_record))
testset_error_record = numpy.zeros((repeat_times, total_record))
# The count of iteration at each check point
record_iteration_axis = numpy.round(numpy.linspace(0, total_iteration, total_record, endpoint=True)).astype('int')


def train(export_filename):
    """
    Train gFM and save learning curve in file named by export_filename
    :param export_filename: the namve of the file saving the learning curve
    :return:
    """
    for repeat_count in xrange(repeat_times):
        print 'generating training toy set'
        n_trainset = 100 * rank_k * dim # size of training set
        n_testset = 10000 # size of testing set

        X = numpy.random.randn(n_trainset, dim) # training set instances
        X_test = numpy.random.randn(n_testset, dim) # testing set instances

        # ---------- Synthetic $M^*$ and $w^*$ as our ground-truth gFM model
        U_true = numpy.random.randn(dim, rank_k) / numpy.sqrt(dim)
        w_true = numpy.random.randn(dim, 1) / numpy.sqrt(dim)
        M_true = U_true.dot(U_true.T)

        # generate true labels for training
        y = X.dot(w_true) + gFM.A_(X.T, U_true, U_true) + 0.5
        y = y.flatten()

        # generate true labels for testing
        y_test = X_test.dot(w_true) + gFM.A_(X_test.T, U_true, U_true) + 0.5
        y_test = y_test.flatten()

        # Initialize gFM BatchRegression. We set max_init_iter=20 steps in the initialization with accuracy init_tol=1e-3.
        # In the training stage, we require to iterate as long as we want, so we set the training accuracy tol=0.
        # The learning rate is set to 1.
        # We can set the learning_rate to a small number e.g. learning_rate=0.1 when the data is not well conditioned.
        the_estimator = gFM.BatchRegression(rank_k=rank_k, max_init_iter=20, learning_rate=1, init_tol=1e-3, tol=0)

        # Initialize gFM with no iteration. This will assign memory space for U,V,w without iteration.
        the_estimator.fit(X, y, n_more_iter=0)
        record_count = 0
        n_more_iter = 0
        y_trainset_pred = the_estimator.decision_function(X)
        y_testset_pred = the_estimator.decision_function(X_test)
        trainset_error = sklearn.metrics.mean_absolute_error(y, y_trainset_pred) / sklearn.metrics.mean_absolute_error(y, numpy.zeros((len(y),)))
        testset_error = sklearn.metrics.mean_absolute_error(y_test, y_testset_pred) / sklearn.metrics.mean_absolute_error(y_test, numpy.zeros((len(y_test),)))
        trainset_error_record[repeat_count, record_count] = trainset_error
        testset_error_record[repeat_count, record_count] = testset_error
        the_recovery_error = (numpy.linalg.norm(the_estimator.w.flatten() - w_true.flatten()) + \
                              numpy.linalg.norm(the_estimator.U.dot(the_estimator.V.T) - M_true, 2)) / (numpy.linalg.norm(w_true.flatten()) + numpy.linalg.norm(M_true, 2))
        recovery_error_record[repeat_count, record_count] = the_recovery_error
        print '[ite=%d(+%d), trainset predict error= %g, testset predict error=%g, the recovery error=%g] ' % \
              (record_count, n_more_iter, trainset_error, testset_error, the_recovery_error)

        # start iteration
        for record_count in xrange(1, total_record):
            n_more_iter = record_iteration_axis[record_count] - record_iteration_axis[record_count - 1]
            # In each fit() call, we limite the number of iteration to be n_more_iter=n_more_iter.
            the_estimator.fit(X, y, n_more_iter=n_more_iter)
            y_trainset_pred = the_estimator.predict(X)
            y_testset_pred = the_estimator.predict(X_test)
            trainset_error = sklearn.metrics.mean_absolute_error(y, y_trainset_pred) / sklearn.metrics.mean_absolute_error(y, numpy.zeros((len(y),)))
            testset_error = sklearn.metrics.mean_absolute_error(y_test, y_testset_pred) / sklearn.metrics.mean_absolute_error(y_test, numpy.zeros((len(y_test),)))
            trainset_error_record[repeat_count, record_count] = trainset_error
            testset_error_record[repeat_count, record_count] = testset_error
            the_recovery_error = (numpy.linalg.norm(the_estimator.w.flatten() - w_true.flatten()) + \
                                  numpy.linalg.norm(the_estimator.U.dot(the_estimator.V.T) - M_true, 2)) / (numpy.linalg.norm(w_true.flatten()) + numpy.linalg.norm(M_true, 2))
            recovery_error_record[repeat_count, record_count] = the_recovery_error
            print '[ite=%d(+%d), trainset predict error= %g, testset predict error=%g, the recovery error=%g] ' % \
                  (record_count, n_more_iter, trainset_error, testset_error, the_recovery_error)
        # end for record_count
    # end for repeat

    # save learning curve to file named export_filename.
    numpy.savez_compressed(export_filename, record_iteration_axis=record_iteration_axis,
                           trainset_error_record=trainset_error_record, testset_error_record=testset_error_record, recovery_error_record=recovery_error_record)
# end def


if __name__ == '__main__':
    export_filename = './convergence_curve.npz'
    if not os.path.isfile(export_filename):
        train(export_filename)

    the_results = numpy.load(export_filename)
    record_iteration_axis = the_results['record_iteration_axis']

    trainset_error_record = the_results['trainset_error_record']
    testset_error_record = the_results['testset_error_record']
    recovery_error_record = the_results['recovery_error_record']

    # plot curves
    export_figname = './prediction_error_curve.png'
    plt.semilogy(record_iteration_axis, numpy.mean(trainset_error_record, axis=0), '-xb', label='train error')
    plt.semilogy(record_iteration_axis, numpy.mean(testset_error_record, axis=0), '-xr', label='test error')
    plt.legend()
    # plt.show()
    plt.savefig(export_figname)
    plt.close()

    export_figname = './recovery_error_curve.png'
    plt.semilogy(record_iteration_axis, numpy.mean(recovery_error_record, axis=0), '-xr', label='recovery error')
    plt.legend()
    # plt.show()
    plt.savefig(export_figname)
    plt.close()