'''
This script demonstrates the usage of low-level interface of gFM package.
@author: Ming Lin
@contact: linmin@umich.edu
'''

import numpy

import sys
sys.path.append('../')
import gFM # import the gFM package

# The mini-batch solver needs to know how to load dataset streamly.
def load_data_func(load_data_func_para):
    X = load_data_func_para['X']
    y = load_data_func_para['y']
    last_n = load_data_func_para['n']
    minibatch_size = 10000

    if last_n+minibatch_size <= X.shape[1]:
        return_X = X[:,last_n:last_n+minibatch_size]
        return_y = y[last_n:last_n+minibatch_size,0]
        return_y = return_y.reshape(len(return_y),1)
        load_data_func_para['n'] = last_n+minibatch_size
        return (return_X,return_y,load_data_func_para)
    elif last_n < X.shape[1]:

        return_X = X[:,last_n:]
        return_y = y[last_n:,0]
        return_y = return_y.reshape(len(return_y),1)
        load_data_func_para['n'] = X.shape[1]
        return (return_X,return_y,load_data_func_para)
    else:
        load_data_func_para['n'] = 0
        return (None,None,load_data_func_para)


#---------- Generate Skewed Gaussian distribution as training data ----------#
d = 100 # feature dimension
k = 3 # the rank of the target matrix in gFM
n = 100*k*d # the number of training instances
n_testset = 1000 # the number of testing instances

X = numpy.random.randn(d, n)
X[X>0.5] = 0

X_test = numpy.random.randn(d,n_testset)
X_test[X_test>0.5] = 0

# It is important to ensure X zero-mean and unit-variance when synthetizing the labels
X_normalized = (X-X.mean(axis=1,keepdims=True))/X.std(axis=1,keepdims=True)
X_test_normalized = (X_test-X.mean(axis=1,keepdims=True))/X.std(axis=1,keepdims=True)

#---------- Synthetic $M^*$ and $w^*$ as our ground-truth gFM model
U_true = numpy.random.randn(d,k)
U_true_unit,_ = numpy.linalg.qr(U_true)
M_true = numpy.dot(U_true,U_true.T)
M_true = M_true/numpy.linalg.norm(M_true)
w_true = numpy.random.randn(d,1)
w_true = w_true/numpy.linalg.norm(w_true)

y = X_normalized.T.dot(w_true) +  gFM.A_(X_normalized,U_true,U_true) # synthetize true labels
y_test = X_test_normalized.T.dot(w_true) +  gFM.A_(X_test_normalized,U_true,U_true) # synthetize true labels

#---------- Initialize gFM ----------#
print 'Initializing gFM minibatch ...'

# First we need to know the moments of data
data_mean = numpy.mean(X,axis=1,keepdims=True)
data_std = numpy.std(X,axis=1,keepdims=True)
data_moment3 = numpy.mean(X_normalized**3,axis=1,keepdims=True)
data_moment4 = numpy.mean(X_normalized**4,axis=1,keepdims=True)

# The parameters passed to the mini-batch data loading function
load_data_func_para={'X':X,'y':y,'n':0}

# Create a new gFM model. The model is of rank rank_k. We choose the regularizer lambda_M and lambda_w to be the twice of the norm of ground-truth.
# Iterate 10 steps in the initialization stage and iterate 20 steps in the training stage.
# Only rank_k must be specified here.
my_gFM_solver = gFM.MiniBatchSolver(k, data_mean, data_std, data_moment3, data_moment4,
                                    max_iter=20, max_init_iter=10,
                                    load_data_func = load_data_func, load_data_func_para = load_data_func_para,
                                    lambda_M=numpy.linalg.norm(U_true) ** 2 * 2,
                                    lambd_w=numpy.linalg.norm(w_true) * 2,
                                    )

# Train gFM
print 'Traing gFM mini-batch...'
my_gFM_solver.fit()
print 'Done'

the_error = numpy.linalg.norm( my_gFM_solver.U.dot(my_gFM_solver.V.T) - U_true.dot(U_true.T),2) \
                + numpy.linalg.norm(my_gFM_solver.w - w_true)

predicted_y = my_gFM_solver.decision_function(X_test)
prediction_error = numpy.linalg.norm(predicted_y - y_test)

print 'estimation error = [%4g], prediction error=[%4g] ' % (the_error,prediction_error)