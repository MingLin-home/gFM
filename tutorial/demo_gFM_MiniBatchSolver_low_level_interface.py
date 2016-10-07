'''
This script demonstrates the usage of low-level interface of gFM package.
@author: Ming Lin
@contact: linmin@umich.edu
'''

import numpy
import matplotlib.pyplot as plt

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

# It is important to ensure X zero-mean and unit-variance when synthetizing the labels
X_normalized = (X-X.mean(axis=1,keepdims=True))/X.std(axis=1,keepdims=True)


#---------- Synthetic $M^*$ and $w^*$ as our ground-truth gFM model
U_true = numpy.random.randn(d,k)
U_true_unit,_ = numpy.linalg.qr(U_true)
M_true = numpy.dot(U_true,U_true.T)
M_true = M_true/numpy.linalg.norm(M_true)
w_true = numpy.random.randn(d,1)
w_true = w_true/numpy.linalg.norm(w_true)

y = X_normalized.T.dot(w_true) +  gFM.A_(X_normalized,U_true,U_true) # synthetize true labels

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
# Only rank_k must be specified here.
my_gFM_solver = gFM.MiniBatchSolver(k, data_mean, data_std, data_moment3, data_moment4,
                                    load_data_func = load_data_func, load_data_func_para = load_data_func_para,
                                    lambda_M=numpy.linalg.norm(U_true) ** 2 * 2,
                                    lambd_w=numpy.linalg.norm(w_true) * 2,
                                    )

# Initialization stage of the gFM. Iterate 10 loops in the initialization stage.
my_gFM_solver.initialization(max_init_iter=10)
print 'gFM minibatch initialized'

T = 20 # The number of iterations in the training stage
err_record = numpy.zeros((20,)) # record the estimation error along iteration
error_record_t = numpy.round(numpy.linspace(0,T,len(err_record))) # we will record the error at step error_record_t[0], error_record_t[1], etc.
err_record_count = 0
for t in xrange(T):
    # invoke one training iteration.
    my_gFM_solver.iterate_train(max_iter=1)
    # record the estimation error at step t
    if t == error_record_t[err_record_count]:
        the_error = numpy.linalg.norm( my_gFM_solver.U.dot(my_gFM_solver.V.T) - U_true.dot(U_true.T),2) \
                    + numpy.linalg.norm(my_gFM_solver.w - w_true)
        err_record[err_record_count] = the_error
        err_record_count+=1
        print '[%4g] ' % (the_error),
    else:
        print '.',
    # end if
# end for

plt.plot(error_record_t,err_record,'b-x',label='gFM Mini-Batch Solver')
plt.xlabel('step t')
plt.ylabel('estimation error')
plt.legend()
plt.show()