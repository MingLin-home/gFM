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


#---------- Generate Skewed Gaussian distribution as training data ----------#
d = 100 # feature dimension
k = 3 # the rank of the target matrix in gFM
n = 20*k*d # the number of training instances


X = numpy.random.randn(d, n)
X[X>0.5] = 0

# It is important to ensure X zero-mean and unit-variance when synthetizing the labels
X_normalized = (X-X.mean(axis=1,keepdims=True))/X.std(axis=1,keepdims=True)

#---------- Synthetic $M^*$ and $w^*$ as our ground-truth gFM model

U_true = numpy.random.randn(d,k)
M_true = numpy.dot(U_true,U_true.T)
M_true /= numpy.linalg.norm(M_true)
U,s,V = numpy.linalg.svd(M_true)
V = s[:,numpy.newaxis] * V
U_true = U
V_true = V.T
w_true = numpy.random.randn(d,1)
w_true /= numpy.linalg.norm(w_true)

y = X_normalized.T.dot(w_true) +  gFM.A_(X_normalized,U_true,V_true) # synthetize true labels

#---------- Initialize gFM ----------#
print 'Initializing gFM minibatch ...'

# Create a new gFM model. The model is of rank rank_k. We choose the regularizer lambda_M and lambda_w to be the twice of the norm of ground-truth.
# Only rank_k must be specified here.
my_gFM_solver = gFM.BatchSolver(rank_k=k,
                                lambda_M=numpy.linalg.norm(U_true) ** 2 * 2,
                                lambda_w=numpy.linalg.norm(w_true) * 2, )

# Initialization stage of the gFM. Iterate 10 loops in the initialization stage.
my_gFM_solver.initialization(X,y,max_init_iter=200)

T = 200 # The number of iterations in the training stage
err_record = numpy.zeros((20,)) # record the estimation error along iteration
error_record_t = numpy.round(numpy.linspace(0,T,len(err_record))) # we will record the error at step error_record_t[0], error_record_t[1], etc.
err_record_count = 0
for t in xrange(T):
    # invoke one training iteration. For maximum efficient, it is better to first z-score normalized X along feature dimension then pass the parameter
    # z_score_normalized = True when calling iterate_train()
    my_gFM_solver.iterate_train(X, y,max_iter=1)

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

plt.plot(error_record_t,err_record,'b-x',label='gFM BatchSolver')
plt.xlabel('step t')
plt.ylabel('estimation error')
plt.legend()
plt.show()