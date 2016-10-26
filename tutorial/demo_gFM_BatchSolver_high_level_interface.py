'''
This script demonstrates the usage of high-level interface of gFM package.
@author: Ming Lin
@contact: linmin@umich.edu
'''

import numpy

import sys
sys.path.append('../')
import gFM # import the gFM package


#---------- Generate Skewed Gaussian distribution as training data ----------#
d = 100 # feature dimension
k = 3 # the rank of the target matrix in gFM
n = 30*k*d # the number of training instances
n_testset = 1000 # the number of testing instances

X = numpy.random.randn(d,n)
X[X>0.5] = 0

X_test = numpy.random.randn(d,n_testset)
X_test[X_test>0.5] = 0


# It is important to ensure X zero-mean and unit-variance when generating the labels
X_normalized = (X-X.mean(axis=1,keepdims=True))/X.std(axis=1,keepdims=True)
X_test_normalized = (X_test-X.mean(axis=1,keepdims=True))/X.std(axis=1,keepdims=True)

#---------- Synthetic $M^*$ and $w^*$ as our ground-truth gFM model
U_true = numpy.random.randn(d,k)
U_true_unit,_ = numpy.linalg.qr(U_true)
M_true = numpy.dot(U_true,U_true.T)
M_true = M_true/numpy.linalg.norm(M_true)
w_true = numpy.random.randn(d,1)
w_true = w_true/numpy.linalg.norm(w_true)

y = X_normalized.T.dot(w_true) +  gFM.A_(X_normalized,U_true,U_true) # generate true labels for training
y_test = X_test_normalized.T.dot(w_true) +  gFM.A_(X_test_normalized,U_true,U_true) # generate true labels for testing



#---------- Initialize gFM ----------#
print 'Initializing gFM minibatch ...'

# Create a new gFM solver. The solver is of rank rank_k. We choose the regularizer lambda_M and lambda_w to be the twice of the norm of ground-truth.
# Iterate 200 steps in the initialization stage and iterate 200 steps in the training stage. The learning rate by default is 1. We can it to be 0.01 for example but don't forget to increase the number of iterations too!
# Only rank_k must be specified here.
my_gFM_solver = gFM.BatchSolver(rank_k=k,
                                max_iter=100, max_init_iter=20,
                                lambda_M=numpy.linalg.norm(U_true) ** 2 * 2,
                                lambda_w=numpy.linalg.norm(w_true) * 2, 
								learning_rate = 1)

# Train gFM
print 'Traing gFM ...'
my_gFM_solver.fit(X.T,y.flatten())
print 'Done'

the_error = numpy.linalg.norm( my_gFM_solver.U.dot(my_gFM_solver.V.T) - U_true.dot(U_true.T),2) \
                + numpy.linalg.norm(my_gFM_solver.w - w_true)

predicted_y = my_gFM_solver.decision_function(X_test.T)
prediction_error = numpy.mean(numpy.abs(predicted_y - y_test.flatten()))

print 'estimation error = [%4g], prediction error=[%4g] ' % (the_error,prediction_error)