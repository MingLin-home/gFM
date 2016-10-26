"""
The gFM toolbox provides efficient solvers for the Generalized Factorization Machine (gFM) that can handle Tera-byte datasets.

There are two solvers provided:
1) gFM_BatchSolver implements the batch updating where the whole dataset can be loaded into memory.
2) gFM_MiniBatchSolver implements the mini-batch version of gFM_BatchSolver where we can load dataset in a mini-batch style. (To be implemented)

For installation and usage information, please refer to README.txt and examples.

@author: Ming Lin
@contact: linmin@umich.edu
"""
import sklearn.decomposition
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy


class BatchSolver(BaseEstimator, ClassifierMixin):
    """
    The batch solver for gFM when the whole dataset can be loaded in memory.
    """
    def __init__(self, rank_k=None, max_iter=None, max_init_iter=None, lambda_M=numpy.Inf, lambda_w=numpy.Inf, learning_rate=1.0,):
        """
        Initialize a gFM_BatchSolver instance.
        :param rank_k: The rank of the target second order matrix in gFM ($M^*$). Should be of type int.
        :param max_iter: The number of iterations for training.
        :param max_init_iter: The number of iterations in initialization step.
        :param lambda_M: The Frobenius norm constraint for M
        :param lambda_w: The $\ell_2$-norm constraint for w
        """
        learning_rate = float(learning_rate)
        self.rank_k = rank_k
        self.lambda_M = lambda_M
        self.lambda_w = lambda_w
        self.max_iter = max_iter
        if self.max_iter is None: self.max_iter = int(100/learning_rate)
        self.max_init_iter = max_init_iter
        if self.max_init_iter is None: self.max_init_iter = int(100/learning_rate)
        self.learning_rate = learning_rate
        return

    def fit(self,X,y=None, sample_weight=None):
        # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> object
        """
        Train gFM with data X and label y.
        :param X: Feature matrix, $n \times d$.
        :param y: Label vector, shape=(n,)
        :return: self
        """
        self.data_mean = None
        self.data_std = None
        self.data_moment3 = None
        self.data_moment4 = None
        self.data_moment5 = None

        if sample_weight is None:
            sample_weight = numpy.ones((X.shape[0],))
            sample_weight = sample_weight/numpy.sum(sample_weight)

        self.Z = None
        self.G = None
        self.feature_selection_bool = None
        self.moment_threshold = 0.05  # don't use features whose Z,G is too singular

        self.initialization(X, y, sample_weight=sample_weight, max_init_iter=self.max_init_iter)
        self.iterate_train(X, y, sample_weight=sample_weight, max_iter=self.max_iter)
        return self

    def decision_function(self,X):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Compute the decision values $s$ of X such that $\sign{s}$ is the predicted labels of X
        :param X: $n \times d$.
        :return: The decision values of X, $n \times 1$ vector
        """
        X = X.T
        X = (X - self.data_mean) / self.data_std
        the_decision_values = A_(X,self.U,self.V) + X.T.dot(self.w)
        return the_decision_values.flatten()

    def predict(self,X):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Predict the labels of X
        :param X: $n \times d$ feature matrix.
        :return: The predicted labels
        """
        return numpy.sign(self.decision_function(X)).flatten()

    def save_model(self,file):
        # type: (str) -> object
        """
        Save gFM model.
        :param file: File-like object or string. Save model to the file.
        :return: self
        """
        numpy.savez_compressed(file, U = self.U, V = self.V, w = self.w,
                               data_mean = self.data_mean, data_std = self.data_std,
                               data_moment3 = self.data_moment3, data_moment4 = self.data_moment4, data_moment5=self.data_moment5,
                               Z = self.Z, G=self.G,
                               feature_selection_bool = self.feature_selection_bool, moment_threshold = self.moment_threshold,
                               max_iter = self.max_iter, max_init_iter = self.max_init_iter, learning_rate=self.learning_rate,
                               lambda_M=self.lambda_M, lambda_w=self.lambda_w, rank_k=self.rank_k)

        return self

    def load_model(self,file):
        """
        Load gFM model from file.
        :param file: File-like object or string. Load model from the file
        :return: self
        """
        the_loaded_file = numpy.load(file)
        self.U = the_loaded_file['U']
        self.V = the_loaded_file['V']
        self.w = the_loaded_file['w']
        self.data_mean = the_loaded_file['data_mean_step1']
        self.data_std = the_loaded_file['data_std']
        self.data_moment3 = the_loaded_file['data_moment3']
        self.data_moment4 = the_loaded_file['data_moment4']
        self.data_moment5 = the_loaded_file['data_moment5']
        self.Z = the_loaded_file['Z']
        self.G = the_loaded_file['G']
        self.feature_selection_bool = the_loaded_file['feature_selection_bool']
        self.moment_threshold = the_loaded_file['moment_threshold']
        self.max_iter = the_loaded_file['max_iter']
        self.max_init_iter = the_loaded_file['max_init_iter']
        self.learning_rate = the_loaded_file['learning_rate']
        self.lambda_M = the_loaded_file['lambda_M']
        self.lambda_w = the_loaded_file['lambda_w']
        self.rank_k = the_loaded_file['rank_k']



        return self
    # end def


    def initialization(self, X, y, sample_weight=None, max_init_iter=10):
        # type: (numpy.ndarray, numpy.ndarray, int) -> numpy.ndarray
        """
        Use trancated SVD to initialize U0,V0. Batch updating.
        :param X: feature matrix, $n\times d$
        :param y: label vector, shape=(n,)
        :param max_init_iter: the number of iterations for initialization. max_iter=10 is usually good enough
        :return: None
        """
        X = X.T
        y =y[:,numpy.newaxis]
        y = numpy.asarray(y,dtype=numpy.float)

        self.d = X.shape[0]
        n = X.shape[1]

        # z-score normalization considering sample weight
        if sample_weight is None:
            sample_weight = numpy.ones((n,))
            sample_weight = sample_weight/numpy.sum(sample_weight)
        sample_weight = sample_weight[:,numpy.newaxis]

        X_times_sample_weight = n * X*sample_weight.T
        self.data_mean = X_times_sample_weight.mean(axis=1,keepdims=True)
        X = X - self.data_mean
        X_weighted_std = numpy.sqrt(n * numpy.mean((X**2)*sample_weight.T,axis=1,keepdims=True))
        self.data_std = numpy.maximum(X_weighted_std,1e-12)
        X = X/self.data_std
        self.data_moment3 = numpy.mean(n*(X**3)*sample_weight.T,axis=1,keepdims=True)
        self.data_moment4 = numpy.mean(n*(X**4)*sample_weight.T,axis=1,keepdims=True)
        self.data_moment5 = numpy.mean(n*(X**5)*sample_weight.T,axis=1,keepdims=True)

        tmp_A = numpy.zeros((2,3,self.d))
        tmp_A[0,0,:] = 1
        tmp_A[0,1,:] = self.data_moment3.ravel()
        tmp_A[0,2,:] = self.data_moment4.ravel()
        tmp_A[1,0,:] = self.data_moment3.ravel()
        tmp_A[1,1,:] = self.data_moment4.ravel()-1
        tmp_A[1,2,:] = self.data_moment5.ravel()-self.data_moment3.ravel()

        # tmp_A = numpy.zeros((2, 2, self.d))
        # tmp_A[0, 0, :] = 1
        # tmp_A[0, 1, :] = self.data_moment3.ravel()
        # tmp_A[1, 0, :] = self.data_moment3.ravel()
        # tmp_A[1, 1, :] = self.data_moment4.ravel() - 1


        tmp_b = numpy.zeros((2,1,self.d))
        tmp_b[0,0,:] = self.data_moment3.ravel()
        tmp_b[1,0,:] = self.data_moment4.ravel()-3

        tmp_bw = numpy.zeros((2,1,self.d))
        tmp_bw[0,0,:] = 1
        tmp_bw[1,0,:] = 0

        self.Z = numpy.zeros((self.d,3))
        self.G = numpy.zeros((self.d, 3))
        # self.Z = numpy.zeros((self.d, 2))
        # self.G = numpy.zeros((self.d, 2))
        sv_record = numpy.zeros((self.d,))
        for i in xrange(self.d):
            tmpu_u,tmp_s,tmp_v = numpy.linalg.svd(tmp_A[:, :, i],full_matrices=False)
            sv_record[i] = tmp_s[1]
            if tmp_s[1]<0.05:
                print 'warning! small singular value when computing Z and G! sv = %f' %(tmp_s[1])
            pinv_tmpA = numpy.linalg.pinv(tmp_A[:,:,i],0.05)
            self.G[i,:] = numpy.ravel(pinv_tmpA.dot(tmp_bw[:,:,i]))
            self.Z[i,:] = numpy.ravel(pinv_tmpA.dot(tmp_b[:,:,i]))


        U,_ = numpy.linalg.qr( numpy.random.randn(self.d,self.rank_k))
        for t in xrange(max_init_iter):
            U = mathcal_M_(n*y*sample_weight,U,X,self.data_moment3, self.Z)/(2*n)
            U,_ = numpy.linalg.qr(U)
        # end for t

        # V = numpy.zeros((self.d, self.rank_k))

        # update V
        V = mathcal_M_(n*y*sample_weight,U, X, self.data_moment3, self.Z)/(2*n)*self.learning_rate
        if numpy.linalg.norm(V) > self.lambda_M: V = V / numpy.linalg.norm(V) * self.lambda_M


        # update w
        hat_y = A_(X, U, V)
        dy = y - hat_y
        dy = n*dy*sample_weight
        w = mathcal_W_(dy, X, self.data_moment3, self.G)/n*self.learning_rate
        if numpy.linalg.norm(w) > self.lambda_w: w_new = w / numpy.linalg.norm(w) * self.lambda_w

        self.U = U
        self.V = V
        self.w = w

        return self
    # end def



    def iterate_train(self, X, y, sample_weight=None, max_iter=1, z_score_normalized=False):
        # type: (numpy.ndaray, numpy.ndaray, int) -> numpy.ndaray
        """
        Update U,V,w using batch iteration.
        :param X: feature matrix, $n \times d$ matrix
        :param y: label vector, shape=(n,)
        :param max_iter: number of iterations
        :param z_score_normalized: If True, it means that the dataset X has been z-score normalized already. If not (default), the solver will z-score normalized it.
        """
        U = self.U
        V = self.V
        w = self.w
        X = X.T
        n = X.shape[1]
        y = numpy.asarray(y,dtype=numpy.float)
        y = y[:,numpy.newaxis]

        if sample_weight is None:
            sample_weight = numpy.ones((n,))
            sample_weight = sample_weight / numpy.sum(sample_weight)
        sample_weight = sample_weight[:, numpy.newaxis]

        if z_score_normalized == False:
            X = (X-self.data_mean)/self.data_std

        for t in xrange(max_iter):
            hat_y = A_(X,U,V) + X.T.dot(w)
            dy = y-hat_y
            dy = n*dy*sample_weight

            # update U
            U_new = mathcal_M_(dy,U,X, self.data_moment3,self.Z)/(2*n)*self.learning_rate + \
                U.dot(V.T.dot(U))/2 + V.dot(U.T.dot(U))/2
            # V_new = U_new
            U_new,_ = numpy.linalg.qr(U_new)

            # update V
            V_new = mathcal_M_(dy,U_new,X, self.data_moment3, self.G)/(2*n)*self.learning_rate + \
                    U.dot(V.T.dot(U_new))/2 + V.dot(U.T.dot(U_new))/2

            # update w
            hat_y = A_(X,U_new,V_new) + X.T.dot(w)
            dy = y-hat_y
            dy = n*dy*sample_weight
            w_new = mathcal_W_(dy,X, self.data_moment3, self.G)/n*self.learning_rate + w


            if numpy.linalg.norm(V_new) > self.lambda_M: V_new = V_new / numpy.linalg.norm(V_new) * self.lambda_M
            if numpy.linalg.norm(w_new) > self.lambda_w: w_new = w_new / numpy.linalg.norm(w_new) * self.lambda_w
            # update old with new variances
            U = U_new
            V = V_new
            w = w_new
        # end for t

        self.U = U
        self.V = V
        self.w = w
        return self
# end class




def mathcal_W_(y,X, data_moment3, G):
    # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Return $\mathcal{W}(y)*n given the constant parameters. X should be zero-mean unit-variance
    """
    p0 = numpy.sum(y)
    p1 = X.dot(y)
    p2 = (X**2).dot(y)
    p3 = (X**3).dot(y)

    return G[:,0,numpy.newaxis]*p1 + G[:,1,numpy.newaxis]*(p2-p0) + G[:,2,numpy.newaxis]*( p3 - data_moment3*p0 )
    # return G[:, 0, numpy.newaxis] * p1 + G[:, 1, numpy.newaxis] * (p2 - p0)

# end def

def mathcal_M_(y,U,X,data_moment3, Z):
    # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Return $\mathcal{M}(y)U*2n, given the constant parameters. X should be zero-mean unit-variance
    """

    p0 = numpy.sum(y)
    p1 = X.dot(y)
    p2 = (X**2).dot(y)
    p3 = (X**3).dot(y)

    term1 = (X*y.T).dot(X.T.dot(U))
    term2 = p0+ Z[:,0,numpy.newaxis]*p1 + Z[:,1,numpy.newaxis]*(p2-p0) + Z[:,2,numpy.newaxis]*( p3 - data_moment3*p0)
    # term2 = p0+ Z[:, 0, numpy.newaxis] * p1 + Z[:, 1, numpy.newaxis] * (p2 - p0)
    return term1 - term2*U
# end def

def A_(X,U,V):
    # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    The sensing operator A in gFM. X is the data matrix; UV'=M as in gFM. The X should be zero-mean and unit-variance.
    \mathcal{A}(M) = { x_i' (M +M') x_i/2}_{i=1}^n where M=UV'
    :param X: a $d \times n$ feature matrix
    :param U: $d \times k$ matrix
    :param V: $d \times k$ matrix
    :return: z = A(UV')
    """

    z = numpy.sum( X.T.dot(U) * X.T.dot(V), axis=1, keepdims=True)
    return z

# end def

def ApW_(X,p,W):
    # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Compute z=A'(p)W, X should be zero-mean and unit-variance
    : param X: feature matrix, $d \times n$
    :param p: $n \times 1$ vector
    :param W: $d \times k$ matrix
    :param mean: mean vector of features, $d \times 1$ vector.
    :param sigma: std of features, $d \times 1$ vector
    :return: $d \times k$ matrix
    """


    z = X.dot(p*X.T.dot(W))
    return z
# end def