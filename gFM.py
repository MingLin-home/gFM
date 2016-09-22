"""
This gFM toolbox provides efficient solvers for the Generalized Factorization Machine (gFM) that can handle Tera-byte datasets.

There are two solvers provided:
*) gFM_BatchSolver implements the batch updating where the whole dataset can be loaded into memory.
*) gFM_MiniBatchSolver implements the mini-batch version of gFM_BatchSolver where we can load dataset in a mini-batch style.

For installation and usage information, please refer to README.txt and demonstration scripts.

@author: Ming Lin
@contact: linmin@umich.edu
"""

import numpy


class MiniBatchSolver(object):
    """
    The mini-batch solver for gFM
    """
    def __init__(self, rank_k, data_mean, data_std, data_moment3, data_moment4, load_data_func,load_data_func_para=None, max_iter=20,max_init_iter=10, lambd_M=numpy.Inf, lambd_w=numpy.Inf, ):
        # type: (int, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, object, dict, numpy.ndarray, numpy.ndarray)
        """
        Create a new gFM_MiniBatchSolver object.
        :param rank_k: The rank of the target second order matrix $M*$ in gFM
        :param data_mean: The mean of the data. $d\times 1$ vector.
        :param data_std: The std of the data. $d\times 1$ vector.
        :param data_moment3: The 3rd order moment of the data. $d\times 1$ vector.
        :param data_moment4: The 4th order moment of the data. $d\times 1$ vector.
        :param load_data_func: load_data_func: call back function to load a mini-batch data.
            The function will be called in each iteration as: X,y,load_data_func_para = load_data_func(load_data_func_para).
            load_data_func should return (None,None,load_data_func_para) if no data can be load.
            When reach the end of the dataset, the function should wrap to the start in next call.
        :param load_data_func_para: The function parameters passed to load_data_func in the call-back.
        :param max_iter: the number of iterations
        :param max_init_iter: the number of initialization iterations

        :param lambd_M: The Frobenius norm constraint for M
        :param lambd_w: The $\ell_2$-norm constraint for w
        """
        self.k = rank_k
        self.lambd_M = lambd_M
        self.lambd_w = lambd_w
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_moment3 = data_moment3
        self.data_moment4 = data_moment4
        self.d = data_mean.shape[0]
        self.one_over_phi_1_kappa2 = 1 / (self.data_moment4 - 1 - self.data_moment3 ** 2)
        self.max_iter = max_iter
        self.max_init_iter = max_init_iter
        self.load_data_func = load_data_func
        self.load_data_func_para = load_data_func_para

        self.U = None
        self.V = None
        self.w = None

        return

    def save_model(self, file_name):
        # type: (str) -> object
        """
        Save gFM model.
        :param file_name: File-like object or string. Save model to the file.
        :return: self
        """
        numpy.savez_compressed(file_name, U=self.U, V=self.V, w=self.w,
                               data_mean=self.data_mean, data_std=self.data_std,
                               data_moment3=self.data_moment3, data_moment4=self.data_moment4,
                               max_iter=self.max_iter, max_init_iter=self.max_init_iter)
        return self

    def load_model(self, file_name):
        """
        Load gFM model from file_name.
        :param file_name: File-like object or string. Load model from the file_name
        :return: self
        """
        the_loaded_file = numpy.load(file_name)
        self.U = the_loaded_file['U']
        self.V = the_loaded_file['V']
        self.w = the_loaded_file['w']
        self.data_mean = the_loaded_file['data_mean']
        self.data_std = the_loaded_file['data_std']
        self.data_moment3 = the_loaded_file['data_moment3']
        self.data_moment4 = the_loaded_file['data_moment4']
        self.max_iter = the_loaded_file['max_iter']
        self.max_init_iter = the_loaded_file['max_init_iter']

        self.one_over_phi_1_kappa2 = 1/(self.data_moment4-1-self.data_moment3**2)
        return self
    # end def

    def initialization(self,max_init_iter=10):
        self.initialization_begin()
        for t in xrange(max_init_iter):
            xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
            while xt is not None:
                self.initialization_load_minibatch_data(xt, yt)
                xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
            # end while
            self.initialization_update_one_epoch()
        # end for

        # Init V
        xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
        while xt is not None:
            self.iteration_load_minibatch_data_to_update_V(xt, yt)
            xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
        # end while
        self.iteration_update_V_one_epoch()

        # Init w
        xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
        while xt is not None:
            self.iteration_load_minibatch_data_to_update_w(xt, yt)
            xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
        # end while
        self.iteration_update_w_one_epoch()

        self.initialization_end()
        self.iteration_begin()
        return self
    # end def

    def iterate_train(self,max_iter=20):
        for t in xrange(max_iter):
            Xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
            while Xt is not None:
                self.iteration_load_minibatch_data_to_update_U(Xt, yt)
                Xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
            # end while
            self.iteration_update_U_one_epoch()

            Xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
            while Xt is not None:
                self.iteration_load_minibatch_data_to_update_V(Xt, yt)
                Xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
            # end while
            self.iteration_update_V_one_epoch()

            Xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
            while Xt is not None:
                self.iteration_load_minibatch_data_to_update_w(Xt, yt)
                Xt, yt, self.load_data_func_para = self.load_data_func(self.load_data_func_para)
            # end while
            self.iteration_update_w_one_epoch()
            self.iteration_update_U_V_w_at_the_end_of_epoch()
        # end for
        return self
    # end def

    def fit(self):
        """
        Train gFM using mimi-batch updating
        """
        self.initialization(self.max_init_iter)
        self.iterate_train(self.max_iter)
        return self
    # end def

    def decision_function(self,X):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Compute the decision values $s$ of X such that $\sign{s}$ is the predicted labels of X
        :param X: $d \times n$ feature matrix.
        :return: The decision values of X, $n \times 1$ vector
        """
        X = (X - self.data_mean) / self.data_std
        the_decision_values = A_(X,self.U,self.V) + X.T.dot(self.w)
        return the_decision_values

    def predict(self,X):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Predict the labels of X
        :param X: $d \times n$ feature matrix.
        :return: The predicted labels
        """
        return numpy.sign(self.decision_function(X))

    def initialization_begin(self):
        self.V = numpy.zeros((self.d, self.k))
        self.w = numpy.zeros((self.d,1))
        U, _ = numpy.linalg.qr(numpy.random.randn(self.d, self.k))
        self.U = U
        self.n = 0
        self.mathcal_M_cache = numpy.zeros((self.d, self.k))
        self.mathcal_W_cache = numpy.zeros((self.d,1))
        self.U_n = 0
        self.V_n = 0
        self.w_n = 0

    # end def

    def initialization_update_one_epoch(self):
        U_new,_ = numpy.linalg.qr(self.mathcal_M_cache/(2*self.n))
        self.mathcal_M_cache = numpy.zeros((self.d,self.k))
        self.U_new = U_new
        self.U = U_new
        self.n = 0

    def initialization_load_minibatch_data(self,X,y):
        X = (X-self.data_mean)/self.data_std
        y = numpy.asarray(y, dtype=numpy.float)
        self.mathcal_M_cache += mathcal_M_(y,self.U,X,self.data_moment3,self.one_over_phi_1_kappa2)
        self.n += X.shape[1]
    # end def

    def initialization_end(self):
        self.mathcal_M_cache = None
        self.n = None
        self.U = self.U_new
        self.V = self.V_new
        self.w = self.w_new

    def iteration_begin(self):
        self.mathcal_M_cache = numpy.zeros((self.d, self.k))
        self.mathcal_W_cache = numpy.zeros((self.d,1))
        self.U_n = 0
        self.V_n = 0
        self.w_n = 0


    def iteration_load_minibatch_data_to_update_U(self,X,y):
        X = (X - self.data_mean) / self.data_std
        y = numpy.asarray(y,dtype=numpy.float)
        hat_y = A_(X,self.U, self.V) + X.T.dot(self.w)
        dy = y-hat_y
        self.U_n += X.shape[1]
        self.mathcal_M_cache += mathcal_M_(dy, self.U,X, self.data_moment3,self.one_over_phi_1_kappa2)
        return self
    # end def

    def iteration_load_minibatch_data_to_update_w(self,X,y):
        X = (X - self.data_mean) / self.data_std
        y = numpy.asarray(y,dtype=numpy.float)
        hat_y = A_(X,self.U_new, self.V_new) + X.T.dot(self.w)
        dy = y-hat_y
        self.w_n += X.shape[1]
        self.mathcal_W_cache += mathcal_W_(dy,X,self.data_moment3,self.data_moment4,self.one_over_phi_1_kappa2)
        return self
    # end def

    def iteration_update_w_one_epoch(self):
        w_new = self.mathcal_W_cache/self.w_n + self.w
        if numpy.linalg.norm(w_new) > self.lambd_w: w_new = w_new / numpy.linalg.norm(w_new) * self.lambd_w
        self.w_new = w_new

        self.w_n = 0
        self.mathcal_W_cache = numpy.zeros((self.d,1))
        return self
    # end def


    def iteration_update_U_one_epoch(self):
        U_new = self.mathcal_M_cache/(2*self.U_n) +  0.5*self.U.dot(self.V.T.dot(self.U))+0.5*self.V.dot(self.U.T.dot(self.U))
        U_new,_ = numpy.linalg.qr(U_new)
        self.U_new = U_new

        self.U_n = 0
        self.mathcal_M_cache = numpy.zeros((self.d, self.k))
        return self
    # end def



    def iteration_load_minibatch_data_to_update_V(self,X,y):
        X = (X - self.data_mean) / self.data_std
        y = numpy.asarray(y,dtype=numpy.float)
        hat_y = A_(X,self.U, self.V) + X.T.dot(self.w)
        dy = y-hat_y
        self.V_n += X.shape[1]
        self.mathcal_M_cache +=  mathcal_M_(dy, self.U_new,X, self.data_moment3,self.one_over_phi_1_kappa2)
        return self
    # end def

    def iteration_update_V_one_epoch(self):
        V_new = self.mathcal_M_cache/(2*self.V_n) +  0.5*self.U.dot(self.V.T.dot(self.U_new))+0.5*self.V.dot(self.U.T.dot(self.U_new))
        if numpy.linalg.norm(V_new) > self.lambd_M: V_new = V_new / numpy.linalg.norm(V_new) * self.lambd_M
        self.V_new = V_new

        self.V_n = 0
        self.mathcal_M_cache = numpy.zeros((self.d, self.k))
        return self
    # end def

    def iteration_update_U_V_w_at_the_end_of_epoch(self):
        self.U = self.U_new
        self.V = self.V_new
        self.w =self.w_new

# end class

class BatchSolver(object):
    """
    The batch solver for gFM when the whole dataset can be loaded in memory.
    """
    def __init__(self, rank_k, data_mean = None, data_std = None, data_moment3=None, data_moment4 = None, max_iter=20, max_init_iter = 10, lambd_M=numpy.Inf, lambd_w=numpy.Inf):
        """
        Initialize a gFM_BatchSolver instance.
        :param rank_k: The rank of the target second order matrix in gFM ($M^*$). Should be of type int.
        :param data_mean: The mean of the data. $d\times 1$ vector.
        :param data_std: The std of the data. $d\times 1$ vector.
        :param data_moment3: The 3rd order moment of the data. $d\times 1$ vector.
        :param data_moment4: The 4th order moment of the data. $d\times 1$ vector.
        :param max_iter: The number of iterations for training.
        :param max_init_iter: The number of iterations in initialization step.
        :param lambd_M: The Frobenius norm constraint for M
        :param lambd_w: The $\ell_2$-norm constraint for w
        """
        self.rank_k = rank_k
        self.lambd_M = lambd_M
        self.lambd_w = lambd_w
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_moment3 = data_moment3
        self.data_moment4 = data_moment4
        self.max_iter = max_iter
        self.max_init_iter = max_init_iter
        return

    def fit(self,X,y):
        # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> object
        """
        Train gFM with data X and label y.
        :param X: Feature matrix, $d \times n$.
        :param y: Label vector, $n times 1$.
        :return: self
        """
        self.initialization(X, y, max_init_iter=self.max_init_iter)
        self.iterate_train(X, y, self.max_iter)
        return self

    def decision_function(self,X):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Compute the decision values $s$ of X such that $\sign{s}$ is the predicted labels of X
        :param X: $d \times n$ feature matrix.
        :return: The decision values of X, $n \times 1$ vector
        """
        X = (X - self.data_mean) / self.data_std
        the_decision_values = A_(X,self.U,self.V) + X.T.dot(self.w)
        return the_decision_values

    def predict(self,X):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        Predict the labels of X
        :param X: $d \times n$ feature matrix.
        :return: The predicted labels
        """
        return numpy.sign(self.decision_function(X))

    def save_model(self,file):
        # type: (str) -> object
        """
        Save gFM model.
        :param file: File-like object or string. Save model to the file.
        :return: self
        """
        numpy.savez_compressed(file,U = self.U, V = self.V, w = self.w,
                               data_mean = self.data_mean, data_std = self.data_std,
                               data_moment3 = self.data_moment3,data_moment4 = self.data_moment4,
                               max_iter = self.max_iter, max_init_iter = self.max_init_iter)
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
        self.data_mean = the_loaded_file['data_mean']
        self.data_std = the_loaded_file['data_std']
        self.data_moment3 = the_loaded_file['data_moment3']
        self.data_moment4 = the_loaded_file['data_moment4']
        self.max_iter = the_loaded_file['max_iter']
        self.max_init_iter = the_loaded_file['max_init_iter']

        self.one_over_phi_1_kappa2 = 1/(self.data_moment4-1-self.data_moment3**2)

        return self
    # end def


    def initialization(self, X, y, max_init_iter=10):
        # type: (numpy.ndarray, numpy.ndarray, int) -> numpy.ndarray
        """
        Use trancated SVD to initialize U0,V0. Batch updating.
        :param X: feature matrix, $d \times n$
        :param y: label vector, $n times 1$
        :param max_init_iter: the number of iterations for initialization. max_iter=10 is usually good enough
        :return: None
        """

        self.d = X.shape[0]
        n = X.shape[1]

        if self.data_mean is None:
            self.data_mean = X.mean(axis=1,keepdims=True)
            X = X - self.data_mean
        if self.data_std is None:
            self.data_std = numpy.maximum(X.std(axis=1,keepdims=True),1e-12)
            X = X/self.data_std
        if self.data_moment3 is None:
            self.data_moment3 = numpy.mean(X**3,axis=1,keepdims=True)
        if self.data_moment4 is None:
            self.data_moment4 = numpy.mean(X**4,axis=1,keepdims=True)


        self.one_over_phi_1_kappa2 = 1/(self.data_moment4-1-self.data_moment3**2)

        y = numpy.asarray(y,dtype=numpy.float)

        U,_ = numpy.linalg.qr( numpy.random.randn(self.d,self.rank_k))
        for t in xrange(max_init_iter/10):
            for i in xrange(10):
                U = mathcal_M_(y,U,X,self.data_moment3, self.one_over_phi_1_kappa2)/(2*n)
            U,_ = numpy.linalg.qr(U)
        # end for t

        # V = numpy.zeros((self.d, self.rank_k))

        # update V
        V = mathcal_M_(y,U, X, self.data_moment3, self.one_over_phi_1_kappa2)/(2*n)

        # update w
        hat_y = A_(X, U, V)
        dy = y - hat_y
        w = mathcal_W_(dy, X, self.data_moment3, self.data_moment4, self.one_over_phi_1_kappa2)/n

        self.U = U
        self.V = V
        self.w = w

        return self
    # end def



    def iterate_train(self, X, y, max_iter=1, z_score_normalized=False):
        # type: (numpy.ndaray, numpy.ndaray, int) -> numpy.ndaray
        """
        Update U,V,w using batch iteration.
        :param X: feature matrix, $d \times n$ matrix
        :param y: label vector, $n \times 1$
        :param max_iter: number of iterations
        :param z_score_normalized: If True, it means that the dataset X has been z-score normalized already. If not (default), the solver will z-score normalized it.
        """
        U = self.U
        V = self.V
        w = self.w
        n = X.shape[1]
        y = numpy.asarray(y,dtype=numpy.float)

        if z_score_normalized == False:
            X = (X-self.data_mean)/self.data_std

        for t in xrange(max_iter):
            hat_y = A_(X,U,V) + X.T.dot(w)
            dy = y-hat_y

            # update U
            U_new = mathcal_M_(dy,U,X, self.data_moment3,self.one_over_phi_1_kappa2)/(2*n) + \
                0.5*U.dot(V.T.dot(U))+0.5*V.dot(U.T.dot(U))
            # V_new = U_new
            U_new,_ = numpy.linalg.qr(U_new)

            # update V
            V_new = mathcal_M_(dy,U_new,X, self.data_moment3, self.one_over_phi_1_kappa2)/(2*n) + \
                    0.5 * U.dot(V.T.dot(U_new)) + 0.5 * V.dot(U.T.dot(U_new))

            # update w
            hat_y = A_(X,U_new,V_new) + X.T.dot(w)
            dy = y-hat_y
            w_new = mathcal_W_(dy,X, self.data_moment3, self.data_moment4, self.one_over_phi_1_kappa2)/n + w


            if numpy.linalg.norm(V_new) > self.lambd_M: V_new = V_new / numpy.linalg.norm(V_new)*self.lambd_M
            if numpy.linalg.norm(w_new) > self.lambd_w: w_new = w_new / numpy.linalg.norm(w_new)*self.lambd_w
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

def mathcal_W_(y,X,data_moment3, data_moment4, one_over_phi_1_kappa2):
    # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Return $\mathcal{W}(y)*n given the constant parameters. X should be zero-mean unit-variance
    """
    term1 = (data_moment4-1)*one_over_phi_1_kappa2*X.dot(y)
    term2 = -data_moment3*one_over_phi_1_kappa2*( (X**2).dot(y)-numpy.sum(y) )
    return term1+term2
# end def

def mathcal_M_(y,U,X,data_moment3, one_over_phi_1_kappa2):
    # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    Return $\mathcal{M}(y)U*2n, given the constant parameters. X should be zero-mean unit-variance
    """

    term1 = (X*y.T).dot(X.T.dot(U))
    term2 = -numpy.sum(y) - 2*data_moment3*one_over_phi_1_kappa2*(X.dot(y))-(1-2*one_over_phi_1_kappa2)*( (X**2).dot(y)- numpy.sum(y))
    return term1 + term2*U
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