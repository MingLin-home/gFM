# gFM: A toolbox for generalized Factorization Machine on Terabyte Data #

The gFM toolbox provides the-state-of-the-art solvers for the generalized Factorization Machine (gFM). The gFM toolbox is able to learning gFM model with tens of thousands of dimensions on Terabyte datasets. The gFM toolbox can run on multiple CPUs or GPUs simultaneously. The gFM toolbox is very fast and has provable guarantees. gFM is compatible to sklearn API.

gFM is available at

https://github.com/MingLin-home/gFM

If you find this software useful, please cite our work:

Ming Lin, Jieping Ye. A Non-convex One-Pass Framework for Generalized Factorization Machine and Rank-One Matrix Sensing. The Thirtieth Annual Conference on Neural Information Processing Systems (NIPS), 2016.

===== Bibtex Entry =====

@inproceedings{ming_lin_non-convex_2016,
  title = {A Non-convex One-Pass Framework for Factorization Machines and Rank-One Matrix Sensing}, 
  booktitle = {The Thirtieth Annual Conference on Neural Information Processing Systems (NIPS)},
  author = {{Ming Lin} and {Jieping Ye}},
  year = {2016}
}


Why another Factorization Machine library?
============

There is libfm and several python wrapper such as pywFM, pyFM or fastFM. However, none of these is able to convergece globally. The algorithms implemented in these packages might converge to local minima or saddle point.  Numerical experiments suggest that with very high probability these algorithms cannot converge to the global optimal solution.

The algorithm implemented in gFM is provably convergent to global optimal solution. The convergence rate is linear (geometrical). In addition, gFM implements a generalized version of Factorization Machine. The second order feature interaction coefficient matrix in gFM doesn't need to be semi-positive definite therefore is more powerful than conventional Factorization Machine.


Installation
============
gFM is a single python library. Copy gFM.py to your project root or any directory in your PYTHONPATH environment variable. That's it!


Quick Usage
===========
Assume we have loaded the feature matrix X and the label vector y into memory. Each column of X presents one instance and y is a column vector. We want to learn a rank-10 gFM model. To train gFM:

import gFM

k = 10

my_gFM_solver = gFM.BatchRegression(rank_k=k)

my_gFM_solver.fit(X,y)

To predict:

predicted_regression_value = my_gFM_solver.predict(X)

You can use gFM with sklearn modules such GridSearch and Cross-validation.


API Design
===========

gFM toolbox implements the algorithm proposed in our NISP 2016 paper. It provides a batch updating implementation and a mini-batch updating implementation. The usage examples are given in the tutorial/ directory. Also check the docstring in gFM.py for advanced usage.

The mini-batch version and GPU version is on the way!
