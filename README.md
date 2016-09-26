# gFM: A toolbox for generalized Factorization Machine with Terabyte Data #

The gFM toolbox provides the-state-of-the-art solvers for the generalized Factorization Machine (gFM). The gFM toolbox is able to learning gFM model with tens of thousands of dimensions on Terabyte datasets. The gFM toolbox can run on multiple CPUs or GPUs simultaneously. The gFM toolbox is very fast and has provable guarantees. Its learning ability is nearly optimal in theory.

gFM is available at

https://github.com/MingLin-home/gFM

If you find this software useful, please cite our work:

Ming Lin, Jieping Ye. A Non-convex One-Pass Framework for Factorization Machines and Rank-One Matrix Sensing. The Thirtieth Annual Conference on Neural Information Processing Systems (NIPS), 2016.

===== Bibtex Entry =====

@inproceedings{ming_lin_non-convex_2016,
  title = {A Non-convex One-Pass Framework for Factorization Machines and Rank-One Matrix Sensing}, 
  booktitle = {The Thirtieth Annual Conference on Neural Information Processing Systems (NIPS)},
  author = {{Ming Lin} and {Jieping Ye}},
  year = {2016}
}

Installation
============
gFM is a single python library. Copy gFM.py to your project root or directory in your PYTHONPATH environment variables. That's it!


Quick Usage
===========
Assume we have loaded the feature matrix X and the label vector y into memory. Each column of X present one instance and y is a column vector. We want to learn a rank-10 gFM model. To train gFM:

import gFM

k = 10

my_gFM_solver = gFM.BatchSolver(rank_k=k)

my_gFM_solver.fit(X,y)

To predict:

predicted_label = my_gFM_solver.predict(X)

To get the decision value (regression value):

decision_value = my_gFM_solver.decision_function(X)

API Design
===========

gFM toolbox implements the algorithm proposed in our NISP 2016 paper. It provides two version of the implementation: a batch version and a mini-batch version. The usage examples are given in the tutorial/ directory. For each version, gFM provides both high-level interface and low-lever interface. The high-level interface provides the standard API as Scipy. For example you call fit(X,y) function to train the model and call predict(X) to get the prediction. The low-level interface provides fine-control of the iteration process. It is useful when you want to check the intermediate results, for example, plotting the convergence curve.

The batch version is implemented in class BatchSolver. BatchSolver loads all data into memory thus is suitable when you have large enough memory. See tutorial/demo_gFM_BatchSolver_high_level_interface.py for high-level interface usage and tutorial/demo_gFM_BatchSolver_low_level_interface.py for low-level interface usage.

The mini-batch version is implemented in class MiniBatchSolver. It allows loading partial dataset in one mini-batch epoch. However the efficiency is traded for the memory constraint in this case. See tutorial/demo_gFM_MiniBatchSolver_high_level_interface.py for high-level interface usage and tutorial/demo_gFM_MiniBatchSolver_low_level_interface.py  for low-level interface usage.
