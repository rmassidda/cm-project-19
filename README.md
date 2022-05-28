Project 13
==========

(P) is the linear least squares problem

$$
\min_w |\hat{X}w - y|
$$

where

$$
\hat{X} = \begin{bmatrix} X^T\\\\ I \end{bmatrix}
$$

with X is the (tall thin) matrix from the ML-cup dataset by prof. Micheli, and y is a random vector.

(A1) an algorithm of the class of limited-memory quasi-Newton methods [references: J. Nocedal, S. Wright, Numerical Optimization, <a href="https://arxiv.org/abs/1406.2572" class="uri">https://arxiv.org/abs/1406.2572</a>].

(A2) is thin QR factorization with Householder reflectors [Trefethen, Bau, Numerical Linear Algebra, Lecture 10].

No off-the-shelf solvers allowed. In particular you must implement yourself the thin QR factorization.


optimization.py
===============
This script contains the implementation of
various optimization techniques such as
BFGS, L-BFGS and the Newton method.

numerical.py
============
This script contains the implementation of
numerical methods such as QR, modified QR
and backsubstitution.

utils.py
========
This script contains various auxiliary
functions.

experiments.py
==============
This script sistematically executes the
experiments used to generate the plots
and the tables in the report.

plot.py
=======
This script reads from the dumps of the
experiments.py script to generate both
plots and tables.
