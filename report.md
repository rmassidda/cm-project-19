---
documentclass: IEEEtran
bibliography: bibliography.bib
...

\title{Analysis of optimization and numerical approaches to solve the linear least square problem}

\author{
\IEEEauthorblockN{
Emanuele Cosenza\IEEEauthorrefmark{1},
Riccardo Massidda\IEEEauthorrefmark{2}} \\
\vspace{2mm}
\IEEEauthorblockA{
Department of Computer Science \\
University of Pisa \\
\vspace{2mm}
\IEEEauthorrefmark{1}\,\href{mailto:e.cosenza3@studenti.unipi.it}{e.cosenza3@studenti.unipi.it},
\IEEEauthorrefmark{2}\,\href{mailto:r.massidda@studenti.unipi.it}{r.massidda@studenti.unipi.it}}}

\maketitle

\begin{abstract}
  The linear least square problem can be tackled using a wide range of optimization or numerical methods. The saddle-free Newton method of the class of limited-memory quasi-Newton algorithms has been chosen for the former, whilst the thin QR factorization with Householder reflectors for the latter. Both these algorithms have been implemented from scratch using Python language, to finally experiment over their performances in terms of precision, stability and speed. The accordance of the implementations with the underlying theoretical models is also studied and discussed.
\end{abstract}

<!--
Setting the stage
=================

The first section of your report should contain a description of the problem and the methods that you plan to use.
This is intended just as a brief recall, to introduce some notation and specify which variants of the methods you are planning to use exactly.
Discuss the reasons behind the choices you make (the one you can make, that is, since several of them will be dictated by the statement of the project and cannot be questioned).
Your target audience should be someone who is already sufficiently familiar with the content of the course.
This is not the place to show your knowledge and repeat a large part of the theory: we are sure that you all can do that, given enough time, books, slides, and internet bandwidth.
A more in-depth mathematical part is expected in the next stage.
In case adapting the algorithm to your problem requires some further mathematical derivation (example: developing an exact line search for your problem, when possible, or adapting an algorithm to deal more efficiently with the special structure of your problem), you are supposed to discuss it here with all the necessary mathematical detail.
You are advised to send us a version of this section by e-mail as soon as it is done, so that we can catch misunderstandings as soon as possible and minimize the amount of work wasted.
Note that we do not want to see code at this point: that would be premature to produce (for you) and unnecessarily complicated to read (for us).
-->

# Introduction
Given a dataset composed by a matrix $\hat{X} \in \mathbb{R}^{m \times n}$ with $m \geq n$ and a vector $y \in \mathbb{R}^m$, the solution of the linear least square (LLS) problem is the vector $w \in \mathbb{R}^n$ that fits best the data assuming a linear function between $\hat{X}$ and $y$. [@nocedal_numerical_2006 p. 50]
This can be formalized as the following minimization problem:

$$
w_* = \min_w \| \hat{\boldmath{X}} w - y \|_2
$$

The LLS problem can be dealt both with iterative methods or with direct numerical methods.
One algorithm has been chosen for each of these fields to discuss then their experimental results.

## Saddle-free Newton method
The presence of numerous saddle-points constitutes an issue to both Newton and quasi-Newton traditional iterative methods, the saddle-free Newton method (SFN) is aimed to replicate Newton dynamics yet repulsing saddle-points. [@dauphin_identifying_2014]

The original implementation uses a fixed number of steps to approximate the solution of the problem, we instead consider valuable a stopping criterion with accuracy $\epsilon$ over the norm of the gradient.

To overcome the constraints related to the positive definitess of the Hessian $H$ the SFN does not directly use it, as in a quasi-Newton method.
A matrix $|\mathbf{H}|$ obtained by taking the absolute value of each eigenvalue of $H$ is used in its place.
Also the exact computation of $|\mathbf{H}|$ is avoided, thus qualifying the SFN as a limited memory method.
This latter feature is obtained by optimizing the function in a lower-dimensional Krylov subspace, exploiting the Lanczos algorithm that produces the $k$ biggest eigenvectors of the Hessian matrix and using them as a base for the subspace.
Even inside the Lanczos algorithm the Hessian can be implicitly computed by using the so called Pearlmutter trick. [@pearlmutter_fast_1994]

The resulting matrix $|\hat{\mathbf{H}}|$ can then be re-used for multiple steps, assuming that the very same won't change much from one iteration to another.
The best number of steps $t$ without updating $|\hat{\mathbf{H}}|$ is not trivially determinable and it is so treated as an hyperparameter.

The damping coefficient $\lambda$ that maximizes the effectiveness of the step is not chosen with a rigorous sub-optimization task, instead as in the original paper a set of discrete values of different order of magnitude is tried.

## Thin QR factorization
For the numerical counterpart the thin QR factorization with Householder reflectors has been implemented as described in [@trefethen_numerical_1997].

By using the Householder QR factorization the matrix $R$ is constructed in place of $\hat{X}$, also the $n$ reflection vectors $v_1, \dots, v_n$ are stored.
The reduced matrix $\hat{R}$ is trivially obtainable by slicing as in $\hat{R} = R_{1:n,1:n}$, given that $\hat{X}$ is already stored in memory and fully needed there would not be any advantage in directly constructing the reduced matrix.

By using the Householder vectors it is also possible to implicitly compute $\hat{Q}^Tb$ to finally obtain $w_*$ by back substitution over the upper-triangular system $\hat{R}w = \hat{Q}^T b$.

# Bibliography

<!--
What to expect from the algorithm(s)
====================================
Next, we expect a brief recall of the algorithmic properties that you expect to see in the experiments. Is the algorithm
(if it is iterative) guaranteed to converge? Is it going to be stable and return a good approximation of the solution
(if it is direct)? What is its complexity? Are there any relevant convergence results? Are the hypotheses of these
convergence results (convexity, compactness, differentiability, etc.) satisfied by your problem? If not, what are the
“closest” possible results you have available, and why exactly they are not applicable? Do you expect this to be
relevant in practice?
Again, you are advised to send us a version of this section by e-mail as soon as it is done. Again, we do not want
to see code at this point.
-->

<!--
What is your input data
=======================
Next, we expect a brief description of the data you will test your algorithms on. For “ML projects” this will typcally
be provided by the ML course, but still a modicum of descripton is required. For “no-ML projects”, it will typically
have to be either generated randomly, or picked up from the Internet, or a combination of both. This is not necessarily
a trivial process, as, say, random generation should ensure that “interesting” properties of the data (what kind of
solution can be expected, how well or ill-conditioned the problem is, . . . ) is properly controlled by the parameters of
the random generator. These aspects should be thoroughly described in the report.
Again, you are advised to send us a version of this section by e-mail as soon as it is done. Again, we do not want
to see code (unless seeing how instances is generated is much simpler by looking at a short well-commented code than
at a long winding report).
-->
