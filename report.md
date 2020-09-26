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
  The linear least square problem can be tackled using a wide range of optimization or numerical methods. The L-BFGS method of the class of limited-memory quasi-Newton algorithms has been chosen for the former, whilst the thin QR factorization with Householder reflectors for the latter. Both these algorithms have been implemented from scratch using Python language, to finally experiment over their performances in terms of precision, stability and speed. The accordance of the implementations with the underlying theoretical models is also studied and discussed.
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

The matrix $\hat{X}$ is actually composed in the following way:

$$
\hat{X} = \begin{bmatrix}X^T\\I\end{bmatrix}
$$

Where $X \in \mathbb{R}^{n \times k}$ is a tall thin matrix, thus $m = k + n$.
The LLS problem can be dealt both with iterative methods or with direct numerical methods.
One algorithm has been chosen for each of these fields to finally discuss their experimental results.

## L-BFGS
<!-- Breve descrizione del metodo L-BFGS, a che famiglia appartiene e come si distingue da BFGS -->
The L-BFGS is an iterative method of the quasi-Newton limited-memory class.
It updates itself using the rule:
$$
x_{k+1} = x_k - \alpha_k H_k \nabla f_k
$$

The fact ??? differentiates it with the standard BFGS method.
The initialization of the $H_0^k$ matrix is fundamental and it is used to prove convergence results.
How are the matrices updated.
What line search do we use, how it is implemented.

## Thin QR factorization
For the numerical counterpart, the thin QR factorization with Householder reflectors has been implemented as described in [@trefethen_numerical_1997].

By using the Householder QR factorization, the matrix $R$ is constructed in place of $\hat{X}$ and the $n$ reflection vectors $v_1, \dots, v_n$ are stored.
The reduced matrix $\hat{R}$ is trivially obtainable by slicing as in $\hat{R} = R_{1:n,1:n}$. In fact, given that $\hat{X}$ is already stored in memory and fully needed, there would be no advantage in directly constructing the reduced matrix.

By using the Householder vectors it is also possible to implicitly compute $\hat{Q}^Tb$ to finally obtain $w_*$ by back substitution over the upper-triangular system $\hat{R}w = \hat{Q}^T b$.

# Algorithmic analysis
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

## Convergence of L-BFGS

@liu_limited_1989 defines three assumptions that lead to a theorem proving that the L-BFGS algorithm globally converges and moreover that it converges $\mathbb{R}$-linearly.

The first assumption is on the objective function $f$, that should be twice continuously differentiable.
This is in fact true and we can define the gradient and the Hessian of the objective function as in:

$$
\nabla f(w) = \hat{X}^T ( \hat{X} w - y ) \\
$$

$$
\nabla^2 f(w) = \hat{X}^T \hat{X}
$$

Moreover the Hessian can be defined as a positive definite since it can be rearranged in the following way:

$$
\begin{split}
\nabla^2 f(w) & = \hat{X}^T \hat{X}  \\
& = \begin{vmatrix}X I\end{vmatrix} \begin{vmatrix}X^T \\ I\end{vmatrix} \\
& = XX^T + I
\end{split}
$$

This also implies that the objective function $f$ is a convex function and comes in handy for the second assumption according to the sublevel set $D=\{w \in \mathbb{R}^n | f(w) \leq f(w_0)\}$ must be convex.
Given that if a function is convex all of its sublevel sets are convex sets this is satisfied.

$$
\begin{split}
\forall x,y \in D \cap \lambda \in [0,1] \\
& \textrm{f convex} \\
& \implies f(\lambda x + (1-\lambda)y) \\
& \leq \lambda f(x) + (1-\lambda) f(y) \\
& \leq \lambda f(w_0) + (1-\lambda) f(w_0) \\
& = f(w_0) \\
& \implies \lambda x + (1-\lambda y) \in D
\end{split}
$$

The third assumption requires that there exist two positive constant $M_1$ and $M_2$ such that $\forall z \in \mathbb{R}^n , w \in D$:

$$
M_1 \|z\|^2 \leq z^T \nabla^2 f(w) \leq M_2 \|z\|^2
$$

or equivalently

$$
M_1 I \preceq \nabla^2 f(w) \preceq M_2 I
$$

The first part of this equation is satisfied by the fact that the Hessian is positive definite.
The second requires instead to satisfy

$$
\begin{split}
\nabla^2 f(w) \preceq M_2 I & \equiv XX^T + I \preceq M_2 I  \\
& \equiv XX^T \preceq (M_2 - 1) I
\end{split}
$$

While this depends on the input data it should be noticed that the resulting matrix $XX^T$ is symmetric, and therefore has $n$ real eigenvalues, therefore it exists $\lambda_{max} = M_2 - 1$ to satisfy this third assertion.

Other then this assumptions, the theorem requires for the sequence of Hessian substitutes to be bounded.
This clearly depends on the matrix chosen to initialize each run.
What else can we say?

## Armijo-Wolfe Line Search
The algorithm described in @al-baali_efficient_1986 to perform an inexact line search is ensured to converge under the assumption that $\sigma > \rho$ (use the other notation here) since both the schemes are ensured to terminate.

# Input data
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

# Bibliography
