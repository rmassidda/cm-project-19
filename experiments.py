from utils import load_dataset, lls_functions, theta_angled
from optimization import LBFGS, BFGS, Newton, optimize
from concurrent.futures import ThreadPoolExecutor
from numerical import qr, modified_qr, q1, back_substitution
import numpy as np
import sys
import time

"""Computes the solution of the least squares problem
   by using an optimization method.

Parameters
----------
y : R^m
    The 
method: Class
    The desired optimizer
params: Dict
    The parameters to construct the optimizer

Returns
-------
w_c : R^n
    The candidate solution
s   : int
    The number of steps
"""
def optimization_solver(y, method, params):
    # Starting point
    f, g, Q      = lls_functions(X_hat, X, y)
    params['w']  = np.random.randn(n)
    params['gw'] = g(params['w'])
    # Construct optimizer and solve
    opt     = method(**params)
    w_c, s  = optimize(f,g,Q,opt)
    return w_c, s

"""Computes the solution of the least squares problem
   by using the QR factorization.

Parameters
----------
y : R^m
    The 
qr_factorization: R^{m,n} -> R^{??}, R^{??}
    The desired factorizer

Returns
-------
w_c : R^n
    The candidate solution
s   : int
    The number of steps
"""
def numerical_solver(y, qr_factorization):
    R, vects = qr_factorization(X_hat)
    Q1       = q1(vects, m)
    c        = np.dot(Q1.T, y)
    w_c      = back_substitution(R[:n, :], c)
    return w_c, 1

"""Computes the solution of the least squares problem
   by using the default numpy method.

Parameters
----------
y : R^m
    The 

Returns
-------
w_c : R^n
    The candidate solution
s   : int
    The number of steps
"""
def numpy_solver(y):
    w_c = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    return w_c, 1

"""Computes the solution of the least squares problem
   by using an arbitrary solver

Parameters
----------
y : R^m
    The 

solver: R^m -> (R^n, in)
    The solver to use for the LLS problem

Returns
-------
dur : R
    The elapsed time
w   : R^n
    The candidate solution
s   : int
    The number of steps
"""
def task(y, solver):
    start = time.time()
    w, s  = solver(y)
    end   = time.time()
    dur   = end - start
    return dur, w, s

"""Computes the solution of the least squares problem
   in parallel for multiple values of y

Parameters
----------
solver: R^m -> (R^n, int)
    Function that solves the LLS problem
Y : List(R^m)
    Contains the values of y for the multiple LLS problems
name : string
    Name of the solver

Returns
-------
log : R^l x 3
    Array containing for each y: duration, residual and steps
"""
def run_experiment(solver, Y, name, nw):
    log = np.zeros((len(Y), 3))
    print('Running', name,len(Y),'times with',nw,'workers')

    # Multithreading to run parallel experiments
    if nw == 0:
        results = list(map(lambda y: task(y, solver), Y))
    else:
        with ThreadPoolExecutor(max_workers=nw) as p:
            results = p.map(lambda y: task(y, solver), Y)

    # Log results (duration, residual, steps)
    for i, (y, res) in enumerate(zip(Y,results)):
        f, _, _ = lls_functions(X_hat, X, y)
        log[i, :] = [res[0], f(res[1]), res[2]]

    return log

if __name__ == '__main__':
    # Number of workers
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Data loading
    X, X_hat = load_dataset()
    m, n     = X_hat.shape

    # Hessian and its inverse
    Q = X @ X.T + np.eye(n)
    H = np.linalg.inv(Q)

    # Input analysis
    KX = np.linalg.cond(X_hat)
    print('Condition number', KX)

    # Default solvers
    np_lls = lambda y: numpy_solver(y)
    newton = lambda y: optimization_solver(y, Newton, {'H': H})
    lbfgs  = lambda y: optimization_solver(y, LBFGS, {})
    std_qr = lambda y: numerical_solver(y, qr)
    mod_qr = lambda y: numerical_solver(y, lambda A: modified_qr(A, m-n+1))
    def_solvers = [np_lls, newton, lbfgs, mod_qr]
    def_names   = ['LLS Numpy', 'Newton', 'LBFGS', 'QR*']

    # Constants
    MAX_REP = 5     # Repetitions
    MAX_G   = 30    # Granularity
    MAX_S   = 2048  # Maximum steps for the iterative methods

    # Test the difference in performances for QR and QR*
    Y = [np.random.randn(m) for i in range(MAX_REP)]
    std_results = run_experiment(std_qr, Y, 'QR', nw)
    mod_results = run_experiment(mod_qr, Y, 'QR*', nw)
    np.save('results/mod_results', mod_results)
    np.save('results/std_results', std_results)

    # Test the different initializations
    # (Number of runs, Initialization methods, Residual per step)
    resid = np.zeros((MAX_REP, 3, MAX_S)) - 1
    for i in range(MAX_REP):
        # Init
        y = np.random.randn(m)
        f, g, Q = lls_functions(X_hat, X, y)
        w  = np.random.randn(n)
        gw = g(w)

        # LBFGS Gamma
        opt    = LBFGS(w, gw)
        w_list = optimize(f,g,Q,opt,conv_array=True, verbose=True)
        _resid = np.array([f(w) for w in w_list])
        resid[i, 0, :len(_resid)] = _resid

        # LBFGS Identity
        opt = LBFGS(w, gw, init='identity')
        w_list = optimize(f,g,Q,opt,conv_array=True, verbose=True)
        _resid = np.array([f(w) for w in w_list])
        resid[i, 1, :len(_resid)] = _resid

        # BFGS
        opt    = BFGS(w, gw, np.eye(n))
        w_list = optimize(f,g,Q,opt,conv_array=True, verbose=True)
        _resid = np.array([f(w) for w in w_list])
        resid[i, 2, :len(_resid)] = _resid

    # Average the runs and sort
    resid = np.average(resid, axis=0)
    np.save('results/initializers', resid)

    # Range of values to plot \theta
    theta_rng = np.linspace(0, np.pi/2, MAX_G)
    results   = np.zeros((MAX_REP, len(def_solvers), MAX_G, 3))
    X_cond  = np.zeros((MAX_REP, MAX_G))
    y_cond  = np.zeros((MAX_REP, MAX_G))
    for i in range(MAX_REP):
        # Random generate y at a given angle \theta
        Y = [theta_angled(X_hat, theta)[1] for theta in theta_rng]

        # Time, steps and residual for each solver
        for j, (solver, name) in enumerate(zip(def_solvers, def_names)):
            results[i,j,:] = run_experiment(solver, Y, name, nw)

        # Absolute conditioning
        X_cond[i,:] = KX + KX**2 * np.tan(theta_rng)
        # Relative conditioning
        y_cond[i,:] = KX / np.cos(theta_rng)
    results  = np.average(results, axis=0)
    X_cond = np.average(X_cond, axis=0)
    y_cond = np.average(y_cond, axis=0)
    np.save('results/theta_defaults', results)
    np.save('results/theta_Xcond', X_cond)
    np.save('results/theta_ycond', y_cond)

    # Range of values to plot t
    t_rng   = np.linspace(1, n, MAX_G).astype(int)
    results = np.zeros((MAX_REP, MAX_G, 3))
    Y       = [np.random.randn(m) for _ in range(MAX_REP)]
    for i, t in enumerate(t_rng):
        lbfgs = lambda y: optimization_solver(y, LBFGS, {'t': t})
        results[:,i,:] = run_experiment(lbfgs, Y, 'LBFGS t='+str(t), nw)
    results = np.average(results, axis=0)
    np.save('results/t_rng', t_rng)
    np.save('results/t_lbfgs', results)
