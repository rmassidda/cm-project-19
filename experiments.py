from utils import load_dataset, lls_functions, theta_angled
from optimization import LBFGS, BFGS, Newton
from concurrent.futures import ThreadPoolExecutor
from numerical import qr, modified_qr, q1, back_substitution
import numpy as np
import sys
import time

def optimization_solver(y, method, params):
    """
    Computes the solution of the least squares problem
    by using an optimization method.

    Parameters
    ----------
    y : ndarray
    method: Optimizer
    params: dict

    Returns
    -------
    w_c : ndarray
        The candidate solution
    s   : int
        The number of required steps
    """
    # Starting point
    f, g, Q      = lls_functions(X_hat, X, y)
    params['w']  = np.random.randn(n)
    params['gw'] = g(params['w'])
    # Construct optimizer and solve
    opt     = method(**params)
    w_c, s  = opt.optimize(f,g,Q)
    return w_c, s

def numerical_solver(y, qr_factorization):
    """
    Computes the solution of the least squares problem
    by using the QR factorization as implemented for
    the project

    Parameters
    ----------
    y : ndarray
    qr_factorization: function

    Returns
    -------
    w_c : ndarray
        The candidate solution
    s   : int
        The number of required steps
    """
    R, vects = qr_factorization(X_hat)
    Q1       = q1(vects, m)
    c        = np.dot(Q1.T, y)
    w_c      = back_substitution(R[:n, :], c)
    return w_c, 1

def numpy_solver(y):
    """
    Computes the solution of the least squares problem
    by using the provided np.linalg.lstsq method.

    Parameters
    ----------
    y : ndarray

    Returns
    -------
    w_c : ndarray
        The candidate solution
    s   : int
        The number of required steps
    """
    w_c = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    return w_c, 1

def numpy_qr_solver(y):
    """
    Computes the solution of the least squares problem
    by using the provided np.linalg.qr method.

    Parameters
    ----------
    y : ndarray

    Returns
    -------
    w_c : ndarray
        The candidate solution
    s   : int
        The number of required steps
    """
    Q1, R = np.linalg.qr(X_hat, mode='reduced')
    c    = np.dot(Q1.T, y)
    w_c  = back_substitution(R[:n, :], c)
    return w_c, 1

def task(y, solver):
    """
    Computes the solution of the least squares problem
    by using an arbitrary solver

    Parameters
    ----------
    y : ndarray
    solver : function

    Returns
    -------
    dur : float
        Time in seconds to solve the problem
    w_c : ndarray
        The candidate solution
    s   : int
        The number of required steps
    """
    start = time.time()
    w, s  = solver(y)
    end   = time.time()
    dur   = end - start
    return dur, w, s

def run_experiment(solver, Y, name, nw=0):
    """
    Computes either sequentially or in parallel
    multiple least squares problems.

    Parameters
    ----------
    solver : function
        Function that solves a LLS problem
    Y : ndarray
        Array containing the different y for each problem
    name : str
        Name of the solver
    nw : int, optional
        Number of workers, if zero the experiments are run
        sequentially without any overhead

    Returns
    -------
    log : ndarray
        Array containing for each solution the duration in
        seconds, the residual and the number of required
        steps
    """
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
    num_qr = lambda y: numpy_qr_solver(y)
    def_solvers = [np_lls, newton, lbfgs, num_qr, mod_qr]
    def_names   = ['LLS Numpy', 'Newton', 'LBFGS', 'QR Numpy', 'QR*']

    # Constants
    MAX_REP = 5     # Repetitions
    MAX_G   = 30    # Granularity
    MAX_S   = 2048  # Maximum steps for the iterative methods

    # Test conditioning upper bound for QR*
    print('QR* conditioning upper bound')
    theta_rng = np.linspace(0, np.pi/2, MAX_G)
    results   = np.zeros((MAX_REP, MAX_G, 3))
    opt_w     = np.zeros((MAX_REP, MAX_G, n))
    tilde_w   = np.zeros((MAX_REP, MAX_G, n))
    X_cond_ub = np.zeros((MAX_REP, MAX_G))
    y_cond_ub = np.zeros((MAX_REP, MAX_G))
    for i in range(MAX_REP):
        # Random generate y at a given angle \theta
        theta_pair  = [theta_angled(X_hat, theta) for theta in theta_rng]
        opt_w[i][:] = np.array([e[0] for e in theta_pair])
        Y           = [e[1] for e in theta_pair]

        for j, y in enumerate(Y):
            tilde_w[i,j] = mod_qr(y)[0]

        # X hat conditioning
        X_cond_ub[i,:] = KX + KX**2 * np.tan(theta_rng)
        # y conditioning
        y_cond_ub[i,:] = KX / np.cos(theta_rng)
    opt_w     = np.average(opt_w, axis=0)
    tilde_w   = np.average(tilde_w, axis=0)
    X_cond_ub = np.average(X_cond_ub, axis=0)
    y_cond_ub = np.average(y_cond_ub, axis=0)
    np.save('results/opt_w',opt_w)
    np.save('results/tilde_w',tilde_w)
    np.save('results/X_cond_ub',X_cond_ub)
    np.save('results/y_cond_ub',y_cond_ub)
    print('QR* conditioning upper bound done')

    # Test the convergence rate for different increasing conditioning
    theta_rng = np.linspace(0, np.pi/2, MAX_G)
    avg_r     = np.zeros((2*MAX_REP,MAX_G))
    for i in range(2*MAX_REP):
        # Random generate y at a given angle \theta
        Y = [theta_angled(X_hat, theta)[1] for theta in theta_rng]
        for j, y in enumerate(Y):
            # Init
            f, g, Q = lls_functions(X_hat, X, y)
            w  = np.random.randn(n)
            gw = g(w)

            # LBFGS Gamma
            opt    = LBFGS(w, gw)
            w_list = opt.optimize(f,g,Q,conv_array=True, verbose=False)
            resid  = np.array([f(w) for w in w_list])

            # Compute r
            f_opt = resid[-1]
            diff  = resid - f_opt
            r_rng = np.linspace(0, 1, 100)
            for r in r_rng:
                comp = [diff[0]*(r**z) for z in range(len(resid))]
                try:
                    if (diff <= comp).all():
                        break
                except RuntimeWarning:
                    r = 1
                    break
            # Update i-th repetition for j-th theta
            avg_r[i,j] = r

    avg_r = np.average(avg_r, axis=0)
    np.save('results/avg_r', avg_r)

    # Test the difference in performances for QR and QR*
    Y = [np.random.randn(m) for i in range(MAX_REP)]
    num_results = run_experiment(num_qr, Y, 'Numpy QR', nw)
    std_results = run_experiment(std_qr, Y, 'QR', nw)
    mod_results = run_experiment(mod_qr, Y, 'QR*', nw)
    np.save('results/num_results', num_results)
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
        w_list = opt.optimize(f,g,Q,conv_array=True, verbose=True)
        _resid = np.array([f(w) for w in w_list])
        resid[i, 0, :len(_resid)] = _resid

        # LBFGS Identity
        opt = LBFGS(w, gw, init='identity')
        w_list = opt.optimize(f,g,Q,conv_array=True, verbose=True)
        _resid = np.array([f(w) for w in w_list])
        resid[i, 1, :len(_resid)] = _resid

        # BFGS
        opt    = BFGS(w, gw, np.eye(n))
        w_list = opt.optimize(f,g,Q,conv_array=True, verbose=True)
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
