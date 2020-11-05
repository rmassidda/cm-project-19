from utils import load_dataset, lls_functions, theta_angled
from optimization import LBFGS, BFGS, Newton, optimize
from numerical import qr, modified_qr, q1, back_substitution
import numpy as np
import time

def solver(y, method, params, i, log):
    # Initial values
    f, g, Q = lls_functions(X_hat, X, y)
    w       = np.random.rand(n)
    gw      = g(w)

    # Fill-up missing parameters
    params['w']  = w
    params['gw'] = gw
    if 'H' not in params:
        params['H'] = np.linalg.inv(Q)

    # Solve the LLS problem
    start  = time.time()
    opt    = method(**params)
    w_c, s = optimize(f,g,Q,opt)
    end    = time.time()

    # Log results
    log['duration'][i] = end - start
    log['residual'][i] = f(w_c)
    log['steps'][i]    = s

def direct_solver(y, qr_factorization, i, log):
    # Initial values
    f, _, _ = lls_functions(X_hat, X, y)
    s       = 1

    # Solve the LLS problem
    start    = time.time()
    R, vects = qr_factorization(X_hat)
    Q1       = q1(vects, m)
    c        = np.dot(Q1.T, y)
    w_c      = back_substitution(R[:n, :], c)
    end      = time.time()

    # Log results
    log['duration'][i] = end - start
    log['residual'][i] = f(w_c)
    log['steps'][i]    = s

metrics = ['duration','residual','steps']

if __name__ == '__main__':
    # Data loading
    X, X_hat = load_dataset()
    m, n     = X_hat.shape

    # Number of experiments
    MAX_EXP  = 4

    # Test: evaluate performances on a random vector
    methods = ['Newton', 'BFGS', 'LBFGS', 'QR*']
    log = {k1: {k2: np.zeros(MAX_EXP) for k2 in metrics} for k1 in methods}
    for i in range(MAX_EXP):
        y = np.random.rand(m)
        solver(y, Newton, {}, i, log['Newton'])
        solver(y, BFGS, {'H': np.eye(n)}, i, log['BFGS'])
        solver(y, LBFGS, {}, i, log['LBFGS'])
        direct_solver(y, lambda A: modified_qr(A, m-n+1), i, log['QR*'])
    for k1 in methods:
        print(k1, *["%.2f" % np.average(log[k1][k2]) for k2 in metrics],sep='\t')

    # Test: evaluate different initialization for LBFGS
    rng      = range(-5, 5, 2)
    methods  = ['γ', *['γ~1e'+str(i) for i in rng]]
    methods += ['I', *['I~1e'+str(i) for i in rng]]
    params   = [{'init': 'gamma'}, *[{'init': 'gamma', 'perturbate': 10**i} for i in rng]]
    params  += [{'init': 'identity'}, *[{'init': 'identity', 'perturbate': 10**i} for i in rng]]
    log = {k1: {k2: np.zeros(MAX_EXP) for k2 in metrics} for k1 in methods}
    for i in range(MAX_EXP):
        y = np.random.rand(m)
        for k1, p in zip(methods,params):
            solver(y, LBFGS, p, i, log[k1])
    for k1 in methods:
        print(k1, *["%.2f" % np.average(log[k1][k2]) for k2 in metrics],sep='\t')

    # Test: evaluate different initialization for BFGS
    #       the matrix may be initialized using the identity
    #       or the actual inverse of the Hessian. When this
    #       is used, then the behaviour is the same as the
    #       Newton method. Hereby both approaches are tested
    #       with various perturbations.
    rng      = range(-5, 5, 2)
    methods  = ['H', *['H~1e'+str(i) for i in rng]]
    methods += ['I', *['I~1e'+str(i) for i in rng]]
    params   = [('H',0), *[('H',10**i) for i in rng]]
    params  += [('I',0), *[('I',10**i) for i in rng]]
    log = {k1: {k2: np.zeros(MAX_EXP) for k2 in metrics} for k1 in methods}
    for i in range(MAX_EXP):
        y = np.random.rand(m)
        _, _, Q = lls_functions(X_hat, X, y)
        H = np.linalg.inv(Q)
        for k1, (init, eps) in zip(methods,params):
            if init == 'H':
                e = {'H': H + np.random.normal(0,eps,n)}
            else:
                e = {'H': np.eye(n) + np.random.normal(0,eps,n)}
            solver(y, BFGS, e, i, log[k1])
    for k1 in methods:
        print(k1, *["%.2f" % np.average(log[k1][k2]) for k2 in metrics],sep='\t')

    # Test: evaluate different θ
    n_int    = 4
    methods  = ['π/'+"%.2f"%(2*n_int/i) if i != 0 else '0' for i in range(0,n_int+1)]
    params   = [i*np.pi/(2*n_int) for i in range(0,n_int+1)]
    log = {k1: {k2: np.zeros(MAX_EXP) for k2 in metrics} for k1 in methods}
    for k1, theta in zip(methods,params):
        _, y = theta_angled(X_hat, theta)
        for i in range(MAX_EXP):
            solver(y, LBFGS, {}, i, log[k1])
    for k1 in methods:
        print(k1, *["%.2f" % np.average(log[k1][k2]) for k2 in metrics],sep='\t')
