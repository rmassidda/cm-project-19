from utils import load_dataset, lls_functions, theta_angled
from optimization import LBFGS, BFGS, Newton, optimize
from numerical import qr, modified_qr, q1, back_substitution
import numpy as np
import time

def optimization_solver(y, method, params):
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
    duration = end - start
    residual = f(w_c)
    steps    = s
    return duration, residual, steps

def numerical_solver(y, qr_factorization):
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
    duration = end - start
    residual = f(w_c)
    steps    = s
    return duration, residual, steps

def numpy_solver(y):
    # Initial values
    f, _, _ = lls_functions(X_hat, X, y)
    s       = 1

    # Solve the LLS problem
    start = time.time()
    w_c   = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    end   = time.time()

    # Log results
    duration = end - start
    residual = f(w_c)
    steps    = s
    return duration, residual, steps

def run_experiment(f, Y, method):
    log = np.zeros((MAX_EXP, 3))
    for i, y in enumerate(Y):
        log[i, :] = f(y)
    print(method, *["%.2f" % np.average(log[:,j]) for j in range(3)],sep='\t')

if __name__ == '__main__':
    # Data loading
    X, X_hat = load_dataset()
    m, n     = X_hat.shape

    # Number of experiments
    MAX_EXP  = 2
    Y = [np.random.rand(m) for _ in range(MAX_EXP)]

    exp = lambda y: numpy_solver(y)
    run_experiment(exp, Y, 'LLS Numpy')

    exp = lambda y: optimization_solver(y, Newton, {})
    run_experiment(exp, Y, 'Newton')

    exp = lambda y: optimization_solver(y, BFGS, {'H': np.eye(n)})
    run_experiment(exp, Y, 'BFGS')

    exp = lambda y: optimization_solver(y, LBFGS, {})
    run_experiment(exp, Y, 'LBFGS')

    exp = lambda y: numerical_solver(y, lambda A: modified_qr(A, m-n+1))
    run_experiment(exp, Y, 'QR*')

    # Evaluate different memory for LBFGS
    rng      = range(1, n, int(n/10))
    methods  = ['t'+str(i) for i in rng]
    params   = [{'t': i} for i in rng]
    for method, p in zip(methods,params):
        exp = lambda y: optimization_solver(y, LBFGS, p)
        run_experiment(exp, Y, 'LBFGS '+method)

    # Evaluate different initialization for LBFGS
    rng      = range(-5, 5, 2)
    methods  = ['γ', *['γ~1e'+str(i) for i in rng]]
    methods += ['I', *['I~1e'+str(i) for i in rng]]
    params   = [{'init': 'gamma'}, *[{'init': 'gamma', 'perturbate': 10**i} for i in rng]]
    params  += [{'init': 'identity'}, *[{'init': 'identity', 'perturbate': 10**i} for i in rng]]
    for method, p in zip(methods,params):
        exp = lambda y: optimization_solver(y, LBFGS, p)
        run_experiment(exp, Y, 'LBFGS '+method)

    # Evaluate different initialization for BFGS
    rng      = range(-5, 5, 2)
    methods  = ['H', *['H~1e'+str(i) for i in rng]]
    methods += ['I', *['I~1e'+str(i) for i in rng]]
    params   = [('H',0), *[('H',10**i) for i in rng]]
    params  += [('I',0), *[('I',10**i) for i in rng]]
    def perturbate_H(y, eps, init):
        _, _, Q = lls_functions(X_hat, X, y)
        H = np.linalg.inv(Q)
        if init == 'H':
            return {'H': H + np.random.normal(0,eps,n)}
        else:
            return {'H': np.eye(n) + np.random.normal(0,eps,n)}
    for method, (init, eps) in zip(methods,params):
        exp = lambda y: optimization_solver(y, BFGS, perturbate_H(y, eps, init))
        run_experiment(exp, Y, 'BFGS '+method)

#     # Test: evaluate different θ
#     n_int    = 4
#     methods  = ['π/'+"%.2f"%(2*n_int/i) if i != 0 else '0' for i in range(0,n_int+1)]
#     params   = [i*np.pi/(2*n_int) for i in range(0,n_int+1)]
#     log = {k1: {k2: np.zeros(MAX_EXP) for k2 in metrics} for k1 in methods}
#     for k1, theta in zip(methods,params):
#         _, y = theta_angled(X_hat, theta)
#         for i in range(MAX_EXP):
#             solver(y, LBFGS, {}, i, log[k1])
#     for k1 in methods:
#         print(k1, *["%.2f" % np.average(log[k1][k2]) for k2 in metrics],sep='\t')
