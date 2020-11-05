from utils import load_dataset, lls_functions
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

metrics = ['duration','residual','steps']

if __name__ == '__main__':
    # Data loading
    X, X_hat = load_dataset()
    m, n     = X_hat.shape

    # Number of experiments
    MAX_EXP  = 4

    # Test: evaluate performances on a random vector
    methods = ['Newton', 'BFGS', 'LBFGS']
    log = {k1: {k2: np.zeros(MAX_EXP) for k2 in metrics} for k1 in methods}
    for i in range(MAX_EXP):
        y = np.random.rand(m)
        solver(y, Newton, {}, i, log['Newton'])
        solver(y, BFGS, {'H': np.eye(n)}, i, log['BFGS'])
        solver(y, LBFGS, {}, i, log['LBFGS'])
    for k1 in methods:
        print(k1, *["%.2f" % np.average(log[k1][k2]) for k2 in metrics],sep='\t')

    # Test: evaluate different initialization for LBFGS
    rng      = range(0, 10, 2)
    methods  = ['γ', *['γ~1e-'+str(i) for i in rng]]
    methods += ['I', *['I~1e-'+str(i) for i in rng]]
    params   = [{'init': 'gamma'}, *[{'init': 'gamma', 'perturbate': 10**-i} for i in rng]]
    params  += [{'init': 'identity'}, *[{'init': 'identity', 'perturbate': 10**-i} for i in rng]]
    log = {k1: {k2: np.zeros(MAX_EXP) for k2 in metrics} for k1 in methods}
    for i in range(MAX_EXP):
        y = np.random.rand(m)
        for k1, p in zip(methods,params):
            solver(y, LBFGS, p, i, log[k1])
    for k1 in methods:
        print(k1, *["%.2f" % np.average(log[k1][k2]) for k2 in metrics],sep='\t')

    # Numerical experiments
    np.set_printoptions(precision=20)

    # Random y vector
    y = np.random.rand(m)
    print('Random y vector:')
    print(y)
    print()

    # Solve the least squares problem (numpy)
    start = time.time()
    w = np.linalg.lstsq(X_hat, y, rcond=None)
    end = time.time()
    print('Solution to the ls found in {:.2f} ms:'.format(end-start))
    print(w[0])
    print()

    start = time.time()
    R, vects = modified_qr(X_hat, m-n+1)
    end = time.time()
    print("Modified qr,", end-start, "ms:\n", R)
    print()

    start = time.time()
    Q1 = q1(vects, m)
    end = time.time()
    print("Q1 reconstruction:", end-start, "ms:\n", Q1.shape)
    print()

    Qnp, Rnp = np.linalg.qr(X_hat, mode='complete')
    print('diy Q1 vs numpy Q1 (norm of difference)')
    print(np.linalg.norm(Qnp[:,0:n] - Q1, ord='fro'))
    print()
    print('diy R vs numpy R (norm of difference):')
    print(np.linalg.norm(R[:,0:n] - Rnp[:,0:n], ord='fro'))
    print()

    start = time.time()
    c = np.dot(Q1.T, y)
    x = back_substitution(R[:n, :], c)
    end = time.time()
    print('diy solution to the ls found in {:.2f} ms'.format(end-start))
    print()
    print('diy solution vs numpy solution (norm of difference):')
    print(np.linalg.norm(x-w[0]))
    print()
