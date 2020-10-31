from utils import load_dataset, lls_functions
from optimization import LBFGS, BFGS, Newton, optimize
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
methods = ['Newton', 'BFGS', 'LBFGS']
classes = [Newton, BFGS, LBFGS]

if __name__ == '__main__':
    # Data loading
    X, X_hat = load_dataset()
    m, n     = X_hat.shape

    # Number of experiments
    MAX_EXP  = 4

    # Test: evaluate performances on a random vector
    log = {k1: {k2: np.zeros(MAX_EXP) for k2 in metrics} for k1 in methods}
    for i in range(MAX_EXP):
        y = np.random.rand(m)
        solver(y, Newton, {}, i, log['Newton'])
        solver(y, BFGS, {'H': np.eye(n)}, i, log['BFGS'])
        solver(y, LBFGS, {}, i, log['LBFGS'])
    for k1 in methods:
        print(k1, *["%.2f" % np.average(log[k1][k2]) for k2 in metrics],sep='\t')
