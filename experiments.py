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

def __optimization_solver(y, method, params):
    # Initial values
    f, g, Q = lls_functions(X_hat, X, y)
    w       = np.random.rand(n)
    gw      = g(w)

    # Fill-up missing parameters
    params['w']  = w
    params['gw'] = gw
    if 'H' not in params:
        params['H'] = np.linalg.inv(Q)

    opt    = method(**params)
    w_c, s = optimize(f,g,Q,opt)

    return w_c, s

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

    # Input analysis
    KX = np.linalg.cond(X_hat)
    print('Condition number', KX)

    # Number of experiments
    MAX_EXP = 15

    y_generator = []
    y_generator.append([np.random.rand(m) for _ in range(MAX_EXP)])

    n_theta  = 4
    ls_theta = [i*np.pi/(2*n_theta) for i in range(n_theta)]
    get_y    = lambda theta: theta_angled(X_hat, theta)[1]
    for theta in ls_theta:
        y_generator.append([get_y(theta) for _ in range(MAX_EXP)])

    theta = np.zeros(MAX_EXP)
    up_X = np.zeros(MAX_EXP)
    up_y = np.zeros(MAX_EXP)
    for c, Y in enumerate(y_generator):
        # Study conditioning
        for i, y in enumerate(Y):
            # Ground truth
            w = __optimization_solver(y, Newton, {})[0]
            # Compute angle
            costheta = np.linalg.norm(X_hat@w)/np.linalg.norm(y)
            costheta = max(-1, min(costheta, 1)) # Force cos in (-1,1)
            theta[i] = np.arccos(costheta)

        # Compute conditioning upper bounds
        up_X = KX + KX**2 * np.tan(theta)
        up_y = KX / costheta

        print('Setup n.',c)
        print('Theta', np.average(theta), np.sqrt(np.var(theta)))
        print('K_rel,y', np.average(up_y), np.sqrt(np.var(up_y)))
        print('K_rel,X', np.average(up_X), np.sqrt(np.var(up_X)))

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
