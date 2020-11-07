import numpy as np
from utils import load_dataset, lls_functions

class Newton:
    def __init__(self, w, gw, H):
        self.w  = w
        self.n  = w.shape[0]
        self.gw = gw
        self.H  = H

    def get_direction(self):
        return - self.H @ self.gw

    def update(self, w, gw):
        return w, gw

    def __str__(self):
        return 'Newton'

class BFGS(Newton):
    def update(self, w, gw):
        s   = np.reshape(w - self.w, (self.n,1))
        Y   = np.reshape(gw - self.gw, (self.n,1))
        rho = 1 / ( Y.T @ s )
        self.H = (np.eye(self.n) - rho * s @ Y.T ) @ self.H @ ( np.eye(self.n) - rho * Y @ s.T ) + rho * s @ s.T
        self.w = w
        self.gw = gw
        return w, gw

    def __str__(self):
        return 'BFGS'

class LBFGS(Newton):
    def __init__(self, w, gw, t=8, init='gamma', perturbate=None):
        self.w  = w
        self.n  = w.shape[0]
        self.gw = gw
        self.k  = 0
        self.t  = t
        self.p  = 0
        self.I = np.ones(self.n)
        self.S = np.zeros((self.t,self.n))
        self.Y = np.zeros((self.t,self.n))
        self.gamma = 1
        self.init  = init
        self.perturbate = perturbate

    def get_direction(self):
        if self.init == 'gamma':
            init = self.gamma * self.I
        elif self.init == 'identity':
            init = self.I
        elif self.init == 'random':
            init = self.random.randn(self.n)

        if self.perturbate is not None:
            init += np.random.normal(0,self.perturbate,self.n)

        q = self.gw
        for i in range(min(self.t,self.k)): # Recent to older
            cp    = self.p - i -1
            si = self.S[cp]
            yi = self.Y[cp]
            rho   = 1 / (yi.T @ si)
            alpha = rho * si.T @ q
            q     = q - alpha * yi

        r = init * q
        for i in range(min(self.t,self.k)): # Older to recent
            cp    = self.p - i -1
            si = self.S[cp]
            yi = self.Y[cp]
            rho   = 1 / (yi.T @ si)
            beta  = rho * yi.T @ r
            alpha = rho * si.T @ q
            r     = r + si * ( alpha - beta )
        
        return - r

    def update(self, w, gw):
        self.S[self.p]  = w - self.w
        self.Y[self.p]  = gw - self.gw
        self.gamma = (self.S[self.p].T @ self.Y[self.p]) / (self.Y[self.p].T @ self.Y[self.p])
        self.p     = ( self.p + 1 ) % self.t
        self.k    += 1
        self.w = w
        self.gw = gw
        return w, gw

    def __str__(self):
        return 'L-BFGS'


"""Computes the solution of the least squares problem
   by using variations of the Newton method such as 
   BFGS and L-BFGS

Parameters
----------
f : R^n -> R
    The least squares problem objective function
g : R^n -> R^n
    The gradient function
Q : R^{n*n}
    The exact Hessian
opt : Optimizer
    The optimizer
eps : R
    The threshold over the gradient norm to stop the iterations
max_step : int
    The threshold over the number of steps to stop the iterations
verbose : bool
    Flag to print the state of the optimizer during each iteration

Returns
-------
w : R^n
    The candidate solution
"""

def optimize(f, g, Q, opt, eps=1e-6, max_step=2048, verbose=False, conv_array=False):
    # Verbose
    if verbose:
        print(opt)
        print('', 'Steps', 'α', '|∇f(w)|',sep='\t')

    # Initial candidate
    w   = opt.w
    gw  = opt.gw
    ngw = np.linalg.norm(opt.gw)

    # List of candidates
    w_list = [w]

    # Main loop
    k = 0
    while ngw > eps and k < max_step:
        # Compute search direction
        d  = opt.get_direction()

        # Line search
        alpha = - (gw.T @ d)/(d.T @ Q @ d)

        # Next candidate
        next_w  = w + alpha * d
        next_gw = g(next_w)

        # Update candidate
        w, gw = opt.update(next_w, next_gw)
        ngw   = np.linalg.norm(gw)
        w_list.append(w)

        # Log
        if verbose:
            print('', k, "%.2f" % alpha, np.format_float_scientific(ngw, precision=4),sep='\t')

        # Update step counter
        k += 1

    if conv_array:
        return w_list

    return w, k

if __name__ == "__main__":
    # Data loading
    X, X_hat = load_dataset()
    m, n     = X_hat.shape

    # Initial values
    y = np.random.randn(m)
    f, g, Q = lls_functions(X_hat, X, y)
    w       = np.random.randn(n)
    gw      = g(w)
    H       = np.linalg.inv(Q)

    def print_mem(o):
        print(o, 'memory usage')
        for attr, value in o.__dict__.items():
            try:
                print('', attr, value.shape, sep='\t')
            except AttributeError:
                pass
        print()

    opt    = Newton(w, gw, H)
    w_c, s = optimize(f,g,Q,opt,verbose=True)
    print_mem(opt)

    opt    = BFGS(w, gw, np.eye(n))
    w_c, s = optimize(f,g,Q,opt,verbose=True)
    print_mem(opt)

    opt    = LBFGS(w, gw)
    w_c, s = optimize(f,g,Q,opt,verbose=True)
    print_mem(opt)
