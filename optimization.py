import numpy as np
from utils import load_dataset, lls_functions

class Optimizer:
    """
    Class representing a generic a optimizer for the
    linear least squares problem. Inheriting objects
    should redefine update and get_direction methods

    Attributes
    ----------
    w : ndarray
        Current candidate point
    n : int
        Shape of the candidate point
    gw : ndarray
        Gradient of the candidate point
    name : str
        Name of the optimizer

    Methods
    -------
    update(w, gw)
        Updates the current candidate point

    get_direction()
        Returns the direction to search for the next candidate

    optimize(f, g, Q, eps, max_step=2048, verbose=False, conv_array=False)
        Solves the LLS problem and returns either the candidate solution
        or the list of all the encountered candidates

    __str__()
        Returns the name of the optimizer
    """
    def __init__(self, w, gw, name):
        self.w  = w
        self.n  = w.shape[0]
        self.gw = gw
        self.name = name

    def update(self, w, gw):
        self.w  = w
        self.gw = gw
        return w, gw

    def get_direction(self):
        raise NotImplementedError

    def optimize(self, f, g, Q, eps=1e-6, max_step=2048, verbose=False, conv_array=False):
        """
        Function that solves the LLS problem using the optimizer

        Parameters
        ----------
        f : function
            Objective function
        g : function
            Gradient function
        Q : ndarray
            Hessian of the objective function
        eps : float, optional
            Threshold for the gradient stopping condition
        max_step : int, optional
            Maximum number of steps to optimize
        verbose : bool, optional
            Flag to print the state of the optimizer during each iteration
        conv_array : bool, optional
            Flag to return the candidate solution and the number of steps
            or the whole sequence of candidates

        Returns
        -------
        w : ndarray
            The candidate solution
        k : int
            Number of steps
        w_list : list
            List of candidates
        """
        # Verbose
        if verbose:
            print(self)
            print('', 'Steps', 'α', '\t|∇f(w)|', '\tf(w)',sep='\t')

        # Initial candidate
        w   = self.w
        gw  = self.gw
        ngw = np.linalg.norm(self.gw)

        # List of candidates
        w_list = [w]

        # Main loop
        k = 0
        while ngw > eps and k < max_step:
            # Compute search direction
            d  = self.get_direction()

            # Line search
            alpha = - (gw.T @ d)/(d.T @ Q @ d)

            # Next candidate
            next_w  = w + alpha * d
            next_gw = g(next_w)

            # Update candidate
            w, gw = self.update(next_w, next_gw)
            ngw   = np.linalg.norm(gw)
            w_list.append(w)

            # Log
            if verbose:
                print('', k,
                    np.format_float_scientific(alpha, precision=4),
                    np.format_float_scientific(ngw, precision=4),
                    np.format_float_scientific(f(w), precision=4),sep='\t')

            # Update step counter
            k += 1

        if conv_array:
            return w_list
        else:
            return w, k

    def __str__(self):
        return self.name

class Gradient(Optimizer):
    """
    Steepest Gradient Descent
    """
    def __init__(self, w, gw, name='Gradient'):
        super().__init__(w, gw, name)

    def get_direction(self):
        return - self.gw

class Newton(Optimizer):
    """
    Newton method

    Attributes
    ----------
    H : ndarray
        Inverse of the Hessian matrix
    """
    def __init__(self, w, gw, H, name='Newton'):
        super().__init__(w, gw, name)
        self.H  = H

    def get_direction(self):
        return - self.H @ self.gw

class BFGS(Newton):
    """
    Broyden-Fletcher-Goldfarb-Shanno method

    Attributes
    ----------
    H : ndarray
        Inverse of the Hessian matrix approximated by the BFGS method.
        A possible initialization for the method is the identity matrix.
    """
    def __init__(self, w, gw, H, name='BFGS'):
        super().__init__(w, gw, H, name)
        self.H  = H

    def update(self, w, gw):
        s   = np.reshape(w - self.w, (self.n,1))
        Y   = np.reshape(gw - self.gw, (self.n,1))
        rho = 1 / ( Y.T @ s )
        self.H = (np.eye(self.n) - rho * s @ Y.T ) @ self.H @ ( np.eye(self.n) - rho * Y @ s.T ) + rho * s @ s.T
        self.w = w
        self.gw = gw
        return w, gw

class LBFGS(Optimizer):
    """
    Limited-memory BFGS method

    Attributes
    ----------
    t : int, optional
        Size of the memory
    init : str, optional
        Initialization of the optimizer, this can be either 'gamma', 'identity' or 'random'
    perturbate : float, optional
        If not none, the initialization of the matrix is summed with a random normal
        vector with zero variance and average equal to perturbate
    """
    def __init__(self, w, gw, t=8, init='gamma', perturbate=None, name='L-BFGS'):
        super().__init__(w, gw, name)
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

    opt    = Gradient(w, gw)
    w_c, s = opt.optimize(f,g,Q,max_step=32,verbose=True)
    print_mem(opt)

    opt    = Newton(w, gw, H)
    w_c, s = opt.optimize(f,g,Q,verbose=True)
    print_mem(opt)

    opt    = BFGS(w, gw, np.eye(n))
    w_c, s = opt.optimize(f,g,Q,verbose=True)
    print_mem(opt)

    opt    = LBFGS(w, gw, name='L-BFGS, γ init')
    w_c, s = opt.optimize(f,g,Q,verbose=True)
    print_mem(opt)

    opt    = LBFGS(w, gw, init='identity', name='L-BFGS, I init')
    w_c, s = opt.optimize(f,g,Q,verbose=True)
    print_mem(opt)
