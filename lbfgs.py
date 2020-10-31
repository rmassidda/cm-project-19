import time
import numpy as np

# Helpers
def log(v,name):
    v = np.array(v)
    print(name+':', v.shape, v if max(v.shape, default=0)<=1 else '', sep='\t')

# Construct the input matrix
def load_dataset():
    X = np.genfromtxt('dataset/ML-CUP19-TR.csv', delimiter=',', usecols=range(1,21))
    I = np.eye(X.shape[0])
    X_hat = np.concatenate((X.T, I))
    log(X,'X')
    log(I,'I')
    log(X_hat,'\hat X')
    return X, X_hat

# Initial data
X, X_hat = load_dataset()
m, n     = X_hat.shape
y        = np.random.rand(m)
f        = lambda w : np.linalg.norm(X_hat @ w - y) ** 2

# Numpy solution
# print("Numpy solution")
# w, r, _, _ = np.linalg.lstsq(X_hat, y, rcond=None)
# evaluate_solution(w,0)

# Gradient
def g(w):
    return X_hat.T @ ( X_hat @ w - y )

# Hessian
H = X @ X.T + np.eye(n) # Explicit
B = np.linalg.inv(H)    # Inverse

MAX_STEP = 256

# Newton method
print("Newton method")
eps = 1e-3
w   = np.random.rand(n)
gw  = g(w)
ngw = np.linalg.norm(gw)
k   = 0
print('', 'Steps', 'α', '|∇f(w)|',sep='\t')
while ngw > eps and k < MAX_STEP:
    # Direction
    d     = - B @ gw
    alpha = - (gw.T @ d)/(d.T @ H @ d)

    # Next candidate
    next_w  = w + alpha * d
    next_gw = g(next_w)

    # Update candidate
    gw  = next_gw
    w   = next_w
    ngw = np.linalg.norm(gw)

    # Log
    print('', k, "%.2f" % alpha, np.format_float_scientific(ngw, precision=4),sep='\t')

    # Update step counter
    k += 1

# L-BFGS
print("L-BFGS method")
eps = 1e-3
w   = np.random.rand(n)
s   = 0
gw  = g(w)
ngw = np.linalg.norm(gw)
k   = 0
t   = 8
p   = 0
S   = np.zeros((t,n))
Y   = np.zeros((t,n))
gamma = 1
print('', 'Steps', 'α', '|∇f(w)|',sep='\t')
while ngw > eps and k < MAX_STEP:
    H_0 = gamma * np.eye(n)

    # L-BFGS two-loop recursion
    q = gw
    for i in range(min(t,k)): # Recent to older
        cp    = ( p - 1 ) % t
        rho   = 1 / (Y[cp].T @ S[cp])
        alpha = rho * S[cp] @ q
        q     = q - alpha * Y[cp]
    r = H_0 @ q
    for i in range(min(t,k)): # Older to recent
        cp    = ( p - 1 ) % t
        rho   = 1 / (Y[cp].T @ S[cp])
        alpha = rho * S[cp] @ q
        beta  = rho * Y[cp] @ r
        r     = r + S[cp] * ( alpha - beta )

    # Direction
    d     = -r
    alpha = - (gw.T @ d)/(d.T @ H @ d)

    # Next candidate
    next_w  = w + alpha * d
    next_gw = g(next_w)

    # Update S and Y
    S[p]  = next_w - w
    Y[p]  = next_gw - gw
    gamma = (S[p] @ Y[p]) / (Y[p] @ Y[p])
    p     = ( p + 1 ) % t

    # Update candidate
    gw  = next_gw
    w   = next_w
    ngw = np.linalg.norm(gw)

    # Log
    print('', k, "%.2f" % alpha, np.format_float_scientific(ngw, precision=4),sep='\t')

    # Update step counter
    k += 1

# BFGS
print("BFGS method")
# Looping
eps = 1e-3
k   = 0
# Initial conditions
w   = np.random.rand(n)
gw  = g(w)
ngw = np.linalg.norm(gw)
Hk  = B + np.eye(n) * np.random.rand()
I   = np.eye(n)
print('', 'Steps', 'α', '|∇f(w)|',sep='\t')
while ngw > eps and k < MAX_STEP:
    # Compute search direction
    d  = - Hk @ gw

    # Line search
    alpha = - (gw.T @ d)/(d.T @ H @ d)

    # Next candidate
    next_w  = w + alpha * d
    next_gw = g(next_w)

    # Update S and Y
    s   = np.reshape(next_w - w, (n,1))
    Y   = np.reshape(next_gw - gw, (n,1))
    rho = 1 / ( Y.T @ s )

    # Update H
    Hk = (I - rho * s @ Y.T ) @ Hk @ ( I - rho * Y @ s.T ) + rho * s @ s.T

    # Update candidate
    gw  = next_gw
    w   = next_w
    ngw = np.linalg.norm(gw)

    # Log
    print('', k, "%.2f" % alpha, np.format_float_scientific(ngw, precision=4),sep='\t')

    # Update step counter
    k += 1
