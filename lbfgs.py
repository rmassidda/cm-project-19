import time
import numpy as np

# Helpers
def log(v,name):
    v = np.array(v)
    print(name+':', v.shape, v if max(v.shape, default=0)<=1 else '', sep='\t')

# def crono(f,args):
#     start = time.time()
#     res   = f(**args)
#     end   = time.time()
#     return res, end-time

# Construct the input matrix
def load_dataset():
    X = np.genfromtxt('dataset/ML-CUP19-TR.csv', delimiter=',', usecols=range(1,21))
    I = np.eye(X.shape[0])
    X_hat = np.concatenate((X.T, I))
    log(X,'X')
    log(I,'I')
    log(X_hat,'\hat X')
    return X, X_hat

X, X_hat = load_dataset()
m, n     = X_hat.shape

# Random y vector
y = np.random.rand(m)
log(y, 'y')

def evaluate_solution(w,steps,time=None):
    r = np.linalg.norm(X_hat @ w - y) ** 2
    print('', 'Steps:',steps,sep='\t')
    print('', 'Residual:',r,sep='\t')

# Numpy solution
print("Numpy solution")
w, r, _, _ = np.linalg.lstsq(X_hat, y, rcond=None) 
evaluate_solution(w,0)

# Gradient
def g(w):
    return X_hat.T @ ( X_hat @ w - y )

# Hessian
H = X @ X.T + np.eye(n) # Explicit
B = np.linalg.inv(H)    # Inverse

# Netwon method
print("Newton method")
eps = 1e-3
w   = np.random.rand(n)
k   = 0
gw  = g(w)
ngw = np.linalg.norm(gw)
while ngw > eps:
    d     = - B @ gw
    alpha = - (gw.T @ d)/(d.T @ H @ d)
    w   = w + alpha * d
    k  += 1
    gw  = g(w)
    ngw = np.linalg.norm(gw)
    # print('Step:',k,sep='\t')
    # log(ngw,'|∇f(w)|')
evaluate_solution(w,k)

# L-BFGS
print("L-BFGS method")
eps = 1e-3
w   = np.random.rand(n)
s   = 0
gw  = g(w)
ngw = np.linalg.norm(gw)
k   = 0
t   = 16
p   = 0
S   = np.zeros((t,n))
Y   = np.zeros((t,n))
gamma = 1
while ngw > eps and k < 256:
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
    p    = ( p + 1 ) % t

    # Update candidate
    gw  = next_gw
    w   = next_w
    ngw = np.linalg.norm(gw)

    # Log
    # print('Step:',k,sep='\t')
    # log(ngw,'|∇f(w)|')

    # Update step counter
    k += 1

evaluate_solution(w,k)
