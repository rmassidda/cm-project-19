import numpy as np

def load_dataset():
    X = np.genfromtxt('dataset/ML-CUP19-TR.csv', delimiter=',', usecols=range(1,21))
    I = np.eye(X.shape[0])
    X_hat = np.concatenate((X.T, I))
    return X, X_hat

def lls_functions(X_hat, X, y):
    _, n = X_hat.shape
    f    = lambda w : np.linalg.norm(X_hat @ w - y) ** 2
    g    = lambda w : X_hat.T @ ( X_hat @ w - y )
    Q    = X @ X.T + np.eye(n)
    return f, g, Q

def theta_angled(A, theta):
    m, n = A.shape
    q, r = np.linalg.qr(A, 'complete')
    q2 = q[:,n:]
    v = q2.T[0,:] # Any row of q2 is perpendicular to A image
    x = np.random.rand(n) # Get any point in the image
    Ax = A@x
    desired_norm = np.linalg.norm(Ax) * np.tan(theta)
    v = v * desired_norm / np.linalg.norm(v)
    b = Ax + v
    return x, b
