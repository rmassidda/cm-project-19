import numpy as np
import time

seed = 1
np.random.seed(seed=seed)
np.set_printoptions(precision=20)

# Loading the dataset
x = np.genfromtxt('dataset/ML-CUP19-TR.csv', delimiter=',', usecols=range(1,21))

# Constructing the X_hat matrix
I = np.eye(x.shape[0])
X_hat = np.concatenate((x.T, I))
m, n = X_hat.shape

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

def householder_vector(x):
    e1 = np.zeros(x.shape[0])
    e1[0] = 1.0
    sign = 1.0 if x[0]==0 else np.sign(x[0])
    s = -sign*np.linalg.norm(x)
    v = x - s*e1
    v = v / np.linalg.norm(v)
    return v, s


def qr(A):
    hh_vectors = []
    m, n = A.shape
    R = np.array(A)
    for k in range(0, n):
        x = R[k:, k]
        if np.all(x==0):
            continue
        v, s = householder_vector(x)
        R[k, k] = s
        R[k+1:, k] = np.zeros(m-k-1)
        R[k:, k+1:] = R[k:, k+1:] - 2*np.outer(v, (np.dot(v, R[k:, k+1:])))
        hh_vectors.append(v)
    return R, hh_vectors


# l: dimension of householder vectors
def modified_qr(A, l):
    assert l > 0
    hh_vectors = []
    m, n = A.shape
    R = np.array(A)
    for k in range(0, n):
        x = R[k:k+l, k]
        if np.all(x==0):
            continue
        v, s = householder_vector(x)
        R[k, k] = s
        R[k+1:k+l, k] = np.zeros(l-1)
        R[k:k+l, k+1:] = R[k:k+l, k+1:] - 2*np.outer(v, (np.dot(v, R[k:k+l, k+1:])))
        hh_vectors.append(v)
    return R, hh_vectors


def q1(hh_vects, m):
    n = len(hh_vects)

    M = np.zeros((m-n+1,n))
    M[0, n-1] = 1.0
    v = hh_vects[n-1]
    v = 2*v[0]*v
    M[:, n-1] = M[:, n-1] - v

    Q1 = np.block([
        [np.eye(n-1), np.zeros((n-1, 1))],
        [M]
    ])

    for i in range(n-1, 0, -1):
        v = hh_vects[i-1]
        Q1[i-1:i+m-n, :] = Q1[i-1:i+m-n, :] - 2*np.outer(v, (np.dot(v, Q1[i-1:i+m-n, :])))

    return Q1


def back_substitution(A, b):
    n = A.shape[1]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = b[i]
        for j in range(n-1, i, -1):
            s -= x[j]*A[i,j]
        x[i] = s/A[i,i]
    return x


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