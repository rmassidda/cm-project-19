import numpy as np
from matplotlib import pyplot as plt

# solver, granularity, (t, r, s)
theta_defaults = np.load('theta_defaults.npy')
def_names      = ['LLS Numpy', 'Newton', 'LBFGS', 'QR*']
titles         = ['Time', 'Residual', 'Steps']
x = np.linspace(0, np.pi/2, theta_defaults.shape[1])
for i, title in enumerate(titles):
    fig = plt.figure()
    fig.suptitle(title)
    for j, name in enumerate(def_names):
        plt.plot(x, theta_defaults[j, :, i], label=name)
    plt.legend()
    plt.show()

# conditioning
X_cond = np.load('theta_Xcond.npy')
y_cond = np.load('theta_ycond.npy')
fig = plt.figure()
fig.suptitle('Conditioning')
plt.plot(x, X_cond, label='Absolute')
plt.plot(x, y_cond, label='Relative')
plt.legend()
plt.show()

# lbfgs memory
t_lbfgs = np.load('t_lbfgs.npy')
t_rng   = np.load('t_rng.npy')
titles  = ['Time', 'Residual', 'Steps']
for i, title in enumerate(titles):
    fig = plt.figure()
    fig.suptitle(title)
    plt.plot(t_rng, t_lbfgs[:, i])
    plt.show()

# init, perturbation steps
init_lbfgs = np.load('init_lbfgs.npy')
try:
    from tabulate import tabulate
    print(tabulate(init_lbfgs, tablefmt='latex'))
except ModuleNotFoundError:
    print(init_lbfgs)
