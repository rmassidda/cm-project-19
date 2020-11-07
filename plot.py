import numpy as np
from matplotlib import pyplot as plt

# solver, granularity, (t, r, s)
theta_defaults = np.load('results/theta_defaults.npy')
def_names      = ['LLS Numpy', 'Newton', 'LBFGS', 'QR*']
titles         = ['Time', 'Residual', 'Steps']
x = np.linspace(0, np.pi/2, theta_defaults.shape[1])
for i, title in enumerate(titles):
    fig = plt.figure()
    fig.suptitle(title)
    if title == 'Residual':
        plt.yscale('log')
    for j, name in enumerate(def_names):
        plt.plot(x, theta_defaults[j, :, i], label=name)
    plt.legend()
    plt.show()

# conditioning
X_cond = np.load('results/theta_Xcond.npy')
y_cond = np.load('results/theta_ycond.npy')
fig = plt.figure()
fig.suptitle('Conditioning')
plt.yscale('log')
plt.plot(x, X_cond, label='\hat{X}') #NOTE LaTeX not rendered
plt.plot(x, y_cond, label='y')
plt.legend()
plt.show()

# lbfgs memory
t_lbfgs = np.load('results/t_lbfgs.npy')
t_rng   = np.load('results/t_rng.npy')
titles  = ['Time', 'Residual', 'Steps']
for i, title in enumerate(titles):
    fig = plt.figure()
    fig.suptitle(title)
    if title == 'Residual':
        plt.yscale('log')
    plt.plot(t_rng, t_lbfgs[:, i])
    plt.show()

# init, perturbation steps
init_lbfgs = np.load('results/init_lbfgs.npy')
try:
    from tabulate import tabulate
    print(tabulate(init_lbfgs, tablefmt='latex'))
except ModuleNotFoundError:
    print(init_lbfgs)
