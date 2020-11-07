import numpy as np
from matplotlib import pyplot as plt

# init
fig = plt.figure()
# fig.suptitle('LBFGS convergence')
plt.yscale('log')
plt.xlabel('Steps')
# lbfgs convergence
lbfgs_conv = np.load('results/lbfgs-convergence.npy')
x = range(len(lbfgs_conv))
plt.plot(x, lbfgs_conv, label=r'LBFGS $\gamma$')
# lbfgs convergence
lbfgs_conv = np.load('results/lbfgs-convergence-identity.npy')
x = range(len(lbfgs_conv))
plt.plot(x, lbfgs_conv, label=r'LBFGS $I$')
# lbfgs convergence
lbfgs_conv = np.load('results/bfgs-convergence.npy')
x = range(len(lbfgs_conv))
plt.plot(x, lbfgs_conv, label=r'BFGS $I$')
# show
plt.legend()
plt.show()

# solver, granularity, (t, r, s)
theta_defaults = np.load('results/theta_defaults.npy')
def_names      = ['LLS Numpy', 'Newton', 'LBFGS', 'QR*']
titles         = ['Time', 'Residual', 'Steps']
x = np.linspace(0, np.pi/2, theta_defaults.shape[1])
for i, title in enumerate(titles):
    fig = plt.figure()
    plt.xlabel(r'$\theta$')
    # fig.suptitle(title)
    if title == 'Residual' or title == 'Steps':
        plt.yscale('log')
    for j, name in enumerate(def_names):
        if ('QR' in name or 'LLS' in name) and title=='Steps':
            continue
        plt.plot(x, theta_defaults[j, :, i], label=name)
    plt.legend()
    plt.show()

# Narrow
def_names      = ['LLS Numpy', 'Newton', 'LBFGS', 'QR*']
titles         = ['Time', 'Residual', 'Steps']
x = np.linspace(0, np.pi/2, theta_defaults.shape[1])
a = int(len(x) / 4)
b = a * 3
for i, title in enumerate(titles):
    fig = plt.figure()
    plt.xlabel(r'$\theta$')
    # fig.suptitle(title)
    if title == 'Residual' or title == 'Steps':
        plt.yscale('log')
    for j, name in enumerate(def_names):
        if ('QR' in name or 'LLS' in name) and title=='Steps':
            continue
        plt.plot(x[a:b], theta_defaults[j, a:b, i], label=name)
    plt.legend()
    plt.show()

# conditioning
X_cond = np.load('results/theta_Xcond.npy')
y_cond = np.load('results/theta_ycond.npy')
fig = plt.figure()
# fig.suptitle('Conditioning')
plt.yscale('log')
plt.xlabel(r'$\theta$')
plt.plot(x, X_cond, label=r'$\kappa_{rel,\hat{X}\to w}$')
plt.plot(x, y_cond, label=r'$\kappa_{rel,y \to w}$')
plt.legend(fontsize="x-large")
plt.show()

# lbfgs memory
t_lbfgs = np.load('results/t_lbfgs.npy')
t_rng   = np.load('results/t_rng.npy')
titles  = ['time', 'residual', 'steps']
for i, title in enumerate(titles):
    fig = plt.figure()
    plt.xlabel(r'$t$')
    plt.ylabel(title)
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
