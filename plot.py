import numpy as np
from matplotlib import pyplot as plt

# Load results of the experiments from file
avg_r       = np.load('results/avg_r.npy')
num_results = np.load('results/num_results.npy')
mod_results = np.load('results/mod_results.npy')
std_results = np.load('results/std_results.npy')
resid       = np.load('results/initializers.npy')
theta_def   = np.load('results/theta_defaults.npy')
X_cond      = np.load('results/theta_Xcond.npy')
y_cond      = np.load('results/theta_ycond.npy')
t_rng       = np.load('results/t_rng.npy')
t_lbfgs     = np.load('results/t_lbfgs.npy')

#
# Average convergence rate for increasing \theta
#
x = np.linspace(0, np.pi/2, len(avg_r))
fig = plt.figure('r-convergence')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$r$-convergence')
plt.plot(x, avg_r)
plt.show()

#
# Conditioning of the problem for various \theta
#
x = np.linspace(0, np.pi/2, X_cond.shape[0])
fig = plt.figure('conditioning')
plt.yscale('log')
plt.xlabel(r'$\theta$')
plt.plot(x, X_cond, label=r'$\kappa_{rel,\hat{X}\to w}$')
plt.plot(x, y_cond, label=r'$\kappa_{rel,y \to w}$')
plt.legend(fontsize="x-large")
plt.show()

#
# Average residual, time and steps for different methods and various \theta
#
def_names      = ['LLS Numpy', 'Newton', 'LBFGS', 'Numpy QR', 'QR*']
titles         = ['Time (s)', 'Residual', 'Steps']
fname          = ['time', 'residual', 'step']
x = np.linspace(0, np.pi/2, theta_def.shape[1])
for i, (title, f) in enumerate(zip(titles,fname)):
    fig = plt.figure('theta_'+f)
    plt.xlabel(r'$\theta$')
    plt.ylabel(title)
    if title == 'Residual' or title == 'Steps':
        plt.yscale('log')
    for j, name in enumerate(def_names):
        if ('QR' in name or 'LLS' in name) and title=='Steps':
            continue
        if 'Newton' in name:
            continue
        plt.plot(x, theta_def[j, :, i], label=name)
    plt.legend()
    plt.show()

#
# Average residual, time and steps for different methods and various \theta
# in a narrow interval
#
x = np.linspace(0, np.pi/2, theta_def.shape[1])
a = int(len(x) / 4)
b = a * 3
for i, (title, f) in enumerate(zip(titles,fname)):
    fig = plt.figure('theta_narrow_'+f)
    plt.xlabel(r'$\theta$')
    plt.ylabel(title)
    if title == 'Residual':
        plt.yscale('log')
    for j, name in enumerate(def_names):
        if ('QR' in name or 'LLS' in name) and title=='Steps':
            continue
        plt.plot(x[a:b], theta_def[j, a:b, i], label=name)
    plt.legend()
    plt.show()

#
# Graphical depiction of the r-convergence of the LBFGS method
# convergence (linscale e logscale)
#
limit = np.argwhere(resid[0] == -1)[0,0]
resid = resid[0, :limit]
f_opt = resid[-1]
diff  = resid - f_opt
r_rng = np.linspace(0, 1, 100)
for r in r_rng:
    comp = [diff[0]*(r**i) for i in range(limit)]
    if (diff <= comp).all():
        break
print('r-line', r)
scale = ['linear', 'log']
for s in scale:
    fig = plt.figure('LBFGS_r_'+s)
    plt.xlabel('Steps')
    plt.yscale(s)
    plt.plot(range(limit), diff, label=r'$f(w_i) - f(w^*)$')
    plt.plot(range(limit), comp, label=r'$r^i(f(w_0) - f(w^*))$')
    plt.legend()
    plt.show()

#
# Plot of the residual curve for the LBFGS method (linscale e logscale)
#
scale = ['linear', 'log']
for s in scale:
    fig = plt.figure('LBFGS_residual_'+s)
    plt.xlabel('Steps')
    plt.yscale(s)
    plt.plot(range(limit), resid)
    plt.show()

#
# Average residual, time and steps for different memory values in LBFGS
#
titles  = ['Time (s)', 'residual', 'steps']
for i, (title, f) in enumerate(zip(titles,fname)):
    fig = plt.figure('memory_'+f)
    plt.xlabel(r'$t$')
    plt.ylabel(title)
    plt.plot(t_rng, t_lbfgs[:, i])
    plt.show()

try:
    from tabulate import tabulate
    no_tab = False
except ModuleNotFoundError:
    no_tab = True

#
# Table avg for the intervals in the narrow
#
x = np.linspace(0, np.pi/2, theta_def.shape[1])
a = int(len(x) / 4)
b = a * 3
models = np.array(['LLS Numpy', 'Newton', 'LBFGS', 'QR Numpy', 'QR*']).reshape((5,1))
heads  = ['Model', 'Time', 'Residual', 'Steps']
avg_metrics = np.average(theta_def[:, a:b, :], axis=1)
avg_metrics = np.concatenate((models, avg_metrics), axis=1)
if no_tab:
    print(heads)
    print(avg_metrics)
else:
    print(tabulate(avg_metrics, headers=heads, tablefmt='latex'), end='\n\n')

#
# Comparison of QR and QR*
#
num_results = np.concatenate((['Numpy QR'], np.average(num_results, axis=0))).reshape((1,4))
mod_results = np.concatenate((['QR*'], np.average(mod_results, axis=0))).reshape((1,4))
std_results = np.concatenate((['QR'], np.average(std_results, axis=0))).reshape((1,4))
qr_compare  = np.concatenate((num_results, mod_results, std_results), axis=0)
heads  = ['Model', 'Time', 'Residual', 'Steps']
if no_tab:
    print(heads)
    print(qr_compare)
else:
    print(tabulate(qr_compare, headers=heads, tablefmt='latex'), end='\n\n')
