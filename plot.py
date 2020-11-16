import numpy as np
from matplotlib import pyplot as plt
try:
    from tabulate import tabulate
    no_tab = False
except ModuleNotFoundError:
    no_tab = True


# Load results of the experiments from file
nar_res     = np.load('results/nar_res.npy')
nar_opt_w   = np.load('results/nar_opt_w.npy')
opt_w       = np.load('results/opt_w.npy')
tilde_w     = np.load('results/tilde_w.npy')
X_cond_ub   = np.load('results/X_cond_ub.npy')
y_cond_ub   = np.load('results/y_cond_ub.npy')
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
# Narrow interval table
#
# ((MAX_REP, len(def_solvers), MAX_G, 3, n))
# ((MAX_REP, MAX_G, n))
def_names  = ['LLS Numpy', 'Newton', 'LBFGS', 'Numpy QR', 'QR*']
heads      = ['Model', 'Time (s)', 'Relative error', 'Steps']
table      = np.zeros((len(def_names),3))
for i, model in enumerate(def_names):
    results = nar_res[:,i,:,:,:]
    # Get the averages
    avg = results
    avg = np.average(avg,axis=0)   # Average over the replicas
    avg = np.average(avg, axis=0)  # Average over the thetas
    table[i,0] = avg[0][0]
    table[i,2] = avg[2][0]

    # Compute relative error
    reps = results.shape[0] # Number of replicas
    gran = results.shape[1] # Number of thetas

    # Compute the relative errors
    rels = []
    for r in range(reps):
        for g in range(gran):
            tilde = results[r,g,1]
            opt   = nar_opt_w[r,g]
            rel   = np.linalg.norm(tilde - opt)/np.linalg.norm(opt)
            rels.append(rel)
    rels = np.array(rels)
    rels = np.average(rels)
    table[i,1] = rels

if no_tab:
    print(heads)
    print(table)
else:
    def_names = np.array(def_names).reshape(5,1)
    table = np.concatenate((def_names,table), axis=1)
    print(tabulate(table, headers=heads, tablefmt='latex'), end='\n\n')

#
# Plot the upper bound
#
x     = np.linspace(0, np.pi/2, len(X_cond_ub))
left  = np.array([np.linalg.norm(wt - wo)/np.linalg.norm(wo) for wt, wo in zip(tilde_w, opt_w)])
prec  = np.finfo(np.float64).eps

right_X = X_cond_ub * prec
fig = plt.figure('conditioning upper-bound Xhat')
plt.xlabel(r'$\theta$')
plt.yscale('log')
plt.plot(x, left, label='Relative error')
plt.plot(x, right_X, label=r'$\partial\hat{X}$ upper-bound')
plt.legend(fontsize="x-large")
plt.show()

right_y = y_cond_ub * prec
fig = plt.figure('conditioning upper-bound y')
plt.xlabel(r'$\theta$')
plt.yscale('log')
plt.plot(x, left, label='Relative error')
plt.plot(x, right_y, label=r'$\partial y$ upper-bound')
plt.legend(fontsize="x-large")
plt.show()

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
