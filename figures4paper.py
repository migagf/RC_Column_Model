# Figures for paper


import numpy as np
import matplotlib.pyplot as plt
import json
import os
from run_column_model import *
import pandas as pd

# Import packages for PCA
from sklearn.decomposition import PCA

# Use latex for plots
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

save_figs_to = 'Figures'
# Load data
with open(os.path.join('quoFEM_TMCMC', 'test_001', 'test_file.json'), 'r') as f:
    test_data = json.load(f)

'''plt.figure(figsize=(5, 3.5))
# Set xlim
plt.xlim([-3.1, 3.1])
plt.ylim([-1.1, 1.1])
# Add x-ticks
plt.yticks([-1.0, 0.0, 1.0])
# Add y-ticks
plt.xticks([-3.0, 0.0, 3.0])
# Add x=0 line
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
# Add y=0 line
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
plt.plot(100 * np.array(test_data['data']['disp']) / test_data['L_Inflection'], 
         np.array(test_data['data']['force']) / max(test_data['data']['force']), 
         label='True model', color='red')
plt.savefig(os.path.join(save_figs_to, 'hysteresis_1.pdf'), bbox_inches='tight')
plt.show()'''

# Run one simulation with the Bouc-Wen
gamma      = 1.277177
kappa      = 0.977073
eta1       = 1.845268
sig        = 0.585665
lam        = 0.529793
mup        = 1.650068
sigp       = 0.779072
rsmax      = 0.948117
alpha      = 0.000115
alpha1     = 4.961202
alpha2     = 1.234721
betam1     = 0.000328
n          = 1.504958
kappa_k    = 3.597515

bw_params = [gamma, kappa, eta1, sig, lam, mup, sigp, rsmax, alpha, alpha1, alpha2, betam1, n, kappa_k]
# results = run_model(test_data, bw_params, do_plots=False)

'''plt.figure(figsize=(5, 3.5))
# Set xlim
plt.xlim([-3.1, 3.1])
plt.ylim([-1.1, 1.1])
# Add x-ticks
plt.yticks([-1.0, 0.0, 1.0])
# Add y-ticks
plt.xticks([-3.0, 0.0, 3.0])
# Add x=0 line
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
# Add y=0 line
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
plt.plot(100 * np.array(test_data['data']['disp']) / test_data['L_Inflection'], 
         np.array(test_data['data']['force']) / max(test_data['data']['force']), 
         label='Experiment', color='red')
plt.plot(100 * np.array(results['sim_data']['drift']), 
         np.array(results['sim_data']['nforce']), 
         label='Simulation', color='blue')
plt.legend()
plt.savefig(os.path.join(save_figs_to, 'hysteresis_2.pdf'), bbox_inches='tight')
plt.show()'''

# Create a plot of a uniform distribution between 0 and 2
# Fill the curve with blue color

plt.figure(figsize=(4, 3))
x = np.linspace(1.0, 3.0, 100)
y = np.ones_like(x) * 0.5
plt.fill_between(x, y, color='red', alpha=0.4, label='Prior')
# Add a posterior distribution that looks like a truncated normal distribution
x = np.linspace(1.0, 3.0, 100)
y = 5 * np.exp(-20 * (x - 2.2)**2)
plt.fill_between(x, y, color='blue', label='Posterior', alpha=0.4)
plt.xlim([0.8, 3.2])
plt.ylim([0, 5.0])
plt.xticks([1.0, 2.2, 3.0])
plt.gca().set_xticklabels([r'$\theta_0$', r'$\theta^*$',r'$\theta_1$'])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.legend()
plt.savefig(os.path.join(save_figs_to, 'bayesiancal.pdf'), bbox_inches='tight')
plt.show()



# Load data_all.csv file
data_all = pd.read_csv('data_all.csv')

X = data_all[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']]

# Do PCA on X
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

'''# Plot the PCA
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5.0)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA of the predictors')
plt.show()'''

# Create a grid with the PCA1 and PCA2
x = np.linspace(-3, 3, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Use X and Y values to back-calculate the original predictors
X_back = pca.inverse_transform(np.array([X.flatten(), Y.flatten()]).T)

# Create 3d plots of the original predictors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_back[:, 0], X_back[:, 1], X_back[:, 2], s=5.0, c='red')
ax.set_xlabel('ar')
ax.set_ylabel('lrr')
ax.set_zlabel('srr')
# Include the pointe in the original data
ax.scatter(data_all['ar'], data_all['lrr'], data_all['srr'], s=5.0, c='blue')
plt.show()