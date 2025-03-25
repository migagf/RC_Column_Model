# Figures for paper

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from run_column_model import *
import pandas as pd

# Import packages for PCA
from sklearn.decomposition import PCA
from get_bw_params import *
from mpl_toolkits.mplot3d import Axes3D

# Use latex for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

save_figs_to = 'Figures'
# Load data
with open(os.path.join('quoFEM_TMCMC', 'test_001', 'test_file.json'), 'r') as f:
    test_data = json.load(f)

'''plt.figure(figsize=(3, 2.5))
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
plt.xlabel('Drift Ratio $\Delta/L$ (\%)')
plt.ylabel('Normalized Shear $V/V_{max}$')
plt.tight_layout()
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
results = run_model(test_data, bw_params, do_plots=False)

'''plt.figure(figsize=(3, 2.5))
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
         label='Exp.', color='red')
plt.plot(100 * np.array(results['sim_data']['drift']), 
         np.array(results['sim_data']['nforce']), 
         label='Sim.', color='blue')
plt.legend()
plt.xlabel('Drift Ratio $\Delta/L$ (\%)')
plt.ylabel('Normalized Shear $V/V_{max}$')
plt.tight_layout()
plt.savefig(os.path.join(save_figs_to, 'hysteresis_2.pdf'), bbox_inches='tight')
plt.show()'''

# Create a plot of a uniform distribution between 0 and 2
# Fill the curve with blue color

'''plt.figure(figsize=(3, 2.5))
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
plt.show()'''



# Load data_all.csv file
data_all = pd.read_csv('data_all.csv')

X = data_all[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']]

# Get max and min values of each column
max_vals = X.max()
min_vals = X.min()

print(max_vals)
print(min_vals)

# [0: gamma,    1: kappa,       2: eta1, 
#  3: sig,      4: lam,         5: mup, 
#  6: sigp,     7: rsmax,       8: alpha, 
#  9: alpha1,   10: alpha2,     11: betam1, 
#  12: n,       13: kappa_k ]

ar_values = np.linspace(1.0, 8.0, 20)
srr_values = np.linspace(0.0, 0.3, 20)

sigma1 = np.zeros((len(ar_values), len(srr_values)))
#sigma2 = np.zeros((len(ar_values), len(srr_values)))
#sigma3 = np.zeros((len(ar_values), len(srr_values)))

#betam11 = np.zeros((len(ar_values), len(srr_values)))
#betam12 = np.zeros((len(ar_values), len(srr_values)))
#betam13 = np.zeros((len(ar_values), len(srr_values)))

#alpha11 = np.zeros((len(ar_values), len(srr_values)))
#alpha12 = np.zeros((len(ar_values), len(srr_values)))
#alpha13 = np.zeros((len(ar_values), len(srr_values)))

for i, ar in enumerate(ar_values):
    for j, srr in enumerate(srr_values):
        lrr = 0.1
        alr = 0.2
        sdr = 0.1
        smr1 = 0.5
        smr2 = 1.0
        smr3 = 1.5

        bw_model_params1, min_error1, failure_mode1 = get_BW_params([ar, lrr, srr, alr, sdr, smr1], mode='other')
        #bw_model_params2, min_error2, failure_mode2 = get_BW_params([ar, lrr, srr, alr, sdr, smr2], mode='other')
        #bw_model_params3, min_error3, failure_mode3 = get_BW_params([ar, lrr, srr, alr, sdr, smr3], mode='other')

        # Store bw_model_params[3]
        sigma1[i, j] = bw_model_params1[3]
        #sigma2[i, j] = bw_model_params2[3]
        #sigma3[i, j] = bw_model_params3[3]

        #betam11[i, j] = bw_model_params1[11]
        #betam12[i, j] = bw_model_params2[11]
        #betam13[i, j] = bw_model_params3[11]

        #alpha11[i, j] = bw_model_params1[9]
        #alpha12[i, j] = bw_model_params2[9]
        #alpha13[i, j] = bw_model_params3[9]


# Create a meshgrid for plotting
AR, SRR = np.meshgrid(ar_values, srr_values)

fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(AR, SRR, sigma1.T, cmap='cividis', alpha=0.8, label='smr=0.5')
'''ax.plot_surface(AR, SRR, sigma2.T, cmap='indianred', alpha=0.6, label='smr=1.0')
ax.plot_surface(AR, SRR, sigma3.T, color='darkred', alpha=0.6, label='smr=1.5')
'''
ax.set_xlabel(r'Non-dimensional $p_1$')
ax.set_ylabel(r'Non-dimensional $p_2$')
ax.set_zlabel(r'Bouc-Wen parameter $\theta_j$')
# Set ticks
ax.set_xticks([1.0, 4.5, 8.0])
ax.set_yticks([0.0, 0.15, 0.30])
ax.set_zticks([0.0, 0.25, 0.5])
# Delete tick labels
ax.set_xticklabels([1.0, '', 8.0])
ax.set_yticklabels([0.0, '', 0.3])
ax.set_zticklabels([0.0, '', 0.5])
plt.tight_layout()
plt.savefig(os.path.join(save_figs_to, 'surrogate_picture.pdf'), bbox_inches='tight')
# Change axes so they look straight
ax.view_init(30, 45)


plt.show()

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(AR, SRR, betam11.T, color='lightcoral', alpha=0.6, label='smr=0.5')
ax.plot_surface(AR, SRR, betam12.T, color='indianred', alpha=0.6, label='smr=1.0')
ax.plot_surface(AR, SRR, betam13.T, color='darkred', alpha=0.6, label='smr=1.5')

ax.set_xlabel('Aspect Ratio ($D/L$)')
ax.set_ylabel('SRR')
ax.set_zlabel('$\\beta_{m,1}$')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(AR, SRR, alpha11.T, color='lightcoral', alpha=0.6, label='smr=0.5')
ax.plot_surface(AR, SRR, alpha12.T, color='indianred', alpha=0.6, label='smr=1.0')
ax.plot_surface(AR, SRR, alpha13.T, color='darkred', alpha=0.6, label='smr=1.5')

ax.set_xlabel('Aspect Ratio ($D/L$)')
ax.set_ylabel('SRR')
ax.set_zlabel('$\\alpha_{1}$')
plt.show()'''



