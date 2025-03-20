# This file is to analyze erros from the surrogate model

# Use latex for plots
import matplotlib.pyplot as plt
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

import numpy as np
import pandas as pd


# Load error_metrics.csv
error_metrics = pd.read_csv('error_metrics.csv')
print(error_metrics)

# Extract the predictors X = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']
X = error_metrics[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']]
y1 = error_metrics['err_cal_exp']
y2 = error_metrics['err_surr_exp']
y3 = error_metrics['err_surr_cal']

# Normalize the predictors
#X = (X - X.mean()) / X.std()

# Do PCA on normalized predictors
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the error metrics using two of the predictors

plot_on = ['ar', 'srr']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X[plot_on[0]], X[plot_on[1]], y1, s=5.0, color='red', label='Cal/Exp error')
sc = ax.scatter(X[plot_on[0]], X[plot_on[1]], y2, s=5.0, color='blue', label='Surr/Exp error')
sc = ax.scatter(X[plot_on[0]], X[plot_on[1]], y3, s=5.0, color='green', label='Surr/Cal error')
ax.set_xlabel(plot_on[0])
ax.set_ylabel(plot_on[1])
ax.set_zlabel('Error')
ax.set_title('Error of calibration experiment')
ax.set_zlim(0, 0.2)
plt.legend()
plt.show()

# Plot same thing but now with respect to two parameters (no PCA)

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X['ar'], X['lrr'], y1, c=y1, s=5.0)
ax.set_xlabel('ar')
ax.set_ylabel('lrr')
ax.set_zlabel('Error')
ax.set_title('Error of calibration experiment')
plt.show()
'''

