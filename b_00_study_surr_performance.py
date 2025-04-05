# This file is to analyze erros from the surrogate model

# Use latex for plots
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import numpy as np
import pandas as pd


# Load error_metrics.csv
error_metrics = pd.read_csv('error_metrics_model_0.csv')
print(error_metrics)

# Remove points with error > 0.5
error_metrics = error_metrics[error_metrics['err_cal_exp'] < 0.5]
error_metrics = error_metrics[error_metrics['err_surr_exp'] < 0.5]

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
sc1 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y1, s=5.0, color='red', label='Cal/Exp error')
sc2 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y2, s=5.0, color='blue', label='Surr/Exp error')

# Join the dots with vertical lines
for i in range(len(X)):
    ax.plot([X[plot_on[0]].iloc[i], X[plot_on[0]].iloc[i]], 
            [X[plot_on[1]].iloc[i], X[plot_on[1]].iloc[i]], 
            [y1.iloc[i], y2.iloc[i]], color='gray', linestyle='--', linewidth=0.5)

ax.set_xlabel(plot_on[0])
ax.set_ylabel(plot_on[1])
ax.set_zlabel('Error')
ax.set_title('Error of calibration experiment')

plt.legend()
plt.show()


# Run a quadratic regression on 'ar' and 'srr' to predict y1 and y2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Define the polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X[plot_on])

# Fit the model for y1
model_y1 = LinearRegression()
model_y1.fit(X_poly, y1)
y1_pred = model_y1.predict(X_poly)

# Fit the model for y2
model_y2 = LinearRegression()
model_y2.fit(X_poly, y2)
y2_pred = model_y2.predict(X_poly)

# Create a grid of values for 'ar' and 'srr'
ar_range = np.linspace(X['ar'].min(), X['ar'].max(), 50)
srr_range = np.linspace(X['srr'].min(), X['srr'].max(), 50)
ar_grid, srr_grid = np.meshgrid(ar_range, srr_range)
X_grid = pd.DataFrame({'ar': ar_grid.ravel(), 'srr': srr_grid.ravel()})

# Transform the grid values to polynomial features
X_grid_poly = poly.transform(X_grid)

# Predict the values on the grid for y1 and y2
y1_grid_pred = model_y1.predict(X_grid_poly)
y2_grid_pred = model_y2.predict(X_grid_poly)

# Plot the predicted values in a 3d surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(ar_grid, srr_grid, y1_grid_pred.reshape(ar_grid.shape), color='red', alpha=0.5, label='Cal/Exp error (pred)')
surf2 = ax.plot_surface(ar_grid, srr_grid, y2_grid_pred.reshape(ar_grid.shape), color='blue', alpha=0.5, label='Surr/Exp error (pred)')
ax.set_xlabel('ar')
ax.set_ylabel('srr')
ax.set_zlabel('Error')
ax.set_title('Error of calibration experiment')

plt.legend()

# Now plot the actual data points on top of the surface plot
sc1 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y1, s=5.0, color='red', label='Cal/Exp error')
sc2 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y2, s=5.0, color='blue', label='Surr/Exp error')

plt.show()


# Do same as above but for y3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc1 = ax.scatter(X[plot_on[0]], X[plot_on[1]], 0*y1, s=5.0, color='red', label='Cal/Cal error')
sc2 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y3, s=5.0, color='blue', label='Surr/Cal error')

# Join the dots with vertical lines
for i in range(len(X)):
    ax.plot([X[plot_on[0]].iloc[i], X[plot_on[0]].iloc[i]], 
            [X[plot_on[1]].iloc[i], X[plot_on[1]].iloc[i]], 
            [0, y3.iloc[i]], color='gray', linestyle='--', linewidth=0.5)

ax.set_xlabel(plot_on[0])
ax.set_ylabel(plot_on[1])
ax.set_zlabel('Error')
ax.set_title('Error of calibration experiment')

plt.legend()
plt.show()

# Run a quadratic regression on 'ar' and 'srr' to predict y3
# Fit the model for y3
model_y3 = LinearRegression()
model_y3.fit(X_poly, y3)
y3_pred = model_y3.predict(X_poly)

# Predict the values on the grid for y3
y3_grid_pred = model_y3.predict(X_grid_poly)

# Plot the predicted values in a 3d surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(ar_grid, srr_grid, 0*y1_grid_pred.reshape(ar_grid.shape), color='red', alpha=0.5, label='Cal/Cal error (pred)')
surf2 = ax.plot_surface(ar_grid, srr_grid, y3_grid_pred.reshape(ar_grid.shape), color='blue', alpha=0.5, label='Surr/Cal error (pred)')
ax.set_xlabel('ar')
ax.set_ylabel('srr')
ax.set_zlabel('Error')
ax.set_title('Error of calibration experiment')

plt.legend()

# Now plot the actual data points on top of the surface plot
sc1 = ax.scatter(X[plot_on[0]], X[plot_on[1]], 0*y1, s=5.0, color='red', label='Cal/Cal error')
sc2 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y3, s=5.0, color='blue', label='Surr/Cal error')

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

# Plot calibration errors vs surrogate errors
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(y1, y2, s=5.0)
ax.set_xlabel('Cal/Exp error')
ax.set_ylabel('Surr/Exp error')
ax.set_title('Calibration error vs Surrogate error')
# Add 1-1 line
plt.plot([0, 0.3], [0, 0.3], color='red')
# Set lims between 0 and 0.3 for both axes
plt.xlim(0, 0.3)
plt.ylim(0, 0.3)
plt.show()



