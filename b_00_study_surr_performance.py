# This file is to analyze erros from the surrogate model

# Use latex for plots
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

figures_dir = 'Figures'

# Load error_metrics.csv
error_metrics = pd.read_csv('gp_training_data/error_metrics/err_metrics_no_split.csv')
print(error_metrics)

# Remove points with error > 0.5
error_metrics = error_metrics[error_metrics['err_cal_exp'] < 0.25]
error_metrics = error_metrics[error_metrics['err_surr_exp'] < 0.25]
error_metrics = error_metrics[error_metrics['err_surr_cal'] < 0.25]

# Extract the predictors X = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']
X = error_metrics[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'FailureMode']]

# Define new column names
data_col_labels = ['AR', 'LRR', 'SRR', 'ALR', 'SDR', 'SMR', 'Failure Mode']

# Rename columns
X.columns = data_col_labels


y1 = error_metrics['err_cal_exp']
y2 = error_metrics['err_surr_exp']
y3 = error_metrics['err_surr_cal']

# Plot the error metrics using two of the predictors
plot_on = ['AR', 'SRR']  # Updated to match new column labels

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc1 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y1, s=5.0, color='red', label='Cal/Exp MAE')
sc2 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y2, s=5.0, color='blue', label='GP/Exp MAE')

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
plt.show()'''


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

# Create a grid of values for 'AR' and 'SRR'
ar_range = np.linspace(X['AR'].min(), X['AR'].max(), 50)  # Updated column name
srr_range = np.linspace(X['SRR'].min(), X['SRR'].max(), 50)  # Updated column name
ar_grid, srr_grid = np.meshgrid(ar_range, srr_range)
X_grid = pd.DataFrame({'AR': ar_grid.ravel(), 'SRR': srr_grid.ravel()})  # Updated column names

# Transform the grid values to polynomial features
X_grid_poly = poly.transform(X_grid)

# Predict the values on the grid for y1 and y2
y1_grid_pred = model_y1.predict(X_grid_poly)
y2_grid_pred = model_y2.predict(X_grid_poly)

# Plot the predicted values in a 3d surface plot (grayscale, high contrast)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(
    ar_grid, srr_grid, y1_grid_pred.reshape(ar_grid.shape),
    color='0.1', alpha=0.5, edgecolor='k', linewidth=0.1, label='Cal/Exp MAE Surface Fit'
)
surf2 = ax.plot_surface(
    ar_grid, srr_grid, y2_grid_pred.reshape(ar_grid.shape),
    color='0.9', alpha=0.5, edgecolor='k', linewidth=0.1, label='GP/Exp MAE Surface Fit'
)
ax.set_xlabel('AR')
ax.set_ylabel('SRR')
ax.set_zlabel('MAE')

# Set zlim
ax.set_zlim(0, 0.3)

ax.view_init(elev=15, azim=-30)

# Now plot the actual data points on top of the surface plot (grayscale markers)
sc1 = ax.scatter(
    X[plot_on[0]], X[plot_on[1]], y1, s=10.0, marker='s', color='k', label='Cal/Exp MAE'
)
sc2 = ax.scatter(
    X[plot_on[0]], X[plot_on[1]], y2, s=12.0, marker='o', edgecolors='k', facecolors='none', linewidth=0.8, label='GP/Exp MAE'
)

plt.legend()
plt.savefig(os.path.join(figures_dir, 'error_metrics_with_surface.pdf'))

print('saved figure 1 ')
plt.show()


'''# Do same as above but for y3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc1 = ax.scatter(X[plot_on[0]], X[plot_on[1]], 0*y1, s=5.0, color='red', label='Cal/Cal MAE')
sc2 = ax.scatter(X[plot_on[0]], X[plot_on[1]], y3, s=5.0, color='blue', label='GP/Cal MAE')

# Join the dots with vertical lines
for i in range(len(X)):
    ax.plot([X[plot_on[0]].iloc[i], X[plot_on[0]].iloc[i]], 
            [X[plot_on[1]].iloc[i], X[plot_on[1]].iloc[i]], 
            [0, y3.iloc[i]], color='gray', linestyle='--', linewidth=0.5)

ax.set_xlabel(plot_on[0])
ax.set_ylabel(plot_on[1])
ax.set_zlabel('MAE')

plt.legend()
plt.savefig('error_metrics_3d.pdf')

plt.show()'''

# Run a quadratic regression on 'ar' and 'srr' to predict y3
# Fit the model for y3
model_y3 = LinearRegression()
model_y3.fit(X_poly, y3)
y3_pred = model_y3.predict(X_poly)

# Predict the values on the grid for y3
y3_grid_pred = model_y3.predict(X_grid_poly)

# Plot the predicted values in a 3d surface plot (grayscale, high contrast)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface with grayscale styling matching first plot
surf2 = ax.plot_surface(
    ar_grid, srr_grid, y3_grid_pred.reshape(ar_grid.shape),
    color='0.6', alpha=0.5, edgecolor='k', linewidth=0.1
)
ax.set_xlabel('AR')
ax.set_ylabel('SRR')
ax.set_zlabel('MAE')

ax.set_zlim(0, 0.3)
# Match view angle from first plot
ax.view_init(elev=15, azim=-30)

# Plot the actual data points with grayscale markers based on Failure Mode
categories = X['Failure Mode'].unique()
# Use grayscale tones and different marker shapes for each category
gray_tones = ['0.1', '0.45', '0.75']  # dark, medium, light gray
markers = ['s', 'o', '^']  # square, circle, triangle

for i, category in enumerate(categories):
    mask = X['Failure Mode'] == category
    ax.scatter(
        X.loc[mask, plot_on[0]], 
        X.loc[mask, plot_on[1]], 
        y3[mask], 
        s=10.0, 
        marker=markers[i % len(markers)],
        facecolors=gray_tones[i % len(gray_tones)],
        edgecolors='k',
        linewidth=0.5,
        label=f'{category}'
    )

plt.legend(title='Failure Mode')
plt.savefig(os.path.join(figures_dir, 'error_metrics_failuremode.pdf'))
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




