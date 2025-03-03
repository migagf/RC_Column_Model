# Surrogate Performance Plots

import os
import pandas as pd
import numpy as np

from database_processing_functions import *
from a_02_create_data_matrix import *
import matplotlib.pyplot as plt

# Load the surrogate_perfomance data
surrogate_performance = pd.read_csv('surrogate_performance_data_new.csv')

# Generate 3D scatter plot
pars = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

# Parameter labels [ar, lrr, srr, alr, sdr, smr]
par_labels = ['Aspect Ratio $D/L$', 'Long. Reinf. Ratio', 'Trans. Reinf. Ratio', 'Axial Load Ratio', 'Spacing', '$V_f/V_s$']

xpar_index, ypar_index = 1, 5

xpar = pars[xpar_index]
ypar = pars[ypar_index]

# Labels for the plots
xpar_label = par_labels[xpar_index]
ypar_label = par_labels[ypar_index]

# Filter only test data
test_data = surrogate_performance[surrogate_performance['TestOrTrain'] == 'Test']
# Plot properties
dpi_plots = 100
figuresize = (5, 5)
colors = {'Flexure': 'b', 'Flexure-Shear': 'k', 'Shear': 'r'}

# Create a 3D scatter plot of rmse_sur 
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')

# Create a color map based on FailureType
color_values = test_data['Type'].map(colors)
ax.scatter(test_data[xpar], test_data[ypar], test_data['RMSESurr'], c=color_values, marker='o', label='Surrogate')
# Add rmse_cal
ax.scatter(test_data[xpar], test_data[ypar], test_data['RMSECal'], c=color_values, marker='^', label='Calibration')  
# Join both with vertical lines

for i in range(len(test_data)):
    ax.plot([test_data[xpar].iloc[i], test_data[xpar].iloc[i]], 
            [test_data[ypar].iloc[i], test_data[ypar].iloc[i]], 
            [test_data['RMSESurr'].iloc[i], test_data['RMSECal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)
    
ax.legend()
plt.title('RMSE on Test Data')
ax.set_zlim(0, 0.015)
#ax.set_ylim(0, 2)
# Add labels for x and y axes
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
plt.show()


# Same as above for train data

# Filter only train data
train_data = surrogate_performance[surrogate_performance['TestOrTrain'] == 'Train']

fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')

# Create a color map based on FailureType
color_values = train_data['Type'].map(colors)

# Create a 3D scatter plot of rmse_sur
ax.scatter(train_data[xpar], train_data[ypar], train_data['RMSESurr'], c=color_values, marker='o', label='Surrogate')
# Add rmse_cal
ax.scatter(train_data[xpar], train_data[ypar], train_data['RMSECal'], c=color_values, marker='^', label='Calibration')
# Join both with vertical lines
for i in range(len(train_data)):
    ax.plot([train_data[xpar].iloc[i], train_data[xpar].iloc[i]], 
            [train_data[ypar].iloc[i], train_data[ypar].iloc[i]], 
            [train_data['RMSESurr'].iloc[i], train_data['RMSECal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)
    
ax.legend()
plt.title('RMSE on Train Data')
ax.set_zlim(0, 0.015)
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
plt.show()
# ax.set_ylim(0, 2)


# Do same with all data
all_data = surrogate_performance
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')
# Create a 3D scatter plot of rmse_sur

# Create a color map based on FailureType

color_values = all_data['Type'].map(colors)

# Create a 3D scatter plot of rmse_sur with color based on FailureType
ax.scatter(all_data[xpar], all_data[ypar], all_data['RMSESurr'], c=color_values, marker='o', label='Surrogate')
ax.scatter(all_data[xpar], all_data[ypar], all_data['RMSECal'], c=color_values, marker='^', label='Calibration')

# Join both with vertical lines
for i in range(len(all_data)):
    ax.plot([all_data[xpar].iloc[i], all_data[xpar].iloc[i]], 
            [all_data[ypar].iloc[i], all_data[ypar].iloc[i]], 
            [all_data['RMSESurr'].iloc[i], all_data['RMSECal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)

ax.legend()
plt.title('RMSE on All Data')
ax.set_zlim(0, 0.015)
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
plt.show()
#ax.set_ylim(0, 2)

