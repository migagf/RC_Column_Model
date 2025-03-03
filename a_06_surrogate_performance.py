# Surrogate Performance Plots

import os
import pandas as pd
import numpy as np

from database_processing_functions import *
from a_02_create_data_matrix import *
import matplotlib.pyplot as plt


# Set latex as intepreter for matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


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
color_labels_surrogate = {'GP - Flexure': 'b', 'GP - Flexure-Shear': 'k', 'GP - Shear': 'r'}
color_labels_cals = {'Cal - Flexure': 'b', 'Cal - Flexure-Shear': 'k', 'Cal - Shear': 'r'}

# Create a 3D scatter plot of rmse_sur 
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')

# Create a color map based on FailureType
color_values = test_data['Type'].map(colors)
# Create a 3D scatter plot of rmse_sur with color labels
ax.scatter(test_data[xpar], test_data[ypar], test_data['RMSESurr'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add rmse_cal
ax.scatter(test_data[xpar], test_data[ypar], test_data['RMSECal'], c=color_values, marker='^')  
# Join both with vertical lines

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')
    
ax.legend(loc='upper left', fontsize=8)

# Add lines connecting calibrated and surrogate points
for i in range(len(test_data)):
    ax.plot([test_data[xpar].iloc[i], test_data[xpar].iloc[i]], 
            [test_data[ypar].iloc[i], test_data[ypar].iloc[i]], 
            [test_data['RMSESurr'].iloc[i], test_data['RMSECal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)
    
plt.title('RMSE on Test Data')
ax.set_zlim(0, 0.015)
#ax.set_ylim(0, 2)
# Add labels for x and y axes
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)

# Set view angle to 45 degrees from all sides
ax.view_init(elev=30, azim=-45)
plt.show()


# Same as above for train data

# Filter only train data
train_data = surrogate_performance[surrogate_performance['TestOrTrain'] == 'Train']

fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')

# Create a color map based on FailureType
color_values = train_data['Type'].map(colors)

# Create a 3D scatter plot of rmse_sur
ax.scatter(train_data[xpar], train_data[ypar], train_data['RMSESurr'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add rmse_cal
ax.scatter(train_data[xpar], train_data[ypar], train_data['RMSECal'], c=color_values, marker='^')  
# Join both with vertical lines

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')
    
ax.legend(loc='upper left', fontsize=8)

# Add lines connecting calibrated and surrogate points
for i in range(len(train_data)):
    ax.plot([train_data[xpar].iloc[i], train_data[xpar].iloc[i]], 
            [train_data[ypar].iloc[i], train_data[ypar].iloc[i]], 
            [train_data['RMSESurr'].iloc[i], train_data['RMSECal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)
    
plt.title('RMSE on Train Data')
ax.set_zlim(0, 0.015)
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)

# Set view angle to 45 degrees from all sides
ax.view_init(elev=30, azim=-45)
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
ax.scatter(all_data[xpar], all_data[ypar], all_data['RMSESurr'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add rmse_cal
ax.scatter(all_data[xpar], all_data[ypar], all_data['RMSECal'], c=color_values, marker='^')  

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')
    
ax.legend(loc='upper left', fontsize=8)

# Join both with vertical lines
for i in range(len(all_data)):
    ax.plot([all_data[xpar].iloc[i], all_data[xpar].iloc[i]], 
            [all_data[ypar].iloc[i], all_data[ypar].iloc[i]], 
            [all_data['RMSESurr'].iloc[i], all_data['RMSECal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)

plt.title('RMSE on All Data')
ax.set_zlim(0, 0.015)
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)

# Set view angle to 45 degrees from all sides
ax.view_init(elev=30, azim=-45)
plt.show()
#ax.set_ylim(0, 2)


#%%
# Do the same as above for EnergyCal and EnergySur

# Filter only test data
test_data = surrogate_performance[surrogate_performance['TestOrTrain'] == 'Test']
# Create a 3D scatter plot of energy_sur
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')

# Create a color map based on FailureType
color_values = test_data['Type'].map(colors)

ax.scatter(test_data[xpar], test_data[ypar], test_data['EnergySurr'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add energy_cal
ax.scatter(test_data[xpar], test_data[ypar], test_data['EnergyCal'], c=color_values, marker='^')

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')

# Join both with vertical lines
for i in range(len(test_data)):
    ax.plot([test_data[xpar].iloc[i], test_data[xpar].iloc[i]], 
            [test_data[ypar].iloc[i], test_data[ypar].iloc[i]], 
            [test_data['EnergySurr'].iloc[i], test_data['EnergyCal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)

ax.legend(loc='upper left', fontsize=8)
plt.title('Energy on Test Data')
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
ax.view_init(elev=30, azim=-45)
plt.show()

# Do same for train data
train_data = surrogate_performance[surrogate_performance['TestOrTrain'] == 'Train']
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')

# Create a color map based on FailureType
color_values = train_data['Type'].map(colors)

# Create a 3D scatter plot of energy_sur
ax.scatter(train_data[xpar], train_data[ypar], train_data['EnergySurr'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add energy_cal
ax.scatter(train_data[xpar], train_data[ypar], train_data['EnergyCal'], c=color_values, marker='^')

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')

# Join both with vertical lines
for i in range(len(train_data)):
    ax.plot([train_data[xpar].iloc[i], train_data[xpar].iloc[i]], 
            [train_data[ypar].iloc[i], train_data[ypar].iloc[i]], 
            [train_data['EnergySurr'].iloc[i], train_data['EnergyCal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)

ax.legend(loc='upper left', fontsize=8)
plt.title('Energy on Train Data')
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
ax.view_init(elev=30, azim=-45)
plt.show()

# Do same for all data
all_data = surrogate_performance
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')

# Create a color map based on FailureType
color_values = all_data['Type'].map(colors)

# Create a 3D scatter plot of energy_sur
ax.scatter(all_data[xpar], all_data[ypar], all_data['EnergySurr'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add energy_cal
ax.scatter(all_data[xpar], all_data[ypar], all_data['EnergyCal'], c=color_values, marker='^')

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')

# Join both with vertical lines
for i in range(len(all_data)):
    ax.plot([all_data[xpar].iloc[i], all_data[xpar].iloc[i]], 
            [all_data[ypar].iloc[i], all_data[ypar].iloc[i]], 
            [all_data['EnergySurr'].iloc[i], all_data['EnergyCal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)

ax.legend(loc='upper left', fontsize=8)
plt.title('Energy on All Data')
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
ax.view_init(elev=30, azim=-45)
plt.show()

# %%
# Generate 3D scatter plot
# Filter only test data
test_data = surrogate_performance[surrogate_performance['TestOrTrain'] == 'Test']
# Create a 3D scatter plot of stiff_sur
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')
# Create a color map based on FailureType
color_values = test_data['Type'].map(colors)
ax.scatter(test_data[xpar], test_data[ypar], test_data['StiffSur'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add stiff_cal
ax.scatter(test_data[xpar], test_data[ypar], test_data['StiffCal'], c=color_values, marker='^')

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')

# Join both with vertical lines
for i in range(len(test_data)):
    ax.plot([test_data[xpar].iloc[i], test_data[xpar].iloc[i]], 
            [test_data[ypar].iloc[i], test_data[ypar].iloc[i]], 
            [test_data['StiffSur'].iloc[i], test_data['StiffCal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)

ax.legend(loc='upper left', fontsize=8)
plt.title('Stiffness on Test Data')
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
ax.view_init(elev=30, azim=-45)
plt.show()

# Do same for train data
train_data = surrogate_performance[surrogate_performance['TestOrTrain'] == 'Train']
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')
# Create a color map based on FailureType
color_values = train_data['Type'].map(colors)
# Create a 3D scatter plot of stiff_sur
ax.scatter(train_data[xpar], train_data[ypar], train_data['StiffSur'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add stiff_cal
ax.scatter(train_data[xpar], train_data[ypar], train_data['StiffCal'], c=color_values, marker='^')

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')

# Join both with vertical lines
for i in range(len(train_data)):
    ax.plot([train_data[xpar].iloc[i], train_data[xpar].iloc[i]], 
            [train_data[ypar].iloc[i], train_data[ypar].iloc[i]], 
            [train_data['StiffSur'].iloc[i], train_data['StiffCal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)

ax.legend(loc='upper left', fontsize=8)
plt.title('Stiffness on Train Data')
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
ax.view_init(elev=30, azim=-45)
plt.show()

# Do same for all data
all_data = surrogate_performance
fig = plt.figure(dpi=dpi_plots, figsize=figuresize)
ax = fig.add_subplot(111, projection='3d')

# Create a color map based on FailureType
color_values = all_data['Type'].map(colors)
# Create a 3D scatter plot of stiff_sur
ax.scatter(all_data[xpar], all_data[ypar], all_data['StiffSur'], c=color_values, marker='o')

# Add color labels
for label, color in color_labels_surrogate.items():
    ax.scatter([], [], c=color, label=f'{label}')

# Add stiff_cal
ax.scatter(all_data[xpar], all_data[ypar], all_data['StiffCal'], c=color_values, marker='^')

# Add color labels for calibration
for label, color in color_labels_cals.items():
    ax.scatter([], [], c=color, marker='^', label=f'{label}')

# Join both with vertical lines
for i in range(len(all_data)):
    ax.plot([all_data[xpar].iloc[i], all_data[xpar].iloc[i]], 
            [all_data[ypar].iloc[i], all_data[ypar].iloc[i]], 
            [all_data['StiffSur'].iloc[i], all_data['StiffCal'].iloc[i]], c='k', linestyle='--', linewidth=0.5)

ax.legend(loc='upper left', fontsize=8)
plt.title('Stiffness on All Data')
ax.set_xlabel(xpar_label)
ax.set_ylabel(ypar_label)
ax.view_init(elev=30, azim=-45)
plt.show()