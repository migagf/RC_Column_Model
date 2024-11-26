# Plot the surrogate model
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

from database_processing_functions import *
from create_data_matrix import *
import matplotlib.pyplot as plt

# Plot Surrogate Predictions
from gp_predict import *
from diff_evolution_col import *
import matplotlib.pyplot as plt
import seaborn as sns

# This code is to plot the surrogate model

# Load the merged_calibration_data.csv
df = pd.read_csv('merged_calibration_data.csv')

# 
# Define surrogate model parameters and code

# ------
# Here, call surrogate model for flexure failure
# ------
f_surrogate_dir = os.getcwd() + '/quoFEM_Surrogate/flexure_pd_010_just_good_data'
f_template_dir = os.path.join(f_surrogate_dir, 'templatedir_SIM')
f_surrogate_file = f_surrogate_dir + '/SimGpModel.json'

# ------
# Here, call surrogate model for shear failure
# ------
s_surrogate_dir = os.getcwd() + '/quoFEM_Surrogate/combined_pd_005_just_good_data'
s_template_dir = os.path.join(f_surrogate_dir, 'templatedir_SIM')
s_surrogate_file = s_surrogate_dir + '/SimGpModel.json'

# Input_json is the same for both cases
input_json = os.getcwd() + '/quoFEM_Surrogate/scInput.json'

# For all the cases in the dataset, get BW model predictions from the surrogate
predicted_params = pd.DataFrame(columns = df.columns[0:14])
predicted_variances = pd.DataFrame(columns = df.columns[0:14])

for ii in range(len(df)):
    name = df['name'][ii]
    
    # Extract the calibrated and nondimensional parameters for both training and testing
    cal_params = df.iloc[ii, 0:14]
    nondim_params = df.iloc[ii, 17:23]
    
    params_list = [
        ["RV_column1", nondim_params.iloc[0]],
        ["RV_column2", nondim_params.iloc[1]],
        ["RV_column3", nondim_params.iloc[2]],
        ["RV_column4", nondim_params.iloc[3]],
        ["RV_column5", nondim_params.iloc[4]], 
        ["RV_column6", nondim_params.iloc[5]]
        ]
    
    # Here, call surrogate model
    try:
        # Select the surrogate model
        if df['FailureType'][ii] == 'Flexure':
            # print('Flexure-type failure')
            surrogate_file = f_surrogate_file
        else:
            # print('Shear-type failure')
            surrogate_file = s_surrogate_file

        params, variance = main(params_list, [], surrogate_file, 'dummyout.out', input_json)
        params = pd.DataFrame(params, columns = df.columns[0:14])
        variance = pd.DataFrame(variance, columns = df.columns[0:14])

        predicted_params = pd.concat([predicted_params, params])
        predicted_variances = pd.concat([predicted_variances, variance])
    
    except Exception as e:
        print('Error in surrogate model', e)
        params = pd.DataFrame(np.nan*np.array(params), columns = df.columns[0:14])
        variance = pd.DataFrame(np.nan*np.array(variance), columns = df.columns[0:14])
        
        predicted_params = pd.concat([predicted_params, params])
        predicted_variances = pd.concat([predicted_variances, variance])
        continue

# Plot ar, all vs BW model predictions
# Pick two: 
# (0) ar 
# (1) lrr 
# (2) srr 
# (3) alr 
# (4) sdr
# (5) smr

xvar = 'ar'
yvar = 'smr'

xvals = df[xvar]
yvals = df[yvar]

# Select index for the parameter of interest...
par_index = 5

# Calibrated parameter
indices_flexure = df[df['FailureType'] == 'Flexure'].index
indices_shear = df[df['FailureType'] != 'Flexure'].index

flexure_data = df.iloc[indices_flexure, par_index]
shear_data = df.iloc[indices_shear, par_index]

# Surrogate prediction
bwpar = predicted_params.iloc[:, par_index]

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xvals, yvals, bwpar, c='r', marker='o', label='Surrogate-Predicted')
ax.scatter(xvals[indices_flexure], yvals[indices_flexure], flexure_data, c='b', marker='o', label='Cal-Flex')
ax.scatter(xvals[indices_shear], yvals[indices_shear], shear_data, c='g', marker='o', label='Cal-Shear')
ax.set_xlabel(xvar)
ax.set_ylabel(yvar)
plt.title('BW Model Parameter Comparison')
plt.legend()
plt.show()


nondim_params_all = df.iloc[:, 17:23]
# nondim_params_all.median()



# Now, lets generate a grid of values for nondimensional parameters 
from tqdm import tqdm

# Define the grid
x = np.linspace(0.01, 1.5, 30)
y = np.linspace(0.01, 1.5, 30)
X, Y = np.meshgrid(x, y)

Z1 = np.zeros_like(X)
Z2 = np.zeros_like(X)
Z3 = np.zeros_like(X)
Z4 = np.zeros_like(X)
Z5 = np.zeros_like(X)
Z6 = np.zeros_like(X)
Z7 = np.zeros_like(X)
Z8 = np.zeros_like(X)
Z9 = np.zeros_like(X)
Z10 = np.zeros_like(X)
Z11 = np.zeros_like(X)
Z12 = np.zeros_like(X)
Z13 = np.zeros_like(X)

# Evaluate the surrogate model on the values of the grid

for ii in tqdm(range(len(x))):
    for jj in range(len(y)):

        # Generate a list of parameters
        params_list = [
            ["RV_column1", x[ii]],
            ["RV_column2", nondim_params_all.median()[1]],
            ["RV_column3", nondim_params_all.median()[2]],
            ["RV_column4", nondim_params_all.median()[3]],
            ["RV_column5", nondim_params_all.median()[4]], 
            ["RV_column6", y[jj]]
            ]
        
        # Simple model for classification of failure mode
        if y[jj] > 0.7:
            failuretype = 'Shear'
        else:
            failuretype = 'Flexure'

        # Here, call surrogate model
        try:
            # Select the surrogate model
            if failuretype == 'Flexure':
                # print('Flexure-type failure')
                surrogate_file = f_surrogate_file
            else:
                # print('Shear-type failure')
                surrogate_file = s_surrogate_file

            params, variance = main(params_list, [], surrogate_file, 'dummyout.out', input_json)
            params = pd.DataFrame(params, columns = df.columns[0:14])
            # variance = pd.DataFrame(variance, columns = df.columns[0:14])

            #predicted_params = pd.concat([predicted_params, params])
            #predicted_variances = pd.concat([predicted_variances, variance])
            Z1[jj, ii] = params.iloc[0, 0]
            Z2[jj, ii] = params.iloc[0, 1]
            Z3[jj, ii] = params.iloc[0, 2]
            Z4[jj, ii] = params.iloc[0, 3]
            Z5[jj, ii] = params.iloc[0, 4]
            Z6[jj, ii] = params.iloc[0, 5]
            Z7[jj, ii] = params.iloc[0, 6]
            Z8[jj, ii] = params.iloc[0, 7]
            Z9[jj, ii] = params.iloc[0, 8]
            Z10[jj, ii] = params.iloc[0, 9]
            Z11[jj, ii] = params.iloc[0, 10]
            Z12[jj, ii] = params.iloc[0, 11]
            Z13[jj, ii] = params.iloc[0, 12]

        except Exception as e:
            print('Error in surrogate model', e)
            params = pd.DataFrame(np.nan*np.array(params), columns = df.columns[0:14])
            variance = pd.DataFrame(np.nan*np.array(variance), columns = df.columns[0:14])
            
            predicted_params = pd.concat([predicted_params, params])
            predicted_variances = pd.concat([predicted_variances, variance])

            Z1[jj, ii] = np.nan
            Z2[jj, ii] = np.nan
            Z3[jj, ii] = np.nan
            Z4[jj, ii] = np.nan
            Z5[jj, ii] = np.nan
            Z6[jj, ii] = np.nan
            Z7[jj, ii] = np.nan
            Z8[jj, ii] = np.nan
            Z9[jj, ii] = np.nan
            Z10[jj, ii] = np.nan
            Z11[jj, ii] = np.nan
            Z12[jj, ii] = np.nan
            Z13[jj, ii] = np.nan

            continue

Zvals = [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12, Z13]

# Plot the surfaces
fig = plt.figure()
for ii in range(13):
    code = '35'+str(ii+1)
    ax = fig.add_subplot(3, 5, ii+1, projection='3d')
    ax.plot_surface(X, Y, Zvals[ii], cmap='viridis', alpha=0.8)
    ax.set_xlabel('ar')
    ax.set_ylabel('smr')
    # ax.set_zlabel(df.columns[ii])
    ax.set_title(df.columns[ii])
    # Add the data points
    ax.scatter(xvals[indices_flexure], yvals[indices_flexure], df.iloc[indices_flexure, ii], c='b', marker='o', label='Cal-Flex')
    ax.scatter(xvals[indices_shear], yvals[indices_shear], df.iloc[indices_shear, ii], c='r', marker='o', label='Cal-Shear')

plt.tight_layout()
plt.show()
