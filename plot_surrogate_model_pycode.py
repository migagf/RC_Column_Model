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
f_surrogate_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\quoFEM_Surrogate\flexure_pd_010_just_good_data'
f_template_dir = os.path.join(f_surrogate_dir, 'templatedir_SIM')
f_surrogate_file = f_surrogate_dir + r'\SimGpModel.json'

# ------
# Here, call surrogate model for shear failure
# ------
s_surrogate_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\quoFEM_Surrogate\combined_pd_005_just_good_data'
s_template_dir = os.path.join(f_surrogate_dir, 'templatedir_SIM')
s_surrogate_file = s_surrogate_dir + r'\SimGpModel.json'

# Input_json is the same for both cases
input_json = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\quoFEM_Surrogate\scInput.json'

# For all the cases in the dataset, get BW model predictions from the surrogate
predicted_params = pd.DataFrame(columns = df.columns[0:14])
predicted_variances = pd.DataFrame(columns = df.columns[0:14])

for ii in range(len(df)):
    name = df['name'][ii]
    
    # Extract the calibrated and nondimensional parameters for both training and testing
    cal_params = df.iloc[ii, 0:14]
    nondim_params = df.iloc[ii, 17:23]
    
    params_list = [["RV_column1", nondim_params.iloc[0]],
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
            print('Flexure-type failure')
            surrogate_file = f_surrogate_file
        else:
            print('Shear-type failure')
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
zvals = df.iloc[:, par_index]

# Surrogate prediction
bwpar = predicted_params.iloc[:, par_index]

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xvals, yvals, bwpar, c='r', marker='o', label='Surrogate-Predicted')
ax.scatter(xvals, yvals, zvals, c='b', marker='o', label='Calibrated')
ax.set_xlabel('$V_s/V_f$')
ax.set_ylabel('Aspect Ratio')
ax.set_zlabel('Parameter #6')
plt.title('BW Model Parameter Comparison')
plt.legend()