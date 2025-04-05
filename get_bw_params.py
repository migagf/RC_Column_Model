from gpPredict import *
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import os

import pickle

from run_column_model import *

# Load pickle file with logistic regression
with open('log_reg_model.pkl', 'rb') as f:
    failure_mode_selection = pickle.load(f)
    # First parameter is the aspect ratio
    # Second parameter is Vp/Vs

    # Codes:
    # 0 = Flexure
    # 1 = Shear

# Use latex for plots
'''plt.rc('text', usetex=True)
plt.rc('font', family='serif')'''
# ndParams = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']
# ndParams = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

def get_BW_params(ndParams, gp_number=0, mode='simple'):

    # Sort parameters to format of gpPredict
    params_list = [
            ["RV_column1", ndParams[0]],
            ["RV_column2", ndParams[1]],
            ["RV_column3", ndParams[2]],
            ["RV_column4", ndParams[3]],
            ["RV_column5", ndParams[4]],
            ["RV_column6", ndParams[5]]
        ]
    
    # Define the failure mode using the logistic regression model
    if mode == 'simple':
        failure_mode = failure_mode_selection.predict(np.array([[ndParams[0], ndParams[5]]]))[0]
    
        # Define the surrogate_file and input_json based on the failure mode
        if failure_mode == 0:
            print('Flexure failure mode')
            surrogate_file = os.path.join('gpModelFlexure', f'SimGpModel{str(gp_number)}.json')
            input_json = os.path.join('gpModelFlexure', 'scInput.json')
        else:
            print('Shear failure mode')
            surrogate_file = os.path.join('gpModelShear', f'SimGpModel{str(gp_number)}.json')
            input_json = os.path.join('gpModelShear', 'scInput.json')

        output = main(params_list, [], surrogate_file, 'dummyout.out', input_json)
        
        # bw model parameters are the all the output parameters except the last one
        bw_model_params = (output[0][:-1].T).tolist()

        # min error is the last output parameter
        min_error = float(output[0][-1])
    
    else:
        # Predict the probability of each failure mode
        prob_failure_mode = failure_mode_selection.predict_proba(np.array([[ndParams[0], ndParams[5]]]))[0]

        # get output parameters for each failure mode
        flex_surrogate_file = os.path.join('gpModelFlexure', f'SimGpModel{str(gp_number)}.json')
        flex_input_json = os.path.join('gpModelFlexure', 'scInput.json')

        shear_surrogate_file = os.path.join('gpModelShear', f'SimGpModel{str(gp_number)}.json')
        shear_input_json = os.path.join('gpModelShear', 'scInput.json')

        # Flexure
        flex_output = main(params_list, [], flex_surrogate_file, 'dummyout.out', flex_input_json)
        flex_bw_model_params = (flex_output[0][:-1].T).tolist()
        flex_min_error = float(flex_output[0][-1])

        # Shear
        shear_output = main(params_list, [], shear_surrogate_file, 'dummyout.out', shear_input_json)
        shear_bw_model_params = (shear_output[0][:-1].T).tolist()
        shear_min_error = float(shear_output[0][-1])

        # Weight the output parameters based on the probability of each failure mode
        bw_model_params = [flex_bw_model_params[i] * prob_failure_mode[0] + shear_bw_model_params[i] * prob_failure_mode[1] for i in range(len(flex_bw_model_params))]
        min_error = flex_min_error * prob_failure_mode[0] + shear_min_error * prob_failure_mode[1]

        failure_mode = failure_mode_selection.predict(np.array([[ndParams[0], ndParams[5]]]))[0]


    return bw_model_params, min_error, failure_mode