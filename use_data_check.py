# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 00:02:12 2024

@author: Miguel
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# Plot the calibrations in calibrated_curves_02
calibrated_data_folder = os.getcwd() + '/calibrated_curves_02/'
calibrated_data = os.listdir(calibrated_data_folder)

use_data = pd.DataFrame(columns=['id', 'name', 'use'])

for file in calibrated_data:
    if file.endswith('.json'):
        #
        with open(calibrated_data_folder + file, 'r') as f:
            test_data = json.load(f)

        # Plot the data
        disp = np.array(test_data['data']['disp'])
        force = np.array(test_data['data']['force'])

        peak_force = max(abs(force))
        l = test_data['L_Inflection']
        
        norm_disp = disp / l
        norm_force = force / peak_force

        plt.figure()
        plt.plot(norm_disp, norm_force, label=file)

        # Here pause and ask the user if the curve looks good
        plt.show()
        print('Does the curve look good?')
        response = input('1/0: ')
        plt.close()
        # Store the response in the json file
        use_data.loc[len(use_data)] = [file[5:8], test_data['Name'], response]

use_data.to_csv('spiral_data_use.csv')
        
        