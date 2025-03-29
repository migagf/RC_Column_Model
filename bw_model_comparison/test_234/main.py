#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:20:55 2024

@author: miguelgomez
"""

import os
import numpy as np
# import pandas as pd
import time
import json

# Import functions to create model
from material_models import *
from structure_model import *
from utilities import *
from model_parameters import *

# from buildModel import *
# from prepare_files import getNDparams

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman"})

# Location of the experimental data
# filesdir = r'/Users/miguelgomez/Documents/GitHub/RC_Column_Model/test_data'

# Define location of the calibration files
#filesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data'
#calfilesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data\calibration_files'

do_plots = False

if do_plots:
    import matplotlib.pyplot as plt

g = 386  #in/s2

if __name__ == "__main__":
    
    # Load an experiment from the PEER performance database
    #with open(filesdir + '/test_' + str(testid).zfill(3) + '.json') as file:
    with open(os.path.join(os.getcwd(), 'test_file.json')) as file:
        test_data = json.load(file)
    
    # Define the elastic properties of the column
    E, I, L = get_elastic_properties(test_data)
    
    # Stiffness and strength of the plastic hinge
    stiffness =  3 * E * I / L   # kN-mm
    strength = 1000 * np.max(test_data["data"]["force"]) * L
    
    # Stiffness and strength of the plastic hinge (no need to update)
    k0 = kappa_k * stiffness
    sy0 = kappa * strength

    # Put them all in a list
    n = np.floor(n)
    deg_bw_params = [eta1, k0, sy0, sig, lam, mup, sigp, rsmax, n, alpha, alpha1, alpha2, betam1]
    
    # Create Plastic Hinge
    my_ph = deg_bw_material_mod(deg_bw_params)
    
    # Elastic Parameters, mass and additional damping
    el_params = [gamma * E * I, L]   # [EI, L]
    mass = np.array([522, 1.0, 3.62e6])   # kips/g
    damping = [0.01]

    # Create structural model
    model = structure_model(el_params, my_ph, mass, damping)
    
    # Define the strains for the pushover analysis
    strains = np.array(test_data["run_data"]["disp"])
    
    # Run the pushover analysis
    t0 = time.time()
    force = run_pushover(model, strains, plot=False, show_info=False)
    t1 = time.time()
    
    print('Finished... Run Time = ', t1-t0, 'sec')
    
    peak_force = np.max(force)

    if do_plots:     
        plt.figure()
        plt.plot(100 * np.array(test_data["data"]["disp"])/L, 1000 * np.array(test_data["data"]["force"])/peak_force, 'b--', label='exp')
        plt.plot(100 * np.array(strains)/L, np.array(force)/peak_force, 'r-', linewidth=0.75, label='model')
        plt.xlabel('Drift Ratio (%)')
        plt.ylabel('Lateral Force')
        plt.legend()
        plt.grid()
        plt.show() 

    # Save the response into response.out
    force = np.array(force)/1000
    
    # Get npts in the calibration data
    npts = test_data['cal_data']['npts']
    
    interpolated_force = interpolator(force, npts)
    interpolated_displacement = interpolator(strains, npts)


    '''plt.figure()
    plt.plot(strains, force, 'k-', linewidth=0.5)
    plt.plot(interpolated_displacement, interpolated_force, 'r.')
    plt.savefig('hysteresis.png')'''
    # Compute mean absolute error
    error = np.mean(np.abs(np.array(test_data["cal_data"]["force"]) - interpolated_force))/np.max(test_data["cal_data"]["force"])

    # Save response to file
    save_response(filename='results.out', array=interpolated_force, save_type='column')
    save_response(filename='mae.out', array=error, save_type='column')


