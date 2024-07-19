#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:20:55 2024

@author: miguelgomez
"""

# import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
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
filesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data'

g = 386  #in/s2

if __name__ == "__main__":
    
    # Load an experiment from the PEER performance database
    with open(filesdir + '/test_260.json') as file:
        test_data = json.load(file)
    
    test_data["data"] = get_effective_force(test_data)
    
    # Define the elastic properties of the column
    E, I, L = get_elastic_properties(test_data)
    
    # Stiffness and strength of the plastic hinge
    stiffness =  10 * 3 * E * I / L   # kN-mm
    strength = 1000 * np.max(test_data["data"]["force"]) * L
    
    # Stiffness and strength of the plastic hinge (no need to update)
    k0 = stiffness
    sy0 = strength

    # Put them all in a list
    deg_bw_params = [eta1, k0, sy0, sig, lam, mup, sigp, rsmax, n, alpha, alpha1, alpha2, betam1]
    
    #% Create Plastic Hinge
    my_ph = deg_bw_material(deg_bw_params)
    
    # Elastic Parameters, mass and additional damping
    el_params = [gamma * E * I, L]   # [EI, L]
    mass = np.array([522, 1.0, 3.62e6])   # kips/g
    damping = [0.01]

    # Create structural model
    model = structure_model(el_params, my_ph, mass, damping)
    
    # Define the strains for the pushover analysis
    strains = np.array(test_data["data"]["disp"])
    
    # Define cycles for pushover
    t0 = time.time()
    force = run_pushover(model, strains, plot=False)
    t1 = time.time()
    
    print('Finished... Run Time = ', t1-t0, 'sec')
    
    peak_force = np.max(force)

    plt.figure()
    plt.plot(100 * np.array(test_data["data"]["disp"])/L, 1000 * np.array(test_data["data"]["force"])/peak_force, 'b:')
    plt.plot(100 * np.array(strains)/L, np.array(force)/peak_force, 'r--')
    plt.xlabel('Drift Ratio (%)')
    plt.ylabel('Lateral Force')
    plt.show()
    
    
    

    
    
    
    
    
    
    
    
    

# %%
