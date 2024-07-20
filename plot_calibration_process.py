# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:35:18 2024

@author: Miguel
"""

import os
import matplotlib.pyplot as plt
import json
import pandas as pd

import numpy as np

# Test files
filesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data'
calfilesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data\calibration_files'

# quoFEM local work dir
localworkdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\quoFEM\LocalWorkDir\tmp.SimCenter'

alldirs = os.listdir(localworkdir)

with open(filesdir + '/test_260.json') as file:
    test_data = json.load(file)

force = np.array(test_data["data"]["force"])
disp = np.array(test_data["data"]["disp"])


for folder in alldirs:
    
    if folder.startswith('workdir'):
        
        try:
            # Access folder and extract results file
            filepath = localworkdir + r'/' + folder + '/results.out'
            
            sim_force = pd.read_csv(filepath, header=None)
            
            # Interpolate the force and displacements from the experimental data
            
            int_disp = np.interp(np.linspace(0, len(disp), len(sim_force)), np.arange(len(disp)), disp)
            
            plt.figure(dpi=300)
            plt.plot(disp, force, 'k')
            plt.plot(int_disp, sim_force, 'r.--')
        except:
            
            pass