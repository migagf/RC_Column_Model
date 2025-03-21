# Run model
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:20:55 2024

@author: miguelgomez
"""

import os
import numpy as np
import time
import json

# Import functions to create model
from column_model.material_models import *
from column_model.structure_model import *
from column_model.utilities import *


# Location of the experimental data
# filesdir = r'/Users/miguelgomez/Documents/GitHub/RC_Column_Model/test_data'

# Define location of the calibration files
#filesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data'
#calfilesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data\calibration_files'

do_plots = False

if do_plots:
    import matplotlib.pyplot as plt

g = 386  #in/s2


def run_model(test_data, bw_params, do_plots=False):

    # Parameter names:
    # 'gamma', 'kappa', 'eta1', 'sig', 'lam', 'mup', 'sigp', 'rsmax', 
    # 'alpha', 'alpha1', 'alpha2', 'betam1', 'n', 'kappa_k'
    
    # Get parameters from bw_params list
    # bw_params = [gamma, kappa, eta1, sig, lam, mup, sigp, rsmax, alpha, alpha1, alpha2, betam1, n, kappa_k]
    # Careful with the order of this values!!!

    gamma = bw_params[0]
    kappa = bw_params[1]
    eta1 = bw_params[2]
    sig = bw_params[3]
    lam = bw_params[4]
    mup = bw_params[5]
    sigp = bw_params[6]
    rsmax = bw_params[7]
    alpha = bw_params[8]
    alpha1 = bw_params[9]
    alpha2 = bw_params[10]
    betam1 = bw_params[11]
    n = bw_params[12]
    kappa_k = float(np.abs(bw_params[13]))

    # print(json.dumps(params_dict, indent=4))

    # Load an experiment from the PEER performance database
    #with open(filesdir + '/test_' + str(testid).zfill(3) + '.json') as file:
    '''with open(test_file) as file:
        test_data = json.load(file)'''
    
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
    
    peak_force = np.max(test_data["data"]["force"])

    # Compute drift ratios
    cal_dr = np.array(test_data["cal_data"]["disp"])/L
    exp_dr = np.array(test_data["data"]["disp"])/L
    sim_dr = np.array(test_data["run_data"]["disp"])/L

    cal_force = np.array(test_data["cal_data"]["force"])
    exp_force = np.array(test_data["data"]["force"])
    sim_force = force

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
 
    # Compute mean absolute error
    mean_err = np.mean(np.abs(np.array(test_data["cal_data"]["force"]) - interpolated_force)) / peak_force
    std_err = np.std(np.abs(np.array(test_data["cal_data"]["force"]) - interpolated_force)) / peak_force
    no_model_err = np.mean(np.abs(np.array(test_data["cal_data"]["force"]))) / peak_force # This assumes that the model gives 0 force for all displacements
    
    one_model_err = np.mean(np.abs(1 - np.abs(np.array(test_data["cal_data"]["force"]) / peak_force))) # This assumes that the model gives peak force for all displacements
    # Create a dictionary with the normalized results as lists
    
    results = {
        'err_data': {
            'mean_err': mean_err,
            'std_err': std_err,
            'no_model_err': no_model_err,
            'one_model_err': one_model_err
        },
        'cal_data': {
            'drift': cal_dr.tolist(),
            'nforce': (cal_force / peak_force / 1000).tolist()
        },
        'exp_data': {
            'drift': exp_dr.tolist(),
            'nforce': (exp_force / peak_force).tolist()
        },
        'sim_data': {
            'drift': sim_dr.tolist(),
            'nforce': (sim_force / peak_force / 1000).tolist()
        },
    }

    return results


if __name__ == '__main__':
    # test_data = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data\test_001.json'
    test_json = os.path.join('quoFEM_TMCMC', 'test_001', 'test_file.json')

    # Load test_data json file
    with open(test_json) as file:
        test_data = json.load(file)

    gamma      = 1.277177
    kappa      = 0.977073
    eta1       = 1.845268
    sig        = 0.685665
    lam        = 0.629793
    mup        = 1.650068
    sigp       = 0.779072
    rsmax      = 0.948117
    alpha      = 0.000115
    alpha1     = 4.961202
    alpha2     = 1.234721
    betam1     = 0.000328
    n          = 1.504958
    kappa_k    = 3.597515

    bw_params = [gamma, kappa, eta1, sig, lam, mup, sigp, rsmax, alpha, alpha1, alpha2, betam1, n, kappa_k]
    
    results = run_model(test_data, bw_params, do_plots=True)
