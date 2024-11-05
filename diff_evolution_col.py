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
from scipy import integrate

# Import functions to create model
from column_model.material_models import *
from column_model.structure_model import *
from column_model.utilities import *
from column_model.model_parameters import *

# from buildModel import *
# from prepare_files import getNDparams

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman"})

from scipy.optimize import differential_evolution

# Location of the experimental data
# filesdir = r'/Users/miguelgomez/Documents/GitHub/RC_Column_Model/test_data'

# Define location of the calibration files
#filesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data'
#calfilesdir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data\calibration_files'

do_plots = True

if do_plots:
    import matplotlib.pyplot as plt

g = 386  #in/s2

import shutil

# Bring relevant files into the model directory
test_files_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data'
# model_files_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\column_model'
model_files_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model'
cal_file_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model'

def compute_error(model, data):

    err = np.sqrt(np.sum((model - data)**2) / len(data))

    if np.isnan(err):
        err = 100
    
    return err


def best_fit_parameters(optimum_params):
    '''
    This function turns the optimum_params stored as a list into a dictionary
    with the parameter names as keys
    '''

    best_fit_dict = {
        'eta1': optimum_params[0],
        'kappa_k': optimum_params[1],
        'kappa': optimum_params[2],
        'sig': optimum_params[3],
        'lam': optimum_params[4],
        'mup': optimum_params[5],
        'sigp': optimum_params[6],
        'rsmax': optimum_params[7],
        'n': optimum_params[8],
        'alpha': optimum_params[9],
        'alpha1': optimum_params[10],
        'alpha2': optimum_params[11],
        'betam1': optimum_params[12],
        'gamma': optimum_params[13]
    }

    return best_fit_dict


def compute_loss(model, experiment, type='force'):
    '''
    This function computes a loss function between the model and the experimental data
    
    '''
    # Force from model and experiment
    disp_e = experiment["disp"]
    force_e = experiment["force"] / max(experiment["force"])

    disp_m = model["disp"]
    force_m = model["force"] / max(experiment["force"])

    #plt.figure()
    #plt.plot(disp_m, force_m, 'r-', label='Model')
    #plt.plot(disp_e, force_e, 'k-', label='Experiment')

    # Integrate to find the energy dissipated for model and experiment
    if type == 'force':
        # Don't calculate anything here
        pass
    else:   
        # If want to also calibrate energy, calculate the energy
        energy_m = integrate.cumulative_trapezoid(force_m, disp_m, initial=0)
        energy_e = integrate.cumulative_trapezoid(force_e, disp_e, initial=0)

    # Compute loss function
    if type == 'force':
        # Disregard the energy in the loss function
        loss = np.sqrt(np.sum((force_m - force_e)**2) / len(force_e))
    else:
        # Include energy in the loss function (need to calibrate the hyperparameter)
        loss = 0.0001 * np.sqrt(np.sum((energy_m - energy_e)**2) / len(energy_e)) + np.sqrt(np.sum((force_m - force_e)**2) / len(force_e))
    
    # print(loss)
    if np.isnan(loss):
        loss = 100

    return loss


def get_residual(ModelParams, test_data, show_plots=False):
    '''
    Compute the residual between the model and the experimental data
    
    '''
    peak_dr = 0.05   # Max drift ratio of analysis

    eta1 = ModelParams[0]
    kappa_k = ModelParams[1]
    kappa = ModelParams[2]
    sig = ModelParams[3]
    lam = ModelParams[4]
    mup = ModelParams[5]
    sigp = ModelParams[6]
    rsmax = ModelParams[7]
    n = ModelParams[8]
    alpha = ModelParams[9]
    alpha1 = ModelParams[10]
    alpha2 = ModelParams[11]
    betam1 = ModelParams[12]
    gamma = ModelParams[13]
    
    # Define the elastic properties of the column
    E, I, L = get_elastic_properties(test_data)
    
    # Stiffness and strength of the plastic hinge
    stiffness =  5 * 3 * E * I / L   # kN-mm
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
    strains = np.array(test_data["data"]["disp"])

    # Cut the strains up to 5% drift ratio
    index = np.array(np.where(np.abs(strains) >= peak_dr * L))
    # Check if index is empty
    if index.size == 0:
        index_min = len(strains)
    else:
        index_min = np.min(index)

    strains = strains[0:index_min]
    
    #print(len(strains))
    #strains = interpolator(strains, npts)
    # print(len(strains))

    # Define cycles for pushover
    #t0 = time.time()
    force_model = run_pushover(model, strains, plot=False, show_info=False)
    #t1 = time.time()

    # print('Finished... Run Time = ', t1-t0, 'sec')
    force_exp = 1000 * np.array(test_data["data"]["force"])
    
    # Cut the vector of forces so it has the same length as the strains
    force_exp = force_exp[0:index_min]
    peak_force = np.max(force_exp)

    if show_plots:
        # Sub-sample force_exp and force_m to 50% of the total length
        npts = len(strains)
        red_npts = int(0.5 * npts)

        force_model_int = interpolator(force_model, red_npts) / peak_force
        force_exp_int = interpolator(force_exp, red_npts) / peak_force

        plt.figure()
        plt.plot(force_exp_int, 'k.', label='Experimental')
        plt.plot(force_model_int, 'r.', label='Model')
        # plt.show()
        plt.savefig(os.getcwd() + r'/plots/force_test'+str(test_id).zfill(3)+'.pdf')
        plt.close()

    # residual = compute_error(force_exp_int, force_model_int)
    model_data = {"disp": strains, "force": force_model}
    exp_data = {"disp": strains, "force": force_exp}

    residual = compute_loss(model_data, exp_data, type='all')

    if show_plots:     
        plt.figure()
        plt.plot(100 * np.array(strains)/L, force_exp/peak_force, 'b--', label='exp')
        plt.plot(100 * np.array(strains)/L, np.array(force_model)/peak_force, 'r-', linewidth=0.75, label='model')
        plt.xlabel('Drift Ratio (%)')
        plt.ylabel('Lateral Force')
        plt.legend()
        plt.grid()
        #plt.show()
        plt.savefig(os.getcwd() + r'/plots/plot_test'+str(test_id).zfill(3)+'.pdf')
        plt.close()

    # Save the response into response.out
    # force = np.array(force_model)/1000

    # Check if the residual is nan...
    if np.isnan(residual):
        residual = 1000

    return residual


def run_model(ModelParams, test_data):
    '''
    Run model and compute the residual between the experimental and the numerical data
    for both the force and the energy
    
    '''
    # Define the max drift ratio for analysis
    peak_dr = 0.05
    
    # Get model parameters from ModelParams list
    eta1 = ModelParams[0]
    kappa_k = ModelParams[1]
    kappa = ModelParams[2]
    sig = ModelParams[3]
    lam = ModelParams[4]
    mup = ModelParams[5]
    sigp = ModelParams[6]
    rsmax = ModelParams[7]
    n = ModelParams[8]
    alpha = ModelParams[9]
    alpha1 = ModelParams[10]
    alpha2 = ModelParams[11]
    betam1 = ModelParams[12]
    gamma = ModelParams[13]
    
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
    
    #% Create Plastic Hinge
    my_ph = deg_bw_material_mod(deg_bw_params)
    
    # Elastic Parameters, mass and additional damping
    el_params = [gamma * E * I, L]   # [EI, L]
    mass = np.array([522, 1.0, 3.62e6])   # kips/g
    damping = [0.01]

    # Create structural model
    model = structure_model(el_params, my_ph, mass, damping)
    
    # Define the strains for the pushover analysis
    strains = np.array(test_data["data"]["disp"])
    
    # Cut the strains up to 5% drift ratio
    index = np.array(np.where(np.abs(strains) >= peak_dr * L))

    # Check if index is empty
    if index.size == 0:
        index_min = len(strains)
    else:
        index_min = np.min(index)

    strains = strains[0:index_min]

    # Define cycles for pushover
    #t0 = time.time()
    force_model = run_pushover(model, strains, plot=False, show_info=False)
    #t1 = time.time()

    # print('Finished... Run Time = ', t1-t0, 'sec')
    force_exp = 1000 * np.array(test_data["data"]["force"])

    # Cut the vector of forces so it has the same length as the strains
    force_exp = force_exp[0:index_min] 

    peak_force_exp = np.max(np.abs(force_exp))
    peak_force_model = np.max(np.abs(force_model))
    # residual = compute_error(force_exp_int, force_model_int)
    model_data = {"disp": np.array(strains) / L, "force": np.array(force_model) / peak_force_model}
    exp_data = {"disp": np.array(strains) / L, "force": np.array(force_exp) / peak_force_exp}

    return model_data, exp_data


def smooth_data(non_smoothed_data, npts=10, do_plots=False):
    '''
    Smooth the data using a moving average of npts
    '''
    # Get the force and displacement data
    force = np.array(non_smoothed_data["force"])
    disp = np.array(non_smoothed_data["disp"])

    # Smooth the data using a moving average
    force_smoothed = np.convolve(force, np.ones((npts,))/npts, mode='valid')
    disp_smoothed = np.convolve(disp, np.ones((npts,))/npts, mode='valid')

    # Plot the smoothed data and the original data
    if do_plots:
        plt.figure()
        plt.plot(disp, force, 'k-', linewidth=0.5)
        plt.plot(disp_smoothed, force_smoothed, 'r--', linewidth=0.5)
        plt.show()

    return {"disp": disp_smoothed, "force": force_smoothed}

if __name__ == "__main__":
    # Run the calibrations for all tests

    maxID = 417
    fullrange = range(241, maxID + 1)

    for id in fullrange:
        test_id = str(id).zfill(3)
        try:
            # Copy the test file
            shutil.copyfile(os.path.join(test_files_dir, f'test_{test_id}.json'), os.path.join(model_files_dir, 'test_file.json'))

            # Copy the calibration file
            #shutil.copyfile(os.path.join(test_files_dir, f'cal_{test_id}.csv'), os.path.join(cal_file_dir, 'calibration_file.csv'))
            #shutil.copyfile(os.path.join(test_files_dir, f'cal_{test_id}.csv'), os.path.join(model_files_dir, 'cal_file.csv'))

            # Load an experiment from the PEER performance database
            #with open(filesdir + '/test_' + str(testid).zfill(3) + '.json') as file:
            with open(os.path.join(os.getcwd(), 'test_file.json')) as file:
                test_data = json.load(file)
            
            # Smooth the data using a moving average
            test_data["data"] = smooth_data(test_data["data"], npts=20, do_plots=True)

            # Get number of points in the calibration data file
            #with open(calfilesdir + '/cal_' + str(testid).zfill(3) + '.csv') as file:
            #with open(os.path.join(os.getcwd(), 'cal_file.csv')) as file:
            #    cal_data = np.genfromtxt(file, delimiter=',')
            #    npts = cal_data.shape[0]
            #    print('Calibration file has', npts, 'points')

            test_data["data"] = get_effective_force(test_data)
            
            # ModelParams = [eta1, kappa_k, kappa, sig, lam, mup, sigp, rsmax, n, alpha, alpha1, alpha2, betam1]
            parameters = np.array([
                1.0, # eta1 [0.1, 10.0] Shape control
                5.0, # kappa_k [1.0, 10.0] Modifies the stiffness of the plastic hinge
                1.0, # kappa [0.95, 1.05] Modifies the strength of the plastic hinge
                0.6, # sig [0.02, 0.95] Pinching Parameter
                0.6, # lam [0.02, 0.95] Pinching Parameter
                1.0, # mup [0.1, 5.0] Pinching parameter 
                2.0, # sigp [0.1, 5.0] Pinching parameter
                0.8, # rsmax [0.01, 1.0] Pinching parameter
                2.0, # n [1.0, 10.0] Exponent
                0.0, # alpha [0.0, 0.05] Post yield stiffness
                10.0,  # alpha1 [0.5, 10.0] Stiffness degradation
                0.1, # alpha2 [0.01, 2.0] Shape control
                0.005, # betam1 [0.0, 0.05] Strength degradation
                2.0 # gamma [0.1, 2.0] Elastic stiffness modifier
                ])
            
            res_0 = get_residual(parameters, test_data, show_plots=True)
            print(res_0)
            
            bounds = [
                (0.5, 2.0),   # eta1 [0.1, 10.0] Shape control
                (0.95, 1.05),  # kappa_k [1.0, 5.0] Modifies the stiffness of the plastic hinge
                (0.95, 1.05), # kappa [0.95, 1.05] Modifies the strength of the plastic hinge
                (0.02, 0.95), # sig [0.02, 0.95] Pinching Parameter
                (0.02, 0.95), # lam [0.02, 0.95] Pinching Parameter
                (0.1, 10.0),   # mup [0.1, 5.0] Pinching parameter
                (0.1, 10.0),   # sigp [0.1, 5.0] Pinching parameter
                (0.01, 1.0),  # rsmax [0.01, 1.0] Pinching parameter
                (1.0, 10),    # n [1.0, 10.0] Exponent
                (0.0, 0.05),  # alpha [0.0, 0.05] Post yield stiffness
                (0.01, 10),    # alpha1 [0.5, 10.0] Stiffness degradation
                (0.01, 2.0),  # alpha2 [0.01, 2.0] Shape control
                (0.0, 0.05),  # betam1 [0.0, 0.05] Strength degradation
                (0.1, 1.0)    # gamma [0.1, 1.0] Elastic stiffness modifier
                ]
            
            #M_hist, H_hist, theta_all = run_model(parameters, ThetaIn)
            #plt.plot(theta_all, M_hist)
            #plt.show()

            # Run the optimization and time it
            start_time = time.time()
            optimum = differential_evolution(get_residual, args=(test_data, False), bounds=bounds, maxiter=8, popsize=32, disp=True, workers=20, polish=False)
            end_time = time.time()

            print(optimum.x)
            get_residual(optimum.x, test_data, show_plots=True)
            
            # Next lines are just for testing..

            #interpolated_force = interpolator(force, npts)
            #interpolated_displacement = interpolator(strains, npts)

            """
            plt.figure()
            plt.plot(strains, force, 'k-', linewidth=0.5)
            plt.plot(interpolated_displacement, interpolated_force, 'r.')
            plt.show()
            """
            # Save response to file
            #save_response(filename='results.out', array=interpolated_force, save_type='column')
            
            # Save the optimum parameters
            best_fit_dict = best_fit_parameters(optimum.x)

            test_data["best_fit"] = best_fit_dict

            with open(os.getcwd() + '/cals_dr_005/test_'+str(test_id).zfill(3)+'.json', 'w') as f:
                json.dump(test_data, f, indent=4)

        except Exception as e:
            print(f"Error processing test ID {test_id}: {e}")
            continue

