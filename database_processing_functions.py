#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:22:08 2024

@author: miguelgomez
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import scipy as sp


# Index of the functions in this file
# 1. save_csv
# 2. plot_hysteresis
# 3. load_json
# 4. create_calibration_file
# 5. get_effective_force
# 6. get_moment_rotation
# 7. save_normalized_hysteresis
# 8. get_backbone_curve
# 9. send_email
# 10. smooth_data

def save_json(data, filename):
    '''
    Function to save a dictionary as a JSON file
    '''
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def save_csv(filename, array, save_type='row'):
    '''
    Function to save to csv file for calibrations
    '''
    # if array is a list, convert to numpy array
    if type(array) == list:
        array = np.array(array)
    
    if save_type == 'row':
        # Save array as a row csv
        array.tofile(filename, sep=',', format='%10.5f')
        
    else:
        # Save array as a column csv
        array.tofile(filename, sep='\n', format='%10.5f')
        
    pass


def plot_hysteresis(disp, force, label):
    '''
    Function to plot the hysteresis loop
    disp: displacement
    force: force
    '''
    plt.plot(disp, force, label=label)
    
    pass


def load_json(json_dir):
    # Open JSON file and store as dictionary
    
    with open(json_dir, 'r') as file:
        data = json.load(file)
        
    return data


def interpolator(original_array, new_length):
    '''
    Function to interpolate an array to a new length

    '''
    # Get the original length
    original_length = len(original_array)
    
    # Create the new array
    interpolated_array = np.interp(np.linspace(0, original_length, new_length), np.arange(original_length), original_array)

    return interpolated_array


def smooth_data(non_smoothed_data, npts=5, dpts=0.05, do_plots=False):
    '''
    Smooth the data using a moving average of npts
    
    Inputs:
    non_smoothed_data: dictionary with keys 'disp' and 'force'
    npts: number of points for the moving average
    dpts: percentage of the peak force to start the curve
    do_plots: boolean to plot the smoothed data

    '''
    # Get the force and displacement data
    force = np.array(non_smoothed_data["force"])
    disp = np.array(non_smoothed_data["disp"])

    # Start displacement and force at 0
    disp = disp - disp[0]
    force = force - force[0]
    
    # Reduce the number of points in the data
    print('Original length', len(disp))
    
    # Count number of cycles as number of crosses per zero displacement
    indicator = disp[0:-1] * disp[1:] < 0
    nzeros = np.sum(indicator)

    # Increase the number of points by a factor of 10
    disp = interpolator(disp, 10*len(disp))
    force = interpolator(force, 10*len(force))

    # Delete initial values for which force is less than 5.0% of the peak force
    peak_force = np.max(force)

    index = np.array(np.where(force >= dpts * peak_force))
    index_min = np.min(index)
    disp = disp[index_min:]
    force = force[index_min:]
        
    # Smooth the data using a moving average
    force_smoothed = np.convolve(force, np.ones((npts,))/npts, mode='same')
    disp_smoothed = np.convolve(disp, np.ones((npts,))/npts, mode='same')
    
    # Delete first npts - 1 points
    disp_smoothed = disp_smoothed[npts - 1:]
    force_smoothed = force_smoothed[npts - 1:]

    # Start displacement and force at 0
    #disp_smoothed = (disp_smoothed - 0*disp_smoothed[0]).tolist()
    #force_smoothed = (force_smoothed - 0*force_smoothed[0]).tolist()
    # Add zero at the beggining of disp_smoothed and force_smoothed
    # Center the displacements
    
    # Initial secant stiffness
    ini_st = (force_smoothed[0]) / (disp_smoothed[0])
    
    # Insert 10*npts equispaced points at the beggining of the disp_smoothed vector,
    # starting from zero and ending at the first value of disp_smoothed

    add_disp = np.linspace(0, disp_smoothed[0], 10*npts)
    add_force = ini_st * add_disp
    disp_smoothed = np.concatenate((add_disp, disp_smoothed))
    force_smoothed = np.concatenate((add_force, force_smoothed))

    # Delete the last 10*npts points
    disp_smoothed = disp_smoothed[:-10*npts]
    force_smoothed = force_smoothed[:-10*npts]

    # Go back to the original length
    disp_smoothed = interpolator(disp_smoothed, int(len(disp_smoothed)/10))
    force_smoothed = interpolator(force_smoothed, int(len(force_smoothed)/10))

    # Turn into lists
    disp_smoothed = disp_smoothed.tolist()
    force_smoothed = force_smoothed.tolist()
    print('Final length', len(disp_smoothed))

    # Plot the smoothed data and the original data
    if do_plots:
        plt.figure()
        plt.plot(disp, force, 'k-', linewidth=0.1)
        plt.plot(disp_smoothed, force_smoothed, 'r-', linewidth=0.1)
        plt.plot(disp_smoothed[0:20], force_smoothed[0:20], 'r.-', markersize=2.0, linewidth=1.0)
        plt.grid()
        plt.show()

    return {"disp": disp_smoothed, "force": force_smoothed}


def send_email(message):
    '''
    Send an email with the message to my bot in Telegram
    '''
    import requests
    # Get token from file
    with open(r"C:\Users\Miguel.MIGUEL-DESK\Documents\myfile.txt") as f:
        token = f.read()

    url = f"https://api.telegram.org/bot{token}"
    params = {"chat_id": "7619956282", "text": message}
    r = requests.get(url + "/sendMessage", params=params)


def create_calibration_file(test_data, test_id, destination, plot=False, save_cal=False):
    
    state = 1
    
    try:
        # Smooth the force-displacement data. Use 1 point for the moving average.
        smoothed_data = smooth_data(test_data["data"], do_plots=True)
        is_good = input("Is the data good? (y/n): ")

        while is_good != 'y':
            npts = int(input("Enter the number of points for the moving average: "))
            dpts = float(input("Enter the percentage of the peak force to start the curve: "))
            smoothed_data = smooth_data(test_data["data"], npts=npts, dpts=dpts, do_plots=True)
            is_good = input("Is the data good? (y/n): ")

        disp = np.array(smoothed_data["disp"])
        force = np.array(smoothed_data["force"])
        
        # Get total number of points and create "time"
        npts = len(disp)
        tt = np.arange(0, npts)
        
        #plt.figure(dpi=500)
        #plt.plot(disp, force)
        
        #plt.figure(dpi=500)
        #plt.plot(disp)
        
        # Run through displacement values and get crosses by zero
        zero_cross = 0
        
        for ii in range(0, len(disp)-1):
            if disp[ii+1] * disp[ii] < 0:
                zero_cross += 1
        
        print('Have ', zero_cross, 'crosses by zero')
        
        # Create interpolation objects for force and displacement
        disp_int = sp.interpolate.interp1d(tt, disp)
        force_int = sp.interpolate.interp1d(tt, force)
        
        # Interpolation for calibration
        cal_tt = np.linspace(0, npts-1, 11 * zero_cross) # 11 points per crossing by zero
        cal_disp = disp_int(cal_tt)
        cal_force = force_int(cal_tt)
        
        # Interpolation for running the analysis
        run_tt = np.linspace(0, npts-1, 110 * zero_cross) # 110 points per crossing by zero
        run_disp = disp_int(run_tt)
        run_force = force_int(run_tt)
        
        # Convert to lists
        cal_disp = (cal_disp).tolist()
        cal_force = (cal_force).tolist()
        run_disp = (run_disp).tolist()
        run_force = (run_force).tolist()

        # Create dictionaries with the calibration and running data
        cal_data = {
            'disp': cal_disp,
            'force': cal_force,
            'npts': len(cal_disp)
        }
        
        print('for quoFEM, the number of points is ', len(cal_disp))

        run_data = {
            'disp': run_disp,
            'force': run_force,
            'npts': len(run_disp)
        }

        if plot:
            # Plot displacement-force curve
            plt.figure(dpi=200)
            plt.plot(np.array(disp)/test_data['L_Inflection'], force, 'k-', linewidth=0.05)
            plt.plot(np.array(cal_disp)/test_data['L_Inflection'], cal_force, 'r.-', linewidth=0.1, markersize=0.5)
            plt.title(test_id)
            #plt.xlim(-0.05, 0.05)
            #plt.ylim(-350, 350)
            plt.show()

            # Plot force vs time
            plt.figure(dpi=200)
            plt.plot(tt, force, 'k-', linewidth=0.05, label='Raw Data')
            plt.plot(cal_tt, cal_force, 'r.-', linewidth=0.1, markersize=0.5, label='For Calibration')
            plt.title(test_id)
            plt.show()

            plt.figure(dpi=200)
            plt.plot(run_disp, run_force, 'b.', markersize=0.8)
            plt.plot(disp, force, 'r.-', linewidth=0.1, markersize=0.5)
            plt.show()
        else:
            pass

        if save_cal:
            # Save the calibration file as column file
            save_csv(destination + 'cal_' + test_id + '.csv', cal_force, save_type='row')
                 
    except Exception as que_paso:
        print('Problem encountered when trying to save force-deformation \n', que_paso)
        state = 0
    
    return state, cal_data, run_data


def get_effective_force(test_data, plot=False):
    '''
    This function takes the test data, and computes the effective force by eliminating the P-Delta effect

    '''

    # Check if the P-Delta effect is present
    if test_data['P_Delta'] == 'Shear provided':
        print('Need to compute effective force')     

        # Get the force-displacement data
        force = np.array(test_data['data']['force'])
        disp = np.array(test_data['data']['disp'])
        
        # Compute the effective force
        effective_force = force + disp * test_data['AxLoad'] / test_data['L_Inflection']

        # Update the data dictionary
        test_data['data']['force'] = effective_force.tolist()

        if plot:
            plt.figure()
            plot_hysteresis(disp, effective_force, 'Effective Force')
            plot_hysteresis(disp, force, 'Original Force')
            plt.legend()
            plt.title(test_data['P_Delta'])
            plt.show()
        
        # Update the 'P_Delta' field to 'Feff computed'
        test_data['P_Delta'] = 'Feff computed'
    else:
        # Feff directly reported. Do nothing
        print("Effective force was directly reported, no need to do anything")
        pass

    return test_data['data']


def get_moment_rotation(test_data, plot=False):
    '''
    This function computes the moment-rotation curve of the column test
    '''

    # Get some properties
    Ec = 4700 * np.sqrt(test_data['fpc']) # Concrete modulus of elasticity (MPa)
    fpc = test_data['fpc'] # Concrete compressive strength (MPa)
    fy = test_data['fyl'] # Steel yield strength (MPa)
    dlb = test_data['dlb'] # Longitudinal bar diameter (mm)

    li = test_data['L_Inflection'] # Inflection point (mm)

    if test_data['Type'] == 'Spiral':
        # Calculation for circular column
        D = test_data['Diameter']  # Diameter of the column (mm)
        Ig = np.pi * D**4 / 64 # Gross moment of inertia (mm^4)

    else:
        # Calculation for rectangular column
        b = test_data['Width']  # Width of the column (mm)
        h = test_data['Depth'] # Height of the column (mm)
        Ig = b * h**3 / 12 # Gross moment of inertia (mm^4)
    
    # Compute plastic hinge length
    lp = 0.05 * li + 0.1 * dlb * fy / np.sqrt(fpc) # Plastic hinge length (mm)

    # Length of elastic portion of the column
    le = li - lp / 2 # Elastic length (mm)

    # Compute the elastic stiffness
    elastic_stiffness = (3 * Ec * Ig / le ** 3) / 1000 # Elastic stiffness (kN/mm)

    # Substract elastic deformation and compute rotation attributed to hinge
    el_def = np.array(test_data['data']['force']) / elastic_stiffness # Elastic deformation (mm)
    rotation = (np.array(test_data['data']['disp']) - el_def) / le # Rotation (rad)
    moment = np.array(test_data['data']['force']) * le

    if plot:
        plt.figure()
        plot_hysteresis(np.array(test_data['data']['disp']) - el_def, moment, 'Moment-Rotation')
        plot_hysteresis(np.array(test_data['data']['disp']), moment, 'Moment-Rotation')
        plt.xlabel('Rotation (rad)')
        plt.ylabel('Moment (kN.mm)')
        plt.title(test_data['P_Delta'])
        plt.show()

    moment_rotation = {
        'disp': rotation,
        'force': moment
        }
    
    return moment_rotation, elastic_stiffness


def save_normalized_hysteresis(normalized_hyst, filename='none', npts=10, plot=False):
    '''
    This function saves the normalized hysteresis curve to a csv file
    '''
    try:
        # Count the number of crosses by zero force in normalized hysteresis
        n_crosses = 0
        for ii in range(1, len(normalized_hyst['force'])):
            if normalized_hyst['force'][ii] * normalized_hyst['force'][ii-1] < 0:
                n_crosses += 1
        
        # Downsample the hysteresis curve to npts * n_crosses points
        downsampled_disp = np.interp(np.linspace(0, len(normalized_hyst['disp']), npts * n_crosses), np.arange(0, len(normalized_hyst['disp'])), normalized_hyst['disp'])
        downsampled_force = np.interp(np.linspace(0, len(normalized_hyst['force']), npts * n_crosses), np.arange(0, len(normalized_hyst['force'])), normalized_hyst['force'])

        df = pd.DataFrame(normalized_hyst)
        df.to_csv(filename, index=False)
        state = 1

        if plot:
            plt.figure()
            plt.plot(downsampled_disp, downsampled_force, 'k.-', linewidth=0.2)
            plt.show()
        
    except:
        state = 0

    return state


def get_backbone_curve(cyclic_test, plot=False):
    '''
    This function computes the backbone curve of a force-rotation pair test
    '''
    # Load displacement and force data
    disp = np.array(cyclic_test['disp'])
    force = np.array(cyclic_test['force'])

    time = np.arange(0, len(disp))
    newtime = np.linspace(0, len(disp), 10_000)

    disp = np.interp(newtime, time, disp)
    force = np.interp(newtime, time, force)

    # Initialize Backbone Curve
    backbone_disp = [0]
    backbone_force = [0]

    for ii in range(0, len(disp)):
        if disp[ii] > backbone_disp[-1]:
            backbone_force.append(force[ii])
            backbone_disp.append(disp[ii])
    
    # Apply smoothing using moving average
    window_size = 5

    # Pad backbone curve with window_size zeroes
    backbone_disp = [0] * window_size + backbone_disp
    backbone_force = [0] * window_size + backbone_force

    # Smooth backbone
    smoothed_disp = np.convolve(backbone_disp, np.ones(window_size)/window_size, mode='valid')
    smoothed_force = np.convolve(backbone_force, np.ones(window_size)/window_size, mode='valid')

    # Update the backbone curve with the smoothed data
    backbone = {
        'disp': smoothed_disp,
        'force': smoothed_force 
    }

    # Compute yield point
    yield_force = np.argmax(smoothed_force >= 0.80 * np.max(smoothed_force))
    yield_disp = smoothed_disp[yield_force]

    yield_point = {
        'disp': yield_disp,
        'force': 0.80 * np.max(smoothed_force)
    }

    #if plot:
    #    plt.figure()
    #    plt.plot(disp, force, 'k--', label='Force-Displacement')
    #    plt.plot(backbone['disp'], backbone['force'], 'r-', label='Backbone Curve')
    #    plt.xlabel('Displacement (mm)')
    #    plt.ylabel('Force (kN)')
    #    plt.title('Backbone Curve')
    #    plt.plot(yield_disp, 0.8 * np.max(smoothed_force), 'ro', label='Yield Point')
    #    plt.show()

    # Compute normalized displacement and force and smooth it
    window_size = 10
    cyc_disp = [0] * window_size + (disp / yield_disp).tolist()
    cyc_force = [0] * window_size + (force / np.max(smoothed_force)).tolist()
    sm_cyc_disp = np.convolve(cyc_disp, np.ones(window_size)/window_size, mode='valid')
    sm_cyc_force = np.convolve(cyc_force, np.ones(window_size)/window_size, mode='valid')
    
    # Put in a dictionary
    normalized_hyst = {
        'disp': sm_cyc_disp,
        'force': sm_cyc_force
    }

    if plot:
        plt.figure()
        plt.plot(normalized_hyst['disp'], normalized_hyst['force'], 'k-', label='Backbone Curve')
        plt.plot([1], [0.80], 'ro', label='Yield Point')
        plt.plot([0, 1], [0, 1], 'b--')
        plt.show()

    return backbone, yield_point, normalized_hyst
