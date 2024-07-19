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


def save_csv(filename, array, save_type='row'):
    ''' 
    Function to save to csv file
    '''
    
    if save_type == 'row':
        # Save array as a row csv
        array.tofile(filename, sep=',', format='%10.5f')
        
    else:
        # Save array as a column csv
        array.tofile(filename, sep='\n', format='%10.5f')
        
    pass


def plot_hysteresis(disp, force, label):
    '''
     
    '''
    plt.plot(disp, force, label=label)
    
    pass


def load_json(json_dir):
    # Open JSON file and store as dictionary
    
    with open(json_dir, 'r') as file:
        data = json.load(file)
        
    return data


def create_calibration_file(test_data, test_id, destination, plot=False):
    
    state = 1
    
    try:
        disp = np.array(test_data["data"]["disp"])
        force = np.array(test_data["data"]["force"])
        
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
        cal_tt = np.linspace(0, npts-1, 10 * zero_cross)
        cal_disp = disp_int(cal_tt)
        cal_force = force_int(cal_tt)
        
        plt.figure(dpi=500)
        plt.plot(disp, force, 'k-')
        plt.plot(cal_disp, cal_force, 'r.-')
        plt.title(test_id)
        
        # Interpolation for running the analysis
        run_tt = np.linspace(0, npts-1, 100 * zero_cross)
        run_disp = disp_int(run_tt)
        run_force = force_int(run_tt)
        
        #plt.figure(dpi=500)
        #plt.plot(run_disp, run_force, 'b.', markersize=0.8)
        #plt.plot(disp, force, 'r.-', linewidth=0.1, markersize=0.5)
        
        # Save the calibration file as row file
        save_csv(destination + 'cal_'+test_id+'.csv', run_disp, save_type='row')
        
                 
    except Exception as que_paso:
        print('Problem encountered when trying to save force-deformation \n', que_paso)
        state = 0
    
    return state


def get_effective_force(test_data, plot=False):
    '''
    This function takes the test data, and computes the effective force by getting rid of the P-Delta effect
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
        test_data['data']['force'] = effective_force

        if plot:
            plt.figure()
            plot_hysteresis(disp, effective_force, 'Effective Force')
            plot_hysteresis(disp, force, 'Original Force')
            plt.legend()
            plt.title(test_data['P_Delta'])
            plt.show()
    else:
        # Feff directly reported. Do nothing

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


if __name__ == "__main__":
    '''
    This code plots the force-displacement relations for the concrete column
    tests in the database
    
    '''
    # Get current folder
    current_folder = os.getcwd()

    # Folder with the JSON files
    json_dir = current_folder + '/test_data/'
   
    # Load the database
    data = pd.read_csv('data_spiral_wnd.csv')
    print(data)
    # For each curve:    

    for ii in range(0, len(data)):
        
        # (1) Create name of file
        test_id = str(data.id[ii]).zfill(3)
        filename = json_dir + 'test_' + test_id + '.json'
        
        # (2) Import JSON file as dictionary
        test_data = load_json(filename)

        # (3) Check P-Delta, and get effecttive force if needed
        test_data['data'] = get_effective_force(test_data, False)
        
        # (4) Save the effective force-displacement curves in their separate files
        create_calibration_file(test_data, test_id, destination=json_dir, plot=True)
        
        # 
        # (4) Get the elastic stiffness and substract elastic deformation
        # moment_rotation, elastic_stiffness = get_moment_rotation(test_data, plot=False)

        # (5) Get backbone curve
        # mr_backbone is the backbone of the moment-rotation
        # mr_backbone, yield_point, normalized_hyst = get_backbone_curve(moment_rotation, plot=True)
        
        # (6) Save the normalized hysteresis curve to a csv file

        # Create filename
        # filename = 'normalized_hysteresis/test_' + str(data.id[ii]).zfill(3) + '.csv'

        # Save the file
        #state = save_normalized_hysteresis(normalized_hyst, filename, npts=10)
        #if state == 0:
        #    print('Error saving file: ' + filename)
        #else:
        #    print('File saved: ' + filename)

        # To do:
        # Write the file with the nondimensional parameters
        # Write the file with the solver with quoFEM

        # df = pd.DataFrame(normalized_hyst)
        # print(df)
        # df.to_csv('normalized_hysteresis.csv', index=False)
        






        