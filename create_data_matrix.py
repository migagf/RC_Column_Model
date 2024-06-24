#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file creates a summary table with the relevant parameters of each test
the result is a csv file with the relevant processed data.
 
Created on Wed May 22 13:26:28 2024

@author: miguelgomez
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def do_pairplots(dataframe):
    plt.figure(dpi = 500)
    sns.pairplot(dataframe)
    pass


def load_json(json_dir):
    # Open JSON file and store as dictionary
    
    with open(json_dir, 'r') as file:
        data = json.load(file)
        
    return data


def get_spiral_props(rawdata, testID):
    
    props = []
    
    # String Data
    props.append(testID)
    props.append(rawdata['Name'])
    props.append(rawdata['Type'])
    props.append(rawdata['TestConfiguration'])
    props.append(rawdata['P_Delta'])
    props.append(rawdata['FailureType'])
    
    # Axial load
    props.append(rawdata['AxLoad'])
    
    # Overall Shape Data
    props.append(rawdata['Diameter'])
    props.append(rawdata['L_Inflection'])
    
    # Material Strength Properties
    props.append(rawdata['fpc'])
    props.append(rawdata['fyl'])
    props.append(rawdata['fsul'])
    
    props.append(rawdata['fyt'])
    props.append(rawdata['fsut'])
    
    # Reinforcement details
    props.append(rawdata['dlb'])
    props.append(rawdata['nlb'])
    props.append(rawdata['cc'])
    props.append(rawdata['dsp'])
    props.append(rawdata['s'])
    
    return props


def get_nd_params_sp(all_data):
    '''
    This function computes the nondimesional physical parameters
    for a spiral column
    
    Properties are:
     'id', 'name', 'type', 'testcf', 'pd', 'ft', 'axl', 'diam', 'l',
     'fpc', 'fyl', 'fsul', 'fyt', 'fsut', 'dlb', 'nlb',
     'cc', 'dsp', 's'
    '''
    ndparams = pd.DataFrame(columns=['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr'])
    
    for ii in range(0, len(all_data)):
        
        # (1) Extract the data for a single test
        test_data = all_data.iloc[ii, :]
        
        # (2) Aspect Ratio
        ar = test_data['diam'] / test_data['l']
        
        # (3) Longitudinal Reinforcement Ratio
        ag = np.pi * (test_data['diam']) ** 2 / 4   # gross area (mm2)
        alr = test_data['nlb'] * np.pi * (test_data['dlb']) ** 2 / 4 # long. reinf. area (mm2)
        rhol = alr / ag
        
        lrr = test_data['fyl'] * rhol / test_data['fpc']
        
        # (4) Spiral Reinforcement Ratio
        d_core = test_data['diam'] - 2 * test_data['cc']   # diameter of the core (mm)
        asp = np.pi * test_data['dsp'] ** 2 / 4  # area of spiral reinf (mm2)
        rhosp = 4 * asp / (np.pi * d_core ** 2)
        
        if test_data['fyt'] == 0:
            # fyt = 0, then use fyl
            srr = test_data['fyl'] * rhosp / np.sqrt(test_data['fpc'])
        else:
            srr = test_data['fyt'] * rhosp / np.sqrt(test_data['fpc'])
            
        # (5) Axial Load Ratio
        ax_cap = test_data['fpc'] * ag / 1000    # kN
        
        alr = test_data['axl'] / ax_cap
        
        # (6) Spiral diameter ratio
        sdr = test_data['dsp'] / test_data['dlb']
        
        # (7) Estimate of the shear strength and the moment strength
        Vs = get_shear_strength(test_data)
        Vm = get_moment_strength(test_data)
        
        smr = Vm/Vs
        
        # Append data at the end of dataframe
        ndparams.loc[len(ndparams)] = [ar, lrr, srr, alr, sdr, smr]
        
        
    return ndparams, pd.concat([all_data, ndparams], axis=1)


def get_shear_strength(test_data):
    '''
    Calculation of the shear strength based on 
    
    Sezen and Moehle (2004)
    
    '''
    
    # Get material properties
    fpc = test_data['fpc']   # (MPa)
    fyl = test_data['fyl']   # (MPa)
    fyt = test_data['fyt']   # (MPa)
    
    # Get geometry parameters
    diam = test_data['diam']  # (mm)
    length = test_data['l']   # (mm)
    
    dlb = test_data['dlb']    # (mm)
    nlb = test_data['nlb']    # (mm)
    
    dsp = test_data['dsp']    # (mm)
    s = test_data['s']        # (mm)

    # Get axial load
    p = test_data['axl'] * 1000   # (N)
    
    # Compute extra parameters
    d = 0.8 * diam     
    ag = np.pi * (diam) ** 2 / 4   # (mm2)
    
    # Mean shear stress at the onset of shear crack
    vc = 0.5 * np.sqrt(fpc) / (length / diam) * np.sqrt(1 + p / (0.5 * np.sqrt(fpc) * ag)) * 0.8
    
    # Concrete contribution to shear strenght
    Vc = vc * ag / 1000
    
    # Transverse reinforcement contribution to the shear strength
    #print('concrete contribution', Vc)
    
    av = 2 * np.pi * (dsp) ** 2 / 4
    
    if s == 0:
        Vs = 0
    else:
        Vs = (av * fyt * d / s) / 1000
        
    #print('reinforcement constribution', Vs)
    
    Vt = Vc + Vs
    
    return Vt


def get_moment_strength(test_data):
    '''
    Computation of the moment strength using Restrepo Equations
    
    '''
    
    # Get material properties
    fpc = test_data['fpc']   # (MPa)
    fyl = test_data['fyl']   # (MPa)
    fyt = test_data['fyt']   # (MPa)
    
    # Get geometry parameters
    diam = test_data['diam']  # (mm)
    length = test_data['l']   # (mm)
    
    dlb = test_data['dlb']    # (mm)
    nlb = test_data['nlb']    # (mm)
    
    dsp = test_data['dsp']    # (mm)
    s = test_data['s']        # (mm)

    # Get axial load
    p = test_data['axl'] * 1000   # (N)
    
    # Compute extra parameters
    d = 0.8 * diam     
    ag = np.pi * (diam) ** 2 / 4   # (mm2)
    
    al = np.pi * nlb / 2 * dlb ** 2 / 4 # (mm2)
    
    
    # Computations
    xc = diam * (0.32 * p / (ag * fpc) + 0.1)  # distance to neutral axis (mm)
    rhol = al / ag  # longitudinal reinf. ratio (-)
    
    a1 = rhol * fyl / fpc * (0.23 + 1/3 * (1/2 - xc/diam)) # nd param
    a2 = p / (ag * fpc) * (1/2 - xc/diam)
    
    mcd = np.pi / 4 * (1.15 * a1 + a2)
    
    Mcd = mcd * fpc * diam ** 3
    
    Vm = (Mcd / length) / 1000
    
    
    return Vm


# Run the script
if __name__ == '__main__':
    
    spiral_cols = [
        'id', 'name', 'type', 'testcf', 'pd', 'ft', 'axl', 'diam', 'l',
        'fpc', 'fyl', 'fsul', 'fyt', 'fsut', 'dlb', 'nlb',
        'cc', 'dsp', 's'
        ]
    
    data_rect = pd.DataFrame()
    data_spiral = pd.DataFrame(columns=spiral_cols)
    
    # For all tests...
    sp_ii = 0
    for testID in range(1, 417):
    
        # (1) Load json file to dictionary
        current_dir = os.getcwd()
        json_dir = current_dir + '/test_data/test_' + str(testID).zfill(3) +'.json'

        # json_dir = r'/Users/miguelgomez/Documents/GitHub/RC_Column_Model/test_data/test_' + str(testID).zfill(3) +'.json'
        rawdata = load_json(json_dir)
        
        # (2) Check if its spiral
        test_type = rawdata['Type']
        
        # (3) Turn dict into list with properties
        if test_type == 'Spiral':
            props = get_spiral_props(rawdata, testID)
            hysteresis = rawdata['data']
            
            data_spiral.loc[len(data_spiral)] = props
    
    # Store dataframe into a csv file
    # data_spiral.to_csv('spiral_data.csv')
    
    ndparams, data_spiral_wnd = get_nd_params_sp(data_spiral)
    #data_spiral_wnd['testcf'] = pd.Categorical(data_spiral_wnd['testcf']).codes
    print(pd.Categorical(data_spiral_wnd['testcf']).categories)

    data_spiral_wnd['ft'] = pd.Categorical(data_spiral_wnd['ft']).codes
    
    # Do pairplot
    sns.pairplot(data_spiral_wnd[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'ft', 'testcf']], hue='testcf')
    
    plt.show()

    # Store dataframe with the newly added columns
    # data_spiral_wnd.to_csv('data_spiral_wnd.csv')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    