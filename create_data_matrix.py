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


def get_rect_props(rawdata, testID):
    '''
    Extract the properties of a rectangular column test and store them in a list

    '''
    props = []
    props.append(testID)
    props.append(rawdata['Name'])
    props.append(rawdata['Type'])
    props.append(rawdata['TestConfiguration'])
    props.append(rawdata['P_Delta'])
    props.append(rawdata['FailureType'])

    # Axial load
    props.append(rawdata['AxLoad'])

    # Overall Shape Data
    props.append(rawdata['Width'])
    props.append(rawdata['Depth'])
    props.append(rawdata['L_Inflection'])

    # Material Strength Properties
    props.append(rawdata['fpc'])
    props.append(rawdata['fyl'])
    props.append(rawdata['fsul'])

    props.append(rawdata['fyt'])
    props.append(rawdata['fsut'])

    # Reinforcement details
    props.append(rawdata['dlb'])
    props.append(rawdata['dlb_c'])
    props.append(rawdata['nlb'])
    props.append(rawdata['cc_per'])   # Clear cover in the direction perpendicular to the load
    props.append(rawdata['nib_per'])  # Number of intermediate bars in the direction perpendicular to the load
    props.append(rawdata['cc_par'])   # Clear cover in the direction parallel to the load
    props.append(rawdata['nib_par'])  # Number of intermediate bars in the direction parallel to the load

    props.append(rawdata['nsl'])      # Number of shear legs
    props.append(rawdata['dtb_rcs'])  # Diameter of the transverse bars in the region of close spacing of stirrups.
    props.append(rawdata['s_rcs'])    # Spacing of the transverse bars in the region of close spacing of stirrups.

    return props


def get_spiral_props(rawdata, testID):
    '''
    Extract the properties of a spiral column test and store them in a list.
    The final goal is to put them in a dataframe.
    '''

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


def get_nd_params(test_data):
    '''
    This function computes the nondimesional physical parameters
    for a reinforced concrete column.

    It returns two dataframes:
        - The first one contains the nondimensional parameters
        - The second one contains the original data plus the nondimensional parameters as additional columns
    
    For spiral columns, the properties are:
        'id', 'name', 'type', 'testcf', 'pd', 'ft', 'axl', 'diam', 'l',
        'fpc', 'fyl', 'fsul', 'fyt', 'fsut', 'dlb', 'nlb',
        'cc', 'dsp', 's'

    For rectangular columns, the properties are:
        'id', 'name', 'type', 'testcf', 'pd', 'ft', 'axl', 'w', 'd', 'l',
        'fpc', 'fyl', 'fsul', 'fyt', 'fsut', 'dlb', 'dlb_c', 'nlb',
        'cc_per', 'nib_per', 'cc_par', 'nib_par', 'nsl', 'dtb_rcs', 's_rcs'
    '''
    
    if test_data['type'] == 'Spiral':
            
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
            srr = test_data['fyl'] * rhosp / test_data['fpc']
        else:
            srr = test_data['fyt'] * rhosp / test_data['fpc']
            
        # (5) Axial Load Ratio
        ax_cap = test_data['fpc'] * ag / 1000    # kN
        
        alr = test_data['axl'] / ax_cap
        
        # (6) Spiral diameter ratio
        if test_data['s'] == 0:
            sdr = 10
        else:
            sdr = (6 * test_data['dsp']) / test_data['s']
        
        # (7) Estimate of the shear strength and the moment strength
        Vs = get_shear_strength(test_data)
        Vm = get_moment_strength(test_data)
        
        smr = Vm/Vs

    else:
        
        # (2) Aspect Ratio
        ar = test_data['d'] / test_data['l']
        
        # (3) Longitudinal Reinforcement Ratio
        ag = test_data['w'] * test_data['d']
        
        # The area of the longitudinal reinforcement is the area of the bars times the number of bars
        alr = test_data['nlb'] * np.pi * (test_data['dlb']) ** 2 / 4
        rhol = alr / ag

        lrr = test_data['fyl'] * rhol / test_data['fpc']

        # (4) Spiral Reinforcement Ratio
        d_core = test_data['d'] - 2 * test_data['cc_per']

        # Area of the transverse reinforcement is the number of shear legs times
        #  the area of the transverse bars in the region of close spacing
        atr = test_data['nsl'] * np.pi * (test_data['dtb_rcs']) ** 2 / 4
        rhot = atr / (test_data['w'] * test_data['s_rcs'])

        if test_data['fyt'] == 0:
            # fyt = 0, then use fyl
            srr = test_data['fyl'] * rhot / test_data['fpc']   # CHECK THIS!!!
        else:
            srr = test_data['fyt'] * rhot / test_data['fpc']
        
        # (5) Axial Load Ratio
        ax_cap = test_data['fpc'] * ag / 1000   # kN
        alr = test_data['axl'] / ax_cap

        # (6) Spiral diameter ratio
        if test_data['s_rcs'] == 0:
            sdr = 10
        else:
            sdr = (6 * test_data['dtb_rcs']) / test_data['s_rcs']

        # (7) Estimate of the shear strength and the moment strength
        Vs = get_shear_strength(test_data)
        Vm = get_moment_strength(test_data)

        smr = Vm / Vs

    return [ar, lrr, srr, alr, sdr, smr]


def get_shear_strength(test_data):
    '''
    Calculation of the shear strength based on the equations of Sezen and Moehle (2004)
    (Only working for spiral columns right now...)

    '''
    
    # Get material properties
    fpc = test_data['fpc']   # (MPa)
    fyl = test_data['fyl']   # (MPa)
    fyt = test_data['fyt']   # (MPa)
    
    # Get geometry parameters
    if test_data['type'] == 'Spiral':
        diam = test_data['diam']  # (mm)
    else:
        w = test_data['w']
        d = test_data['d']

    length = test_data['l']   # (mm)
        
    if test_data['type'] == 'Spiral':
        dtb = test_data['dsp']    # (mm)
        s = test_data['s']        # (mm)
    else:
        # Get properties in the region of close spacing of stirrups
        dtb = test_data['dtb_rcs']
        s = test_data['s_rcs']

    # Get axial load
    p = test_data['axl'] * 1000   # (N)
    
    if test_data['type'] == 'Spiral':
        # Compute extra parameters
        d = 0.8 * diam
        ag = np.pi * (diam) ** 2 / 4   # (mm2)

        # Mean shear stress at the onset of shear crack
        vc = 0.5 * np.sqrt(fpc) / (length / diam) * np.sqrt(1 + p / (0.5 * np.sqrt(fpc) * ag)) * 0.8
        nsl = 2

    else:
        d = 0.8 * d
        ag = w * d

        # Mean shear stress at the onset of shear crack
        vc = 0.5 * np.sqrt(fpc) / (length / d) * np.sqrt(1 + p / (0.5 * np.sqrt(fpc) * ag)) * 0.8
        nsl = test_data['nsl']
        
    # Concrete contribution to shear strenght
    Vc = vc * ag / 1000
    
    # Transverse reinforcement contribution to the shear strength
    #print('concrete contribution', Vc)
    
    av = nsl * np.pi * (dtb) ** 2 / 4
    
    if s == 0:
            Vs = 0
    else:
        if fyt == 0:
            Vs = (av * fyl * d / s) / 1000
        else:
            Vs = (av * fyt * d / s) / 1000
            
    #print('reinforcement constribution', Vs)
    Vt = Vc + Vs
    
    return Vt


def get_moment_strength(test_data):
    '''
    Computation of the moment strength using Restrepo Equations
    
    '''
    if test_data['type'] == 'Spiral':
        # Get material properties
        fpc = test_data['fpc']   # (MPa)
        fyl = test_data['fyl']   # (MPa)
        
        # Get geometry parameters
        diam = test_data['diam']  # (mm)
        length = test_data['l']   # (mm)
        
        dlb = test_data['dlb']    # (mm)
        nlb = test_data['nlb']    # (mm)

        # Get axial load
        p = test_data['axl'] * 1000   # (N)
        
        # Compute extra parameters 
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

    else:
        # Get material properties
        fpc = test_data['fpc']   # (MPa)
        fyl = test_data['fyl']   # (MPa)
        
        # Get geometry parameters
        d = test_data['d']  # (mm)
        w = test_data['w']  # (mm)
        length = test_data['l']   # (mm)
        
        dlb = test_data['dlb']    # (mm)
        nlb = test_data['nlb']    # (mm)

        # Get axial load
        p = test_data['axl'] * 1000   # (N)
        
        # Compute extra parameters
        ag = w * d   # (mm2)
        
        al = np.pi * (nlb / 2) * dlb ** 2 / 4 # (mm2)
        
        # Computations
        xc = d * (0.34 * p / (ag * fpc) + 0.07)  # distance to neutral axis (mm)
        rhol = al / ag  # longitudinal reinf. ratio (-)
        
        a1 = rhol * fyl / fpc * (0.30 + 1/4 * (1/2 - xc/d)) # nd param
        a2 = p / (ag * fpc) * (1/2 - xc/d)
        
        mcd = 1.15 * a1 + a2
        
        Mcd = mcd * fpc * w * d ** 2
        
        Vm = (Mcd / length) / 1000
    
    return Vm


# Run the script
if __name__ == '__main__':
    
    # Properties of spiral columns
    spiral_cols = [
        'id', 'name', 'type', 'testcf', 'pd', 'ft', 'axl', 'diam', 'l',
        'fpc', 'fyl', 'fsul', 'fyt', 'fsut', 'dlb', 'nlb',
        'cc', 'dsp', 's'
        ]
    
    # Properties of rectangular columns
    rect_cols = [
        'id', 'name', 'type', 'testcf', 'pd', 'ft', 'axl', 'w', 'd', 'l',
        'fpc', 'fyl', 'fsul', 'fyt', 'fsut', 'dlb', 'dlb_c', 'nlb',
        'cc_per', 'nib_per', 'cc_par', 'nib_par', 'nsl', 'dtb_rcs', 's_rcs'
    ]
    
    # Create dataframes for rectangular and spiral columns
    data_rect = pd.DataFrame(columns=rect_cols)
    data_spiral = pd.DataFrame(columns=spiral_cols)
    
    # For all tests...
    sp_ii = 0

    # Create dataframe for nondimensional parameters (actually, can save this only and make things easier)
    ndparams_spiral = pd.DataFrame(columns=['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr'])
    ndparams_rect = pd.DataFrame(columns=['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr'])

    for testID in range(1, 417):
    
        # (1) Load json file to dictionary
        current_dir = os.getcwd()
        json_dir = current_dir + '/test_data/test_' + str(testID).zfill(3) +'.json'

        # json_dir = r'/Users/miguelgomez/Documents/GitHub/RC_Column_Model/test_data/test_' + str(testID).zfill(3) +'.json'
        rawdata = load_json(json_dir)
        
        # (2) Check if its spiral
        test_type = rawdata['Type']
        
        # (3) Turn dict into list with properties
        try:
            if test_type == 'Spiral':
                # Get the properties of the spiral column in a list
                props = get_spiral_props(rawdata, testID)
                
                # Get the hysteresis data
                # hysteresis = rawdata['data']
                # Append the properties to the dataframe
                data_spiral.loc[len(data_spiral)] = props
                
                # Get nondimensional parameters for the spiral column
                ndparams_ii = get_nd_params(data_spiral.loc[len(data_spiral)-1])
                # Append at the end of the dataframe
                ndparams_spiral.loc[len(ndparams_spiral)+1] = ndparams_ii

            else:
                # Get the properties of the rectangular column in a list
                props = get_rect_props(rawdata, testID)

                # Get the hysteresis data
                # hysteresis = rawdata['data']
                # Append the properties to the dataframe
                data_rect.loc[len(data_rect)] = props
                
                # Get nondimensional parameters for the rectangular column
                ndparams_ii = get_nd_params(data_rect.loc[len(data_rect)-1])
                # Append at the end of the dataframe
                ndparams_rect.loc[len(ndparams_rect)] = ndparams_ii

        except Exception as que_paso:
            print('Error in test', testID, 'Y', que_paso)
            continue
    
    
    # Store dataframe into a csv file
    data_spiral.to_csv('spiral_data.csv')
    data_rect.to_csv('rect_data.csv')

    # Add the nondimensional parameters to the data_spiral dataframe
    data_spiral_wnd = pd.concat([data_spiral, ndparams_spiral], axis=1)

    # Add the nondimensional parameters to the data_rect dataframe
    data_rect_wnd = pd.concat([data_rect, ndparams_rect], axis=1)

    #data_spiral_wnd['testcf'] = pd.Categorical(data_spiral_wnd['testcf']).codes
    print(pd.Categorical(data_spiral_wnd['testcf']).categories)

    # Get the nondimensional parameters for the spiral columns
    data_spiral_wnd['ft'] = pd.Categorical(data_spiral_wnd['ft']).codes
    data_rect_wnd['ft'] = pd.Categorical(data_rect_wnd['ft']).codes
    
    # Do pairplot for spiral columns
    sns.pairplot(data_spiral_wnd[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'ft', 'testcf']], hue='ft')
    plt.show()

    sns.pairplot(data_rect_wnd[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'ft', 'testcf']], hue='ft')
    plt.show()

    # Store dataframe with the newly added columns
    data_spiral_wnd.to_csv('data_spiral_wnd.csv')
    data_rect_wnd.to_csv('data_rect_wnd.csv')

    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    