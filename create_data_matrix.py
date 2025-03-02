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


def do_pairplot(dataframe, subset, hue):
    # Define tex type
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    sns.pairplot(dataframe[subset], hue=hue, height=1.5, plot_kws={'alpha':0.5})

    # Use latex for the labels
    plt.show()

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
        - The second one contains the original test data plus the nondimensional parameters as additional columns
    
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
        # ---------------------- Spiral Column -----------------------------
        # (1) Aspect Ratio
        ar = 1 / (test_data['diam'] / test_data['l'])
            
        # (2) Longitudinal Reinforcement Ratio
        ag = np.pi * (test_data['diam']) ** 2 / 4   # gross area (mm2)
        alr = test_data['nlb'] * np.pi * (test_data['dlb']) ** 2 / 4 # long. reinf. area (mm2)
        rhol = alr / ag
        
        lrr = rhol * (test_data['fyl'] / test_data['fpc'])
        
        # (3) Transverse Reinforcement Ratio
        '''d_core = test_data['diam'] - 2 * test_data['cc']   # diameter of the core (mm)
        asp = np.pi * test_data['dsp'] ** 2 / 4  # area of spiral reinf (mm2)
        rhosp = 4 * asp / (np.pi * d_core ** 2)'''

        asv = np.pi * test_data['dsp'] ** 2 / 2  # area of two legs of spiral reinf (mm2)
        rhot = asv / (test_data['diam'] * test_data['s'])  # transverse reinforcement ratio in the region of close spacing of stirrups
        
        if test_data['fyt'] == 0:
            # fyt = 0, then use fyl
            trr = rhot * (test_data['fyl'] / test_data['fpc'])
        else:
            trr = rhot * (test_data['fyt'] / test_data['fpc'])
        
        # (4) Axial Load Ratio
        ax_cap = test_data['fpc'] * ag / 1000    # kN
        alr = test_data['axl'] / ax_cap

        # (5) Transverse spacing ratio
        if test_data['s'] == 0:
            tsr = 2.0
        else:
            tsr = test_data['s'] / (6 * test_data['dsp'])

        # (6) Estimate of the shear strength and the moment strength
        Vs = get_shear_strength(test_data)
        Vp = get_moment_strength(test_data)
        
        vpvs = Vp/Vs

    else:
        # If the column is rectangular

        # (1) Aspect ratio (as defined by the concrete people)
        ar = 1 / (test_data['d'] / test_data['l'])
        
        # (2) Longitudinal reinforcement ratio
        ag = test_data['w'] * test_data['d']
        
        # The area of the longitudinal reinforcement is the area of the bars times the number of bars
        alr = test_data['nlb'] * np.pi * (test_data['dlb']) ** 2 / 4
        rhol = alr / ag

        lrr = rhol * (test_data['fyl'] / test_data['fpc'])

        # (3) Transverse reinforcement ratio
        # Area of the transverse reinforcement is the number of shear legs times
        # the area of the transverse bars in the region of close spacing
        asv = test_data['nsl'] * np.pi * (test_data['dtb_rcs']) ** 2 / 4
        rhot = asv / (test_data['w'] * test_data['s_rcs'])

        if test_data['fyt'] == 0:
            # fyt = 0, then use fyl
            trr = rhot * (test_data['fyl'] / test_data['fpc'])   # CHECK THIS!!!
        else:
            trr = rhot * (test_data['fyt'] / test_data['fpc'])
        
        # (4) Axial load ratio
        ax_cap = test_data['fpc'] * ag / 1000   # kN
        alr = test_data['axl'] / ax_cap

        # (5) Transverse spacing ratio
        if test_data['s_rcs'] == 0:
            tsr = 2.0
        else:
            tsr = test_data['s_rcs'] / (6 * test_data['dtb_rcs'])

        # (6) Estimate of the shear strength and the moment strength
        Vs = get_shear_strength(test_data)
        Vp = get_moment_strength(test_data)

        vpvs = Vp / Vs

    return [ar, lrr, trr, alr, tsr, vpvs]


def get_shear_strength(test_data):
    '''
    Calculation of the shear strength based on the equations of Sezen and Moehle (2004)
    
    '''

    # Get material properties
    fpc = test_data['fpc']   # (MPa)
    fyl = test_data['fyl']   # (MPa)
    fyt = test_data['fyt']   # (MPa)
    
    # Get geometry parameters
    if test_data['type'] == 'Spiral':
        d = test_data['diam']  # (mm)
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
    
    # k factor in Sezen and Moehle (2004) for ductility-displacement dependence of the shear strength
    k = 0.9

    # Compute a/d ratio
    a_dratio = length / d
    if a_dratio > 4.0:
        a_dratio = 4.0
    elif a_dratio < 2.0:
        a_dratio = 2.0

    if test_data['type'] == 'Spiral':
        # Gross area of the cross section
        ag = np.pi * (d) ** 2 / 4   # (mm2)

        # Mean shear stress at the onset of shear crack
        vc = 0.5 * np.sqrt(fpc) / (a_dratio) * np.sqrt(1 + p / (0.5 * np.sqrt(fpc) * ag))
        nsl = 2

    else:
        # Gross area of the cross section
        ag = w * d  # (mm2)

        # Mean shear stress at the onset of shear crack
        vc = 0.5 * np.sqrt(fpc) / (a_dratio) * np.sqrt(1 + p / (0.5 * np.sqrt(fpc) * ag))
        nsl = test_data['nsl']
        
    # Concrete contribution to shear strenght
    Vc = vc * 0.8 * ag / 1000
    
    # Transverse reinforcement contribution to the shear strength
    #print('concrete contribution', Vc)
    
    av = nsl * np.pi * (dtb) ** 2 / 4
    
    if s == 0:
        Vs = 0
    else:
        if fyt == 0:
            Vs = k * (av * fyl * d / s) / 1000
        else:
            Vs = k * (av * fyt * d / s) / 1000
            
    #print('reinforcement constribution', Vs)
    Vt = Vc + Vs
    
    return Vt


def get_moment_strength(test_data, props='expected'):
    '''
    Computation of the probable moment strength
    Ref: Restrepo and Rodriguez (2013)

    Props can be either nominal or expected. Use nominal for design values, use expected when using for 
    experimental data, where the values of concrete and steel strength were obtaied from measurements.
    
    '''
    
    if props == 'expected':
        # If using expected properties
        lam_h = 1.15
        lam_co = 1.0
    else:
        # If using nominal properties
        lam_h = 1.25
        lam_co = 1.7


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
        al = np.pi * nlb * (dlb) ** 2 / 4 # (mm2)
        
        # Computations
        xc = diam * (0.32 * p / (lam_co * ag * fpc) + 0.1)  # distance to neutral axis (mm)
        rhol = al / ag  # longitudinal reinf. ratio (-)
        
        a1 = rhol * fyl / fpc * (0.23 + 1/3 * (1/2 - xc/diam)) # nd param
        a2 = p / (ag * fpc) * (1/2 - xc/diam)
        
        mcd = np.pi / 4 * (lam_h * a1 + a2)
        Mcd = mcd * fpc * diam ** 3
        
        Vpr = (Mcd / length) / 1000

    else:
        # If the column is rectangular

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
        
        al = np.pi * nlb * (dlb ** 2 / 4) # (mm2)
        
        # Computations
        xc = d * (0.34 * p / (lam_co * ag * fpc) + 0.07)  # distance to neutral axis (mm)
        rhol = al / ag  # longitudinal reinf. ratio (-)
        
        a1 = rhol * fyl / fpc * (0.30 + 1/4 * (1/2 - xc/d)) # nd param
        a2 = p / (ag * fpc) * (1/2 - xc/d)
        
        mcd = lam_h * a1 + a2
        Mcd = mcd * fpc * w * d ** 2
        
        Vpr = (Mcd / length) / 1000
    
    return Vpr


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
    ndparams_rect   = pd.DataFrame(columns=['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr'])

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
                data_rect.loc[len(data_rect)] = props
                
                # Get nondimensional parameters for the rectangular column
                ndparams_ii = get_nd_params(data_rect.loc[len(data_rect)-1])

                # Append at the end of the dataframe
                ndparams_rect.loc[len(ndparams_rect)] = ndparams_ii

        except Exception as que_paso:

            print('Error in test', testID, 'Y', que_paso)

            continue
    

    # Read the cals_dr_005.csv file (use or don't use)
    cals_dr_005 = pd.read_csv('cals_dr_005_use.csv')
    print(cals_dr_005.columns)
    
    # Extract rows where cals_dr_005 in type are Rectangular
    use_rect_data = cals_dr_005[cals_dr_005['type'] == 'Rectangular']
    use_spiral_data = cals_dr_005[cals_dr_005['type'] == 'Spiral']

    use_rect_data = use_rect_data['use']
    use_spiral_data = use_spiral_data['use']
    
    use_rect_data = use_rect_data.reset_index(drop=True)
    use_spiral_data = use_spiral_data.reset_index(drop=True)

    print(use_spiral_data)

    # Check that use_rect data and data_rect have the same length
    print(len(use_rect_data), len(data_rect))
    print(len(use_spiral_data), len(data_spiral))

    # Add the use column to the dataframes
    data_rect['use'] = use_rect_data
    data_spiral['use'] = use_spiral_data

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
    
    # RELEVANT!! Drop rows where the use column is 0
    data_spiral_wnd = data_spiral_wnd[data_spiral_wnd['use'] == 1]
    data_rect_wnd = data_rect_wnd[data_rect_wnd['use'] == 1]

    # Do pairplot for spiral columns
    do_pairplot(data_spiral_wnd, ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'ft'], 'ft')
    
    # Do pairplot for rectangular columns
    do_pairplot(data_rect_wnd, ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'ft'], 'ft')

    # Store dataframe with the newly added columns
    # data_spiral_wnd.to_csv('data_spiral_wnd.csv')
    # data_rect_wnd.to_csv('data_rect_wnd.csv')