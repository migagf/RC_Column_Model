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


def plot_hysteresis(disp, force, title):
    
    plt.figure(dpi=200)
    plt.plot(disp, force)
    plt.title(title)
    
    
    pass


def load_json(json_dir):
    # Open JSON file and store as dictionary
    
    with open(json_dir, 'r') as file:
        data = json.load(file)
        
    return data


if __name__ == "__main__":
    '''
    This code plots the force-displacement relations for the concrete column
    tests in the database
    
    '''
    # Folder with the JSON files
    json_dir = r'/Users/miguelgomez/Documents/GitHub/RC_Column_Model/test_data/'
    
    # Load the database
    data = pd.read_csv('data_spiral_wnd.csv')
    
    # For each curve in the spiral database, plot the hysteresis, and show
    # the failure mode
    
    for ii in range(0, len(data)):
        
        # Create name of file
        filename = json_dir + 'test_' + str(data.id[ii]).zfill(3) + '.json'
        
        # Import JSON file as dictionary
        force_disp = load_json(filename)['data']
        failure_type = load_json(filename)['FailureType']
        pdelta = load_json(filename)['P_Delta']
        testconfig = load_json(filename)['TestConfiguration']
        title = failure_type + '--' + pdelta + '--' + testconfig
        
        plot_hysteresis(force_disp['disp'], force_disp['force'], title)
        
        