# Utilities for the cyclic puyshover analysis
import numpy as np
import matplotlib.pyplot as plt

'''
This file contains the utility functions for the cyclic pushover analysis
'''

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
            plt.plot(disp, effective_force, label='Effective Force')
            plt.plot(disp, force, label='Original Force')
            plt.legend()
            plt.title(test_data['P_Delta'])
            plt.show()

    else:
        # Feff directly reported. Do nothing
        print('Feff directly reported, no need to compute effective force')
        pass

    return test_data['data']


def get_elastic_properties(test_data):
    
    # Concrete strength and stiffness
    fpc = test_data["fpc"]  # MPa
    E = 4_700 * np.sqrt(fpc) # MPa
    
    # Define a section type
    section_type = test_data["Type"]
    
    # Geometry of the column
    L = test_data["L_Inflection"] # mm
    
    if section_type == "Rectangular":
        
        b = test_data["Width"] # mm
        h = test_data["Depth"] # mm
        I = b * h ** 3 / 12
        
    else:
        
        d = test_data["Diameter"]
        I = np.pi * d ** 4 / 64 # mm^4
    
    return E, I, L


def save_response(filename, array, save_type='row'):
    ''' 
    Function to save to csv file
    '''
    
    if save_type == 'row':
        # Save array as a row file
        array.tofile(filename, sep=',', format='%10.5f')
        
    else:
        # Save array as a column file
        array.tofile(filename, sep='\n', format='%10.5f')
        
    pass


def interpolator(original_array, new_length):
    '''
    Function to interpolate an array to a new length

    '''
    # Get the original length
    original_length = len(original_array)
    
    # Create the new array
    interpolated_array = np.interp(np.linspace(0, original_length, new_length), np.arange(original_length), original_array)

    return interpolated_array

