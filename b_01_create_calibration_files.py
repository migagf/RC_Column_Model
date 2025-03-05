import os
import pandas as pd
import numpy as np

from database_processing_functions import *
from a_02_create_data_matrix import *

'''
This code plots the force-displacement relations for the concrete column
tests in the database
'''

# Get current folder
current_folder = os.getcwd()

# Folder with the JSON files
json_dir = current_folder + '/test_data/'
   
# Load the database
data = pd.read_csv('merged_data.csv')
# print(data)
# For each curve:    
maxii = len(data)
#selId = int(input('Select test id (0 to ' + str(maxii) + '): '))

# Restart in 15
selii = 1
for ii in range(selii, selii+1):
    
    # (1) Create name of file
    print(ii)
    test_id = str(int(data.UniqueId[ii])).zfill(3)
    filename = json_dir + 'test_' + test_id + '.json'
    print(filename)
    
    # (2) Import JSON file as dictionary
    test_data = load_json(filename)

    # (3) Check P-Delta, and get effecttive force if needed
    test_data['data'] = get_effective_force(test_data, False)

    # (4) Create the calibration file
    state, cal_data, run_data = create_calibration_file(test_data, test_id, destination=json_dir, plot=True)

    # Add run_data and cal_data into the dictionary
    test_data['run_data'] = run_data
    test_data['cal_data'] = cal_data

    # (5) Save the dictionary as a JSON file
    save_json(test_data, filename)

    print('---')

    # (5) Extract the nondimensional parameters from the csv file (last 6 columns)
    nondim_params = data.iloc[ii, -8::]
    # print(nondim_params)

    # Add a column to data
    data['stiff_type'] = np.ones(len(data))
    # These cases have a different stiffness type, since they were tested to
    # a large displacement from the initial cycle.
    cases = [267, 269, 270, 271, 272, 273]
    # For ids in cases, set the stiff_type to 0
    data.loc[data['UniqueId'].isin(cases), 'stiff_type'] = 0

    # Get last 7 values in data in a new dataframe
    nondim_params = data.iloc[:, -8::]
    

# Create pairplot of the data withh hue='stiff_type'
import seaborn as sns
sns.pairplot(nondim_params, hue='stiff_type')


# Note: cases are in the median of the distributions for all the possible parameters... Not too good.

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